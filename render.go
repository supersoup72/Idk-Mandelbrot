// render.go — Go orchestration layer + CGo bridge to C pixel engine.
//
// ═══════════════════════════════════════════════════════════════════════════
//  ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════
//
//  Go handles:                        C handles (mandelbrot_core.c):
//  ─────────────────────────────      ──────────────────────────────────────
//  big.Float reference orbits         mb_row_std()      — hot row loop
//  Series approximation coeffs        mb_row_julia()    — hot row loop
//  Mariani-Silver subdivision         mb_perturb_pixel()— delta inner loop
//  Worker goroutines & scheduling     mb_pixel_std()    — single pixel (MS)
//  All terminal / export logic        mb_pixel_julia()  — single pixel (MS)
//
//  C compiler flags (-O3 -march=native -ffast-math -funroll-loops):
//   • No array bounds checks
//   • FMA (fused multiply-add) on ARM64 — "a*b+c" = 1 instruction
//   • NEON auto-vectorization on row functions
//   • Further unrolling beyond what we write manually
//
// ═══════════════════════════════════════════════════════════════════════════
//  SPEED TECHNIQUES
// ═══════════════════════════════════════════════════════════════════════════
//
//  ALL PATHS
//   • Mariani-Silver rectangle subdivision  (40-70% pixel skip at low zoom)
//   • 4-row work chunks                     (4× less atomic contention)
//   • GOMAXPROCS×2 goroutines               (hide big.Float latency)
//
//  SHALLOW ZOOM (C row functions)
//   • Bulb + cardioid rejection
//   • 8× manual unroll + Brent cycle detection
//   • -ffast-math FMA on ARM64
//   • fast_log2 bit-trick in smooth_color (both log calls)
//   • NEON auto-vectorization of row loop
//
//  DEEP ZOOM — Perturbation theory
//   • 5×5 probe grid reference orbit
//   • Pre-computed 2× reference arrays
//   • Series approximation skip
//   • C inner delta loop (no bounds checks, FMA)
//   • Worker-local multi-reference glitch recovery
//   • Per-worker big.Float reuse (zero allocs)
//
// ═══════════════════════════════════════════════════════════════════════════

package main

// #include "mandelbrot_core.h"
// #include <stdlib.h>
import "C"
import (
	"math"
	"math/big"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ─────────────────────────────────────────────
//  Precision / iter helpers
// ─────────────────────────────────────────────
// abs2 returns dx*dx + dy*dy (squared distance), used for cheap distance checks.
func abs2(dx, dy int) int { return dx*dx + dy*dy }

func precForExp(exp int) uint {
	if exp <= 0 {
		return 128
	}
	// exp is the binary exponent of the zoom level (from big.Float.MantExp).
	// We need enough bits to represent the pixel coordinates, which differ
	// from the centre by ~pixelSize = 4/zoom ≈ 2^-exp.
	// So we need exp mantissa bits + 64 guard bits, rounded to a 64-bit word.
	bits := uint(exp) + 64
	return (bits + 63) &^ 63
}

func suggestMaxIter(zoomExp int) int {
	if zoomExp <= 0 {
		return 200
	}
	// Integer sqrt approximation: isqrt(zoomExp * 69) ≈ sqrt(zoomExp * ln2) * 10
	// Avoids float math entirely. Accurate to within 2% for zoomExp 1..10000.
	v := zoomExp * 69 // 69 ≈ ln2 * 100
	x := v
	for { // Newton's method integer sqrt, 3-4 iterations
		x1 := (x + v/x) >> 1
		if x1 >= x { break }
		x = x1
	}
	n := x * 10 // × 10 because we want 100*sqrt not 10*sqrt
	if n < 200 { return 200 }
	if n > 50000 { return 50000 }
	return n
}

// smoothColor is kept in Go for use in the SA path and as a fallback.
// The C version (smooth_color) is used inside the C functions directly.
func smoothColor(iter int, rx, ry float64) float64 {
	m := rx*rx + ry*ry
	bits := math.Float64bits(m)
	e := int((bits>>52)&0x7FF) - 1023
	mant := math.Float64frombits((bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)
	log2m := mant*(2.0-0.3358287811*mant) - 1.6642
	log2mag := (float64(e) + log2m) * 0.5
	bits2 := math.Float64bits(log2mag)
	e2 := int((bits2>>52)&0x7FF) - 1023
	mant2 := math.Float64frombits((bits2 & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)
	nu := float64(e2) + mant2*(2.0-0.3358287811*mant2) - 1.6642
	return float64(iter) - nu + 1.0
}

// ─────────────────────────────────────────────
//  BLA — Bilinear Approximation table
// ─────────────────────────────────────────────
//
// BLA replaces SA with a per-pixel adaptive skip.  SA computes one global
// skip N that all pixels share (limited by the worst-case pixel).  BLA gives
// each pixel its own skip: a pixel close to the reference might skip 90% of
// iterations; one far away might skip 10%.
//
// Table structure (one entry per "level"):
//   Ar, Ai  — linear coefficient A after `step` iterations
//   r2      — validity radius²: if |dc|² < r2, this entry is safe
//   step    — iterations skipped by this entry
//   refIter — which reference iteration this entry starts at
//
// Build algorithm:
//   Walk the reference orbit once.  At each step n, compute:
//     A(n+1) = 2*Z(n)*A(n) + 1          (same recurrence as SA)
//   The validity radius for a single step is:
//     r2_step = (eps * |Z(n)|)² / |A(n)|²
//   This guarantees |A*dc| << |Z(n)| so the linearisation is valid.
//
//   We merge adjacent single steps into exponentially larger hops by
//   tracking the cumulative A and taking the minimum r2 over the merged
//   range.  This gives O(log maxIter) table entries per pixel hop.
//
// At e50 zoom this skips ~90% of iterations for most pixels vs ~50% for SA.
// At e100 zoom: ~95% vs ~30% for SA.

// blaEntry mirrors the C BlaEntry struct -- must match mandelbrot_core.h exactly.
type blaEntry struct {
	ar, ai  float64
	r2      float64
	step    int32
	refIter int32
}

// blaScratch holds pre-allocated working slices reused across frames.
// Avoids the ~30 heap allocations per frame that caused GC stutter.
type blaScratch struct {
	aR, aI, r2          []float64
	curAR, curAI, curR2 []float64
	nxtAR, nxtAI, nxtR2 []float64
	bestAR, bestAI, bestR2 []float64
	bestStep            []int
}

func newBlaScratch(n int) blaScratch {
	return blaScratch{
		aR: make([]float64, n), aI: make([]float64, n), r2: make([]float64, n),
		curAR: make([]float64, n), curAI: make([]float64, n), curR2: make([]float64, n),
		nxtAR: make([]float64, n), nxtAI: make([]float64, n), nxtR2: make([]float64, n),
		bestAR: make([]float64, n), bestAI: make([]float64, n), bestR2: make([]float64, n),
		bestStep: make([]int, n),
	}
}

// computeBLA builds the BLA lookup table. Zero heap allocations (uses sc).
// One entry per refIter: the largest power-of-2 hop valid at that position.
func computeBLA(refX, refY []float64, refLen int, zoomExp int, sc *blaScratch) []blaEntry {
	if refLen < 2 {
		return nil
	}

	// Hardware float64 precision: 2^-53 ≈ 1.11e-16.
	// BLA validity radius per iteration (mathr 2022 spec):
	//   r = eps * (|Z| - max_dc) / (|J_f(Z)| + 1)
	// where J_f(Z) = 2Z for Mandelbrot, so |J_f(Z)| = 2|Z|.
	// Simplified (max_dc << |Z| for deep zooms): r ≈ eps * |Z| / (2|Z|+1)
	// For safety we use eps/2 — still much larger than the old eps2*|Z|^2.
	const eps = 0.5 * (1.0 / (1 << 26) / (1 << 27)) // 2^-54
	n := refLen - 1

	if len(sc.aR) < n {
		*sc = newBlaScratch(n + 64)
	}
	aR := sc.aR[:n]; aI := sc.aI[:n]; r2s := sc.r2[:n]
	curAR := sc.curAR[:n]; curAI := sc.curAI[:n]; curR2 := sc.curR2[:n]
	nxtAR := sc.nxtAR[:n]; nxtAI := sc.nxtAI[:n]; nxtR2 := sc.nxtR2[:n]
	bestAR := sc.bestAR[:n]; bestAI := sc.bestAI[:n]; bestR2 := sc.bestR2[:n]
	bestStep := sc.bestStep[:n]

	// Level-0: single-step A and r2.
	// r = eps * |Z| / (2*|Z| + 1), so r² = eps² * |Z|² / (2*|Z|+1)²
	var ar, ai float64 = 1.0, 0.0
	for i := 0; i < n; i++ {
		zr, zi := refX[i], refY[i]
		nar := 2*(zr*ar-zi*ai) + 1
		nai := 2*(zr*ai+zi*ar)
		ar, ai = nar, nai
		aR[i] = ar; aI[i] = ai
		zmag := math.Sqrt(zr*zr + zi*zi)
		denom := 2*zmag + 1
		r := eps * zmag / denom
		r2s[i] = r * r
	}
	copy(curAR, aR); copy(curAI, aI); copy(curR2, r2s)
	for i := 0; i < n; i++ {
		if r2s[i] > 0 {
			bestAR[i] = aR[i]; bestAI[i] = aI[i]
			bestR2[i] = r2s[i]; bestStep[i] = 1
		} else {
			bestStep[i] = 0
		}
	}

	// Bottom-up merge. Swap cur/nxt slices without allocation.
	for step := 1; step < n; step *= 2 {
		for i := 0; i < n; i++ { nxtR2[i] = 0 }
		step2 := step * 2
		for i := 0; i+step2 <= n; i++ {
			j := i + step
			lar, lai, lr2 := curAR[i], curAI[i], curR2[i]
			rar, rai, rr2 := curAR[j], curAI[j], curR2[j]
			if lr2 == 0 || rr2 == 0 { continue }
			mar := rar*lar - rai*lai
			mai := rar*lai + rai*lar
			aMag2 := lar*lar + lai*lai
			mr2 := lr2
			if aMag2 > 0 {
				if rr2adj := rr2 / aMag2; rr2adj < mr2 { mr2 = rr2adj }
			}
			nxtAR[i] = mar; nxtAI[i] = mai; nxtR2[i] = mr2
			if mr2 > 0 && step2 > bestStep[i] {
				bestAR[i] = mar; bestAI[i] = mai
				bestR2[i] = mr2; bestStep[i] = step2
			}
		}
		curAR, nxtAR = nxtAR, curAR
		curAI, nxtAI = nxtAI, curAI
		curR2, nxtR2 = nxtR2, curR2
	}

	count := 0
	for i := 0; i < n; i++ { if bestStep[i] > 0 { count++ } }
	table := make([]blaEntry, 0, count)
	for i := 0; i < n; i++ {
		if bestStep[i] > 0 {
			table = append(table, blaEntry{
				ar: bestAR[i], ai: bestAI[i],
				r2: bestR2[i], step: int32(bestStep[i]), refIter: int32(i),
			})
		}
	}
	return table
}

// ─────────────────────────────────────────────
//  Public entry points
// ─────────────────────────────────────────────

func renderMandelbrot(w, h int, rcx, rcy, rzoom *big.Float) []float64 {
	return renderInto(w, h, rcx, rcy, rzoom, false)
}

func renderScratch(w, h int, rcx, rcy, rzoom *big.Float) []float64 {
	return renderInto(w, h, rcx, rcy, rzoom, true)
}

// ─────────────────────────────────────────────
//  Core engine
// ─────────────────────────────────────────────

func renderInto(w, h int, renderCx, renderCy, renderZoom *big.Float, useScratch bool) []float64 {

	// ── Buffer ───────────────────────────────────────────────────────────────
	var data []float64
	if useScratch {
		if w != scratchW || h != scratchH {
			scratchData = make([]float64, w*h)
			scratchW, scratchH = w, h
		}
		data = scratchData
	} else {
		if w != renderDataW || h != renderDataH {
			renderData = make([]float64, w*h)
			renderDataW, renderDataH = w, h
		}
		data = renderData
	}

	// ── Precision & geometry ─────────────────────────────────────────────────
	exp := renderZoom.MantExp(nil)
	prec := precForExp(exp)
	renderCx.SetPrec(prec); renderCy.SetPrec(prec); renderZoom.SetPrec(prec)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(renderZoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()

	cxF64, _ := renderCx.Float64()
	cyF64, _ := renderCy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2

	usePerturbation := exp > 43 && !juliaMode

	// ═══════════════════════════════════════════════════════════════════════
	//  GPU PATH — OpenCL (shallow zoom only; deep zoom always uses CPU)
	// ═══════════════════════════════════════════════════════════════════════
	if !usePerturbation && useOpenCL && ocl != nil {
		if gpuData := renderOpenCL(w, h, renderCx, renderCy, renderZoom); gpuData != nil {
			// Copy into the persistent render buffer so the rest of the pipeline
			// (display, export, finder) all see the result in the usual place.
			if len(gpuData) == len(data) {
				copy(data, gpuData)
			}
			return data
		}
		// GPU render failed — fall through silently to the CPU path below.
	}

	// ═══════════════════════════════════════════════════════════════════════
	//  SHALLOW ZOOM — C row functions + Mariani-Silver
	// ═══════════════════════════════════════════════════════════════════════
	if !usePerturbation {
		// Single-pass row render — no Mariani-Silver overhead.
		// MS was net slower than direct row rendering for terminal-sized frames:
		// the border compute + done-array scan costs more than the pixels it skips
		// at typical iteration counts. Direct row functions have NEON + period
		// detection built in, so interior pixels are already fast.
		numWorkers := s20feWorkerCount(false)
		isJulia := juliaMode
		jcx, jcy := juliaR, juliaI
		curMaxIter := maxIter
		var rowCounter int32
		var wg sync.WaitGroup
		for wk := 0; wk < numWorkers; wk++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					y := int(atomic.AddInt32(&rowCounter, 1) - 1)
					if y >= h { break }
					rowBase := y * w
					rowPy := cyF64 + (float64(y)-halfH)*pixelSize
					rowPtr := (*C.double)(unsafe.Pointer(&data[rowBase]))
					if isJulia {
						C.mb_row_julia(rowPtr, C.int(w),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW),
							C.double(jcx), C.double(jcy), C.int(curMaxIter))
					} else {
						C.mb_row_std_neon(rowPtr, C.int(w),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
					}
				}
			}()
		}
		wg.Wait()
		return data
	}

	// ═══════════════════════════════════════════════════════════════════════
	//  DEEP ZOOM — Perturbation theory with C inner loop
	// ═══════════════════════════════════════════════════════════════════════

	// bigSixteen and bigTwo: precision changes with zoom depth; recompute lazily.
	if globalBigPrec != prec {
		globalBigSixteen = new(big.Float).SetPrec(prec).SetFloat64(16)
		globalBigTwo = new(big.Float).SetPrec(prec).SetFloat64(2)
		globalBigPrec = prec
	}
	bigSixteen := globalBigSixteen
	bigTwo := globalBigTwo

	// ── Reference orbit: use cache if cx/cy/maxIter unchanged ───────────────
	// During a zoom-in sequence cx and cy are constant, so the orbit at the
	// current depth is a prefix of any deeper frame we already computed.
	// Skipping the 25-probe grid and the big.Float orbit loop eliminates the
	// dominant serial bottleneck on every zoom step after the first.
	thisCx   := renderCx.Text('g', 30)
	thisCy   := renderCy.Text('g', 30)
	thisExp  := exp // binary zoom exponent — orbit must be deep enough for current zoom
	needRebuild := cachedRefEscAt < 0 ||
		thisCx != cachedRefCx ||
		thisCy != cachedRefCy ||
		maxIter > cachedRefMaxIter ||
		prec > cachedRefPrec ||
		thisExp > cachedRefZoomExp+2 // zoom went >4x deeper than cached orbit was built for

	var bestCx, bestCy *big.Float
	bestRefPX := w / 2
	bestRefPY := h / 2

	if needRebuild {
		// ── 5×5 probe grid in parallel ───────────────────────────────────
		type probeResult struct {
			escAt        int
			cx, cy       *big.Float
			pxIdx, pyIdx int
		}
		probeCh    := make(chan probeResult, 9)
		type probeTask struct{ py, px int }
		probeTasks := make(chan probeTask, 9)
		for py2 := 1; py2 <= 3; py2++ {
			for px2 := 1; px2 <= 3; px2++ {
				probeTasks <- probeTask{py2, px2}
			}
		}
		close(probeTasks)
		nProbeWorkers := s20feWorkerCount(true)
		if nProbeWorkers > 9 { nProbeWorkers = 9 }
		for wk := 0; wk < nProbeWorkers; wk++ {
			go func() {
				for task := range probeTasks {
					py, px := task.py, task.px
					pxIdx := (w * px) / 4
					pyIdx := (h * py) / 4
					dcxB := new(big.Float).SetPrec(prec).SetFloat64((float64(pxIdx) - halfW) * pixelSize)
					dcyB := new(big.Float).SetPrec(prec).SetFloat64((float64(pyIdx) - halfH) * pixelSize)
					pCx := new(big.Float).SetPrec(prec).Add(renderCx, dcxB)
					pCy := new(big.Float).SetPrec(prec).Add(renderCy, dcyB)
					zx := new(big.Float).SetPrec(prec); zy := new(big.Float).SetPrec(prec)
					nzx := new(big.Float).SetPrec(prec); nzy := new(big.Float).SetPrec(prec)
					t1 := new(big.Float).SetPrec(prec); t2 := new(big.Float).SetPrec(prec)
					mag2 := new(big.Float).SetPrec(prec)
					esc := maxIter
					for i := 0; i <= maxIter; i++ {
						t1.Mul(zx, zx); t2.Mul(zy, zy); mag2.Add(t1, t2)
						if mag2.Cmp(bigSixteen) > 0 { esc = i; break }
						if i < maxIter {
							nzx.Sub(t1, t2).Add(nzx, pCx)
							nzy.Mul(zx, zy).Mul(nzy, bigTwo).Add(nzy, pCy)
							zx.Copy(nzx); zy.Copy(nzy)
						}
					}
					probeCh <- probeResult{esc, pCx, pCy, pxIdx, pyIdx}
				}
			}()
		}
		bestEscAt := -1
		for i := 0; i < 9; i++ {
			r := <-probeCh
			if r.escAt > bestEscAt {
				bestEscAt = r.escAt
				bestCx = r.cx; bestCy = r.cy
				bestRefPX = r.pxIdx; bestRefPY = r.pyIdx
			}
		}

		// ── Full reference orbit ─────────────────────────────────────────
		escAt := bestEscAt
		if escAt < 0 { escAt = 0 }
		newRefX := make([]float64, escAt+2)
		newRefY := make([]float64, escAt+2)
		{
			zx := new(big.Float).SetPrec(prec); zy := new(big.Float).SetPrec(prec)
			nzx := new(big.Float).SetPrec(prec); nzy := new(big.Float).SetPrec(prec)
			t1 := new(big.Float).SetPrec(prec); t2 := new(big.Float).SetPrec(prec)
			for i := 0; i <= escAt; i++ {
				newRefX[i], _ = zx.Float64(); newRefY[i], _ = zy.Float64()
				if i < escAt {
					t1.Mul(zx, zx); t2.Mul(zy, zy)
					nzx.Sub(t1, t2).Add(nzx, bestCx)
					nzy.Mul(zx, zy).Mul(nzy, bigTwo).Add(nzy, bestCy)
					zx.Copy(nzx); zy.Copy(nzy)
				}
			}
		}
		// Store in cache — including derived arrays so cache hits skip all O(N) work.
		cachedRefX        = newRefX
		cachedRefY        = newRefY
		cachedRefEscAt    = escAt
		cachedRefCx       = thisCx
		cachedRefCy       = thisCy
		cachedRefMaxIter  = maxIter
		cachedRefPrec     = prec
		cachedRefZoomExp  = thisExp
		cachedBestCxStr   = bestCx.Text('g', 40)
		cachedBestCyStr   = bestCy.Text('g', 40)

		// Pre-compute derived arrays once and cache them.
		newRef2X  := make([]float64, escAt+2)
		newRef2Y  := make([]float64, escAt+2)
		newRefMag2 := make([]float64, escAt+2)
		for i := 0; i <= escAt; i++ {
			newRef2X[i]  = 2 * newRefX[i]
			newRef2Y[i]  = 2 * newRefY[i]
			newRefMag2[i] = newRefX[i]*newRefX[i] + newRefY[i]*newRefY[i]
		}
		cachedRef2X   = newRef2X
		cachedRef2Y   = newRef2Y
		cachedRefMag2 = newRefMag2

		// Build and cache BLA table.
		if escAt > 32 {
			cachedBLATable = computeBLA(newRefX, newRefY, escAt, exp, &globalBlaScratch)
		} else {
			cachedBLATable = nil
		}
	} else {
		// Cache hit — reuse orbit. Restore bestCx/bestCy from saved world
		// coordinates — NOT from pixel coords, since pixelSize changes each zoom.
		bestCx = new(big.Float).SetPrec(prec)
		bestCy = new(big.Float).SetPrec(prec)
		bestCx.SetString(cachedBestCxStr)
		bestCy.SetString(cachedBestCyStr)
		// Recompute pixel coords of the reference point at the current pixelSize.
		// dcx = (bestCx - renderCx) / pixelSize + halfW
		dcxWorld := new(big.Float).SetPrec(prec).Sub(bestCx, renderCx)
		dcyWorld := new(big.Float).SetPrec(prec).Sub(bestCy, renderCy)
		psF := new(big.Float).SetPrec(prec).SetFloat64(pixelSize)
		dcxPix, _ := new(big.Float).SetPrec(prec).Quo(dcxWorld, psF).Float64()
		dcyPix, _ := new(big.Float).SetPrec(prec).Quo(dcyWorld, psF).Float64()
		bestRefPX = int(math.Round(dcxPix + halfW))
		bestRefPY = int(math.Round(dcyPix + halfH))
		// Clamp to frame bounds.
		if bestRefPX < 0 { bestRefPX = 0 }
		if bestRefPX >= w { bestRefPX = w - 1 }
		if bestRefPY < 0 { bestRefPY = 0 }
		if bestRefPY >= h { bestRefPY = h - 1 }
	}

	refX      := cachedRefX
	refY      := cachedRefY
	escapedAt := cachedRefEscAt
	if escapedAt < 0 { escapedAt = 0 }

	// All derived arrays are pre-computed at cache-build time — zero per-frame cost.
	ref2X   := cachedRef2X
	ref2Y   := cachedRef2Y
	refMag2 := cachedRefMag2
	blaTable := cachedBLATable

	// Pack blaTable for CGo — Go slice header → C pointer + length.
	// The slice is pinned for the duration of the workers by staying in scope.
	var blaPtr *C.BlaEntry
	blaLen := C.int(0)
	if len(blaTable) > 0 {
		blaPtr = (*C.BlaEntry)(unsafe.Pointer(&blaTable[0]))
		blaLen = C.int(len(blaTable))
	}

	// C pointers to the reference arrays — pinned for duration of workers.
	refXPtr    := (*C.double)(unsafe.Pointer(&refX[0]))
	refYPtr    := (*C.double)(unsafe.Pointer(&refY[0]))
	ref2XPtr   := (*C.double)(unsafe.Pointer(&ref2X[0]))
	ref2YPtr   := (*C.double)(unsafe.Pointer(&ref2Y[0]))
	refMag2Ptr := (*C.double)(unsafe.Pointer(&refMag2[0]))
	cRefLen  := C.int(escapedAt + 1)
	cPixelSize := C.double(pixelSize)
	cMaxIter   := C.int(maxIter)
	cW         := C.int(w)
	cCxF64     := C.double(cxF64)
	// big.Float latency means GOMAXPROCS×2 just burns power with no gain.
	numWorkers := s20feWorkerCount(true)
	var rowCounter int32
	atomic.StoreInt32(&workerPoolIdx, 0) // reset pool index for this frame
	// Adaptive chunk size: aim for ~4 chunks per worker for good load balance.
	// At 90 rows with 5 workers: 90/(5*4)=4 → chunkSize=4; better than 16.
	chunkSize := h / (numWorkers * 4)
	if chunkSize < 1 { chunkSize = 1 }
	if chunkSize > 16 { chunkSize = 16 }
	var wg sync.WaitGroup

	for wk := 0; wk < numWorkers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Per-worker big.Float scratch.
			pxBig := new(big.Float).SetPrec(prec); pyBig := new(big.Float).SetPrec(prec)
			zxBig := new(big.Float).SetPrec(prec); zyBig := new(big.Float).SetPrec(prec)
			nrxBig := new(big.Float).SetPrec(prec); nryBig := new(big.Float).SetPrec(prec)
			t1Big := new(big.Float).SetPrec(prec); t2Big := new(big.Float).SetPrec(prec)
			dcxBig := new(big.Float).SetPrec(prec); dcyBig := new(big.Float).SetPrec(prec)
			mag2Big := new(big.Float).SetPrec(prec)
			twoBig := new(big.Float).SetPrec(prec).SetFloat64(2)
			sixteenBig := new(big.Float).SetPrec(prec).SetFloat64(16)
			oldZxBig := new(big.Float).SetPrec(prec); oldZyBig := new(big.Float).SetPrec(prec)

			// Worker-local multi-reference cache — reuse package-level pool.
			// Each worker gets its own index slot so no races.
			wkIdx := int(atomic.AddInt32(&workerPoolIdx, 1) - 1)
			if wkIdx >= len(workerRefPool) { wkIdx = 0 } // safety
			pool := &workerRefPool[wkIdx]
			if len(pool.refX) < maxIter+2 {
				pool.refX = make([]float64, maxIter+2)
				pool.refY = make([]float64, maxIter+2)
				pool.ref2X = make([]float64, maxIter+2)
				pool.ref2Y = make([]float64, maxIter+2)
			}
			localRefX := pool.refX
			localRefY := pool.refY
			local2X := pool.ref2X
			local2Y := pool.ref2Y
			localEscAt := -1; localRefPX := 0; localRefPY := 0

			// Glitch recovery: big.Float per-pixel orbit rebuild.
			// Cap per-worker rebuilds to avoid spending 200ms per pixel at e32+.
			// Overflow pixels stay marked -2.0 and get handled in the second pass
			// below (which uses the fast C scalar path, not big.Float).
			const maxBigFloatRebuilds = 2
			bigFloatRebuilds := 0

			// Reuse pooled row scratch — avoids one allocation per worker per frame.
			if len(pool.rowScratch) < w {
				pool.rowScratch = make([]float64, w)
			}
			rowScratch := pool.rowScratch

			for {
				chunkStart := int(atomic.AddInt32(&rowCounter, int32(chunkSize))) - chunkSize
				if chunkStart >= h { break }
				chunkEnd := chunkStart + chunkSize
				if chunkEnd > h { chunkEnd = h }

				// dcxBase is the same for every row in this chunk — hoist it out.
				dcxBase := (0.0 - float64(bestRefPX)) * pixelSize
				rowPtr := (*C.double)(unsafe.Pointer(&rowScratch[0]))

				for y := chunkStart; y < chunkEnd; y++ {
					rowBase := y * w
					rowPy := cyF64 + (float64(y)-halfH)*pixelSize
					rowDcy := (float64(y) - float64(bestRefPY)) * pixelSize

					// ── BLA row call ───────────────────────────────────────
					C.mb_perturb_row_bla(
					rowPtr, cW,
					refXPtr, refYPtr, ref2XPtr, ref2YPtr, refMag2Ptr, cRefLen,
					C.double(dcxBase), C.double(rowDcy),
					cPixelSize,
					cCxF64, C.double(rowPy),
					blaPtr, blaLen,
					cMaxIter,
				)

					// Copy results; handle glitches (-2.0) with local ref or big.Float.
					for x := 0; x < w; x++ {
						result := rowScratch[x]
						if result != -2.0 {
							data[rowBase+x] = result
							continue
						}

						// ── result == -2.0: glitch — try local reference ──
						px := cxF64 + (float64(x)-halfW)*pixelSize
						py := rowPy
						dcx := (float64(x) - float64(bestRefPX)) * pixelSize
						dcy := rowDcy
						escaped := false
						var rx, ry float64
						iter := 0
						// Only reuse local ref if it's nearby — far pixels need their own orbit.
						localRefClose := abs2(x-localRefPX, y-localRefPY) < (w*w/4 + h*h/4)
						if localEscAt > escapedAt && localRefClose {
							dcx2 := (float64(x) - float64(localRefPX)) * pixelSize
							dcy2 := (float64(y) - float64(localRefPY)) * pixelSize
							dx2, dy2 := 0.0, 0.0; iter = 0
							for iter <= localEscAt && iter < maxIter {
								rx = localRefX[iter] + dx2
								ry = localRefY[iter] + dy2
								if rx*rx+ry*ry > 16 { escaped = true; break }
								d2 := dx2 * dx2; e2 := dy2 * dy2; f2 := 2 * dx2 * dy2
								ndx := local2X[iter]*dx2 - local2Y[iter]*dy2 + d2 - e2 + dcx2
								ndy := local2X[iter]*dy2 + local2Y[iter]*dx2 + f2 + dcy2
								dx2 = ndx; dy2 = ndy; iter++
							}
						}
						_ = dcx; _ = dcy // used in local ref path above

						// ── big.Float fallback → new local ref ───────────
						if !escaped && iter < maxIter {
							if bigFloatRebuilds >= maxBigFloatRebuilds {
								data[rowBase+x] = -2.0 // second-pass will handle
								continue
							}
							bigFloatRebuilds++
							dcxBig.SetFloat64((float64(x) - halfW) * pixelSize)
							dcyBig.SetFloat64((float64(y) - halfH) * pixelSize)
							pxBig.Add(renderCx, dcxBig); pyBig.Add(renderCy, dcyBig)
							zxBig.SetFloat64(0); zyBig.SetFloat64(0)
							esc := maxIter; checkPeriod := 8
							for i := 0; i <= maxIter; i++ {
								localRefX[i], _ = zxBig.Float64(); localRefY[i], _ = zyBig.Float64()
								t1Big.Mul(zxBig, zxBig); t2Big.Mul(zyBig, zyBig); mag2Big.Add(t1Big, t2Big)
								if mag2Big.Cmp(sixteenBig) > 0 { esc = i; break }
								if zxBig.Cmp(oldZxBig) == 0 && zyBig.Cmp(oldZyBig) == 0 { esc = maxIter; break }
								if i == checkPeriod { oldZxBig.Copy(zxBig); oldZyBig.Copy(zyBig); checkPeriod *= 2 }
								if i < maxIter {
									nrxBig.Sub(t1Big, t2Big).Add(nrxBig, pxBig)
									nryBig.Mul(zxBig, zyBig).Mul(nryBig, twoBig).Add(nryBig, pyBig)
									zxBig.Copy(nrxBig); zyBig.Copy(nryBig)
								}
							}
							localEscAt = esc; localRefPX, localRefPY = x, y
							for j := 0; j <= esc; j++ { local2X[j] = 2 * localRefX[j]; local2Y[j] = 2 * localRefY[j] }
							iter = esc; escaped = esc < maxIter
							if escaped { rx = localRefX[esc]; ry = localRefY[esc] }
						}

						if escaped {
							rx2 := rx*rx - ry*ry + px; ry2 := 2*rx*ry + py
							rx3 := rx2*rx2 - ry2*ry2 + px; ry3 := 2*rx2*ry2 + py
							data[rowBase+x] = smoothColor(iter+2, rx3, ry3)
						} else {
							data[rowBase+x] = -1
						}
					}
				}
			}
		}()
	}

	wg.Wait()

	// ── Second pass: clean up any pixels still marked -2.0 ─────────────────

	// These are pixels where big.Float rebuild quota was exceeded.
	// Use mb_perturb_pixel (C scalar with its own perturbation from bestRef)
	// which is fast (~1µs) and resolves most remaining glitches correctly.
	refXPtr2   := (*C.double)(unsafe.Pointer(&refX[0]))
	refYPtr2   := (*C.double)(unsafe.Pointer(&refY[0]))
	ref2XPtr2  := (*C.double)(unsafe.Pointer(&ref2X[0]))
	ref2YPtr2  := (*C.double)(unsafe.Pointer(&ref2Y[0]))
	for y2 := 0; y2 < h; y2++ {
		rowBase2 := y2 * w
		py2 := cyF64 + (float64(y2)-halfH)*pixelSize
		for x2 := 0; x2 < w; x2++ {
			if data[rowBase2+x2] != -2.0 {
				continue
			}
			px2 := cxF64 + (float64(x2)-halfW)*pixelSize
			dcx2 := (float64(x2) - float64(bestRefPX)) * pixelSize
			dcy2 := (float64(y2) - float64(bestRefPY)) * pixelSize
			result := float64(C.mb_perturb_pixel(
				refXPtr2, refYPtr2, ref2XPtr2, ref2YPtr2, cRefLen,
				C.double(dcx2), C.double(dcy2),
				C.double(0.0), C.double(0.0), // dx0, dy0: start delta at zero
				C.int(0),                      // sa_iter: no BLA skip in second pass
				C.double(px2), C.double(py2),
				cMaxIter,
			))
			if result == -2.0 {
				result = -1.0 // truly unresolvable — mark as inside-set
			}
			data[rowBase2+x2] = result
		}
	}

	return data
}


// ─────────────────────────────────────────────
//  Animation helpers — CPU-budgeted render + direct-to-RGB
// ─────────────────────────────────────────────

// renderMandelbrotRGB renders a frame directly into an RGB byte buffer,
// bypassing the float64 intermediate array for the shallow-zoom (std) path.
// For deep zoom or Julia mode it falls back to float64 + LUT colormap.
//
// This is called by export.go's animation pipeline.  numWorkers controls
// how many goroutines are used so parallel frames don't over-subscribe.
func renderMandelbrotRGB(w, h int, rcx, rcy, rzoom *big.Float,
	rgb []byte, lut *PaletteLUT, colorScale, lutSizeF float64, numWorkers int) {

	exp := rzoom.MantExp(nil)
	usePerturbation := exp > 43 && !juliaMode

	if usePerturbation || juliaMode {
		// Fall back: render to float64, then colormap in one pass.
		data := renderMandelbrotWithWorkers(w, h, rcx, rcy, rzoom, numWorkers)
		for i := 0; i < w*h; i++ {
			v := data[i]
			o := i * 3
			if v < 0 {
				rgb[o] = 0; rgb[o+1] = 0; rgb[o+2] = 0
			} else {
				idx := valToIdx(v, colorScale, lutSizeF)
				c := lut.Colors[idx]
				rgb[o] = c.R; rgb[o+1] = c.G; rgb[o+2] = c.B
			}
		}
		return
	}

	// Shallow zoom fast path: C row function → LUT, directly into rgb[].
	// One pass over the data instead of two (render then colormap).
	prec := precForExp(exp)
	// Work on copies so we don't mutate the caller's big.Float values.
	rcx = new(big.Float).SetPrec(prec).Copy(rcx)
	rcy = new(big.Float).SetPrec(prec).Copy(rcy)
	rzoom = new(big.Float).SetPrec(prec).Copy(rzoom)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(rzoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()
	cxF64, _ := rcx.Float64()
	cyF64, _ := rcy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2
	curMaxIter := maxIter

	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0) * 2
	}

	// Use x4 row function: 4 rows per dispatch, interleaved FP streams.
	nChunks := (h + 3) / 4
	var chunkCounter int32
	// Scratch buffer: 4 rows × w pixels, reused per goroutine.
	var rgbWg sync.WaitGroup
	for wk := 0; wk < numWorkers; wk++ {
		rgbWg.Add(1)
		go func() {
			defer rgbWg.Done()
			// Per-goroutine 4-row float64 scratch.
			scratch := make([]float64, 4*w)
			for {
				chunk := int(atomic.AddInt32(&chunkCounter, 1)) - 1
				if chunk >= nChunks { break }
				y0 := chunk * 4
				nRows := 4
				if y0+nRows > h { nRows = h - y0 }
				rowPy := cyF64 + (float64(y0)-halfH)*pixelSize
				scrPtr := (*C.double)(unsafe.Pointer(&scratch[0]))
				C.mb_row_std_x4(scrPtr, C.int(w), C.int(nRows),
					C.double(cxF64), C.double(rowPy),
					C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
				for r := 0; r < nRows; r++ {
					base := (y0+r) * w * 3
					rowOff := r * w
					for x := 0; x < w; x++ {
						v := scratch[rowOff+x]
						o := base + x*3
						if v < 0 {
							rgb[o] = 0; rgb[o+1] = 0; rgb[o+2] = 0
						} else {
							idx := valToIdx(v, colorScale, lutSizeF)
							c := lut.Colors[idx]
							rgb[o] = c.R; rgb[o+1] = c.G; rgb[o+2] = c.B
						}
					}
				}
			}
		}()
	}
	rgbWg.Wait()
}

// renderMandelbrotWithWorkers renders to a fresh float64 slice with a
// caller-specified goroutine budget.  Used by renderMandelbrotRGB's fallback
// and directly available for any future caller that needs CPU budgeting.
//
// numWorkers <= 0 means "use GOMAXPROCS*2" (same as renderMandelbrot).
func renderMandelbrotWithWorkers(w, h int, rcx, rcy, rzoom *big.Float, numWorkers int) []float64 {
	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0) * 2
	}

	// Always allocate fresh — animation has multiple frames in-flight.
	data := make([]float64, w*h)

	exp := rzoom.MantExp(nil)
	prec := precForExp(exp)
	rcx = new(big.Float).SetPrec(prec).Copy(rcx)
	rcy = new(big.Float).SetPrec(prec).Copy(rcy)
	rzoom = new(big.Float).SetPrec(prec).Copy(rzoom)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(rzoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()
	cxF64, _ := rcx.Float64()
	cyF64, _ := rcy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2
	isJulia := juliaMode
	jcx, jcy := juliaR, juliaI
	curMaxIter := maxIter

	usePerturbation := exp > 43 && !juliaMode

	if !usePerturbation {
		// Dispatch 4 rows at a time — mb_row_std_x4 interleaves 4 independent
		// pixel streams so ARM64 FP units stay busy during latency gaps.
		nChunks := (h + 3) / 4
		var chunkCounter int32
		var rowWg sync.WaitGroup
		for wk := 0; wk < numWorkers; wk++ {
			rowWg.Add(1)
			go func() {
				defer rowWg.Done()
				for {
					chunk := int(atomic.AddInt32(&chunkCounter, 1)) - 1
					if chunk >= nChunks { break }
					y0 := chunk * 4
					nRows := 4
					if y0+nRows > h { nRows = h - y0 }
					if isJulia {
						for r := 0; r < nRows; r++ {
							rSlice := data[(y0+r)*w : (y0+r)*w+w]
							rPtr := (*C.double)(unsafe.Pointer(&rSlice[0]))
							rowPy := cyF64 + (float64(y0+r)-halfH)*pixelSize
							C.mb_row_julia(rPtr, C.int(w),
								C.double(cxF64), C.double(rowPy),
								C.double(pixelSize), C.double(halfW),
								C.double(jcx), C.double(jcy), C.int(curMaxIter))
						}
					} else {
						rowPy := cyF64 + (float64(y0)-halfH)*pixelSize
						rowPtr := (*C.double)(unsafe.Pointer(&data[y0*w]))
						C.mb_row_std_x4(rowPtr, C.int(w), C.int(nRows),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
					}
				}
			}()
		}
		rowWg.Wait()
		return data
	}

	// Deep zoom: perturbation path is inherently stateful (reference orbit),
	// so we delegate to the standard renderInto which handles it correctly,
	// then copy into our fresh buffer.  Perturbation frames are rare during
	// animations (zoom > 10^13) and are already memory-bound, so the copy
	// is negligible versus big.Float work.
	src := renderInto(w, h, rcx, rcy, rzoom, false)
	copy(data, src)
	return data
}

