// prefetch.go — Zoom look-ahead cache.
//
// Press 'C' to pre-compute the next N zoom levels straight in from the
// current position.  Each cached frame is a fully rendered []float64 buffer
// identical to what renderMandelbrot would produce.  When you press 'z' the
// next frame is popped from the cache and copied into renderData instantly
// instead of being recomputed.
//
// Cache is invalidated automatically on:
//   pan (w/a/s/d), zoom-out (x/X), iter change, palette change,
//   histoEQ toggle, julia toggle, or a new 'C' press.
//
// The pre-computation runs on a background goroutine using a reduced worker
// count (s20feWorkerCount) so the foreground frame stays responsive.

package main

import (
	"fmt"
	"math/big"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ─────────────────────────────────────────────
//  Cache entry
// ─────────────────────────────────────────────

type prefetchEntry struct {
	data    []float64
	w, h    int
	zoomExp int // binary exponent of zoom — used for display only
}

// ─────────────────────────────────────────────
//  Cache state  (all guarded by prefetchMu)
// ─────────────────────────────────────────────

var prefetchMu sync.Mutex

// prefetchFrames holds ready frames in zoom order: [0] = next zoom in.
var prefetchFrames []prefetchEntry

// prefetchReady counts how many entries in prefetchFrames are fully rendered.
// Written atomically by the background goroutine; read by the main loop.
var prefetchReady int32

// prefetchTotal is the target number of frames requested by the user.
var prefetchTotal int32

// prefetchAbort: set to 1 to cancel an in-progress pre-computation.
var prefetchAbort int32

// prefetchAutoSize: how many frames to auto-refill when cache runs low.
// Set to the last user-requested count so refills match original request.
var prefetchAutoSize int32 = 5

// prefetchAutoRunning: 1 while a silent background refill is in progress.
var prefetchAutoRunning int32

// prefetchAnchorCx/Cy/Zoom are the view coordinates at the moment 'C' was
// pressed.  Used to detect stale cache entries after a pan or reset.
var prefetchAnchorCx   string
var prefetchAnchorCy   string
var prefetchAnchorZoom string

// prefetchPalette / prefetchMaxIter / prefetchHistoEQ / prefetchJulia snapshot
// rendering settings at cache-build time so we can detect stale cache.
var prefetchPalette  string
var prefetchMaxIter  int
var prefetchHistoEQ  bool
var prefetchJulia    bool

// ─────────────────────────────────────────────
//  Invalidation
// ─────────────────────────────────────────────

// invalidatePrefetch discards the cache and cancels any running background
// computation.  Safe to call from the main goroutine at any time.
func invalidatePrefetch() {
	atomic.StoreInt32(&prefetchAbort, 1)
	atomic.StoreInt32(&prefetchAutoRunning, 0)
	prefetchMu.Lock()
	prefetchFrames = prefetchFrames[:0]
	prefetchMu.Unlock()
	atomic.StoreInt32(&prefetchReady, 0)
	atomic.StoreInt32(&prefetchTotal, 0)
	cachedRefEscAt = -1
	prevW = 0 // force full terminal redraw — view changed
}

// startSilentRefill kicks off a background refill of the prefetch cache
// without blocking the UI or showing a progress bar. Called automatically
// when the cache drops to 1 frame remaining while zooming.
func startSilentRefill(n int) {
	if atomic.CompareAndSwapInt32(&prefetchAutoRunning, 0, 1) == false {
		return // already running
	}

	// Snapshot the current step count — we compute each frame's zoom as
	// 1.5^(baseSteps + i + 1) to avoid accumulated multiplication error.
	baseSteps := zoomSteps
	deepExp   := int(float64(baseSteps+n)*0.585) + 2
	prec      := precForExp(deepExp)
	if prec < 128 { prec = 128 }

	baseCx := new(big.Float).SetPrec(prec).Copy(cx)
	baseCy := new(big.Float).SetPrec(prec).Copy(cy)
	// baseZoom still needed for anchor string comparison


	// Snapshot settings.
	snapPalette  := currentPaletteName
	snapMaxIter  := maxIter
	snapHistoEQ  := histoEQ
	snapJulia    := juliaMode
	snapCx       := cx.Text('g', 30)
	snapCy       := cy.Text('g', 30)
	snapZoom     := prefetchAnchorZoom

	tw, trows := getTermSize()
	th := (trows - 2) * 2
	if th < 2 { th = 2 }

	go func() {
		defer atomic.StoreInt32(&prefetchAutoRunning, 0)

		for i := 0; i < n; i++ {
			if atomic.LoadInt32(&prefetchAbort) != 0 { return }
			// Abort if view changed while we were rendering.
			if cx.Text('g', 30) != snapCx || cy.Text('g', 30) != snapCy ||
				currentPaletteName != snapPalette || maxIter != snapMaxIter ||
				histoEQ != snapHistoEQ || juliaMode != snapJulia {
				return
			}

			// Compute this frame's zoom as 1.5^(baseSteps+i+1) fresh — no accumulation.
			fz := computeZoomForSteps(baseSteps + i + 1, prec)
			fcx := new(big.Float).SetPrec(prec).Copy(baseCx)
			fcy := new(big.Float).SetPrec(prec).Copy(baseCy)

			exp  := fz.MantExp(nil)
			data := renderMandelbrotWithWorkers(tw, th, fcx, fcy, fz,
				s20feWorkerCount(exp > 43))

			if atomic.LoadInt32(&prefetchAbort) != 0 { return }

			// Only append if cache is still valid for our snapshot.
			prefetchMu.Lock()
			if prefetchAnchorCx == snapCx && prefetchAnchorCy == snapCy &&
				currentPaletteName == snapPalette {
				prefetchFrames = append(prefetchFrames, prefetchEntry{
					data: data, w: tw, h: th, zoomExp: exp,
				})
				atomic.AddInt32(&prefetchReady, 1)
				atomic.AddInt32(&prefetchTotal, 1)
				// Advance the snapshot anchor so next frame's stale check works.
				snapZoom = fz.Text('g', 30)
				prefetchAnchorZoom = snapZoom
			}
			prefetchMu.Unlock()
		}
	}()
}
func prefetchStale() bool {
	if atomic.LoadInt32(&prefetchTotal) == 0 {
		return true
	}
	if cx.Text('g', 30) != prefetchAnchorCx {
		return true
	}
	if cy.Text('g', 30) != prefetchAnchorCy {
		return true
	}
	if currentPaletteName != prefetchPalette {
		return true
	}
	if maxIter != prefetchMaxIter {
		return true
	}
	if histoEQ != prefetchHistoEQ {
		return true
	}
	if juliaMode != prefetchJulia {
		return true
	}
	return false
}

// ─────────────────────────────────────────────
//  Pop — called by the 'z' handler
// ─────────────────────────────────────────────

// popPrefetchFrame returns the next pre-computed frame and advances the
// cache, or returns nil if the cache is empty or stale.
// It also shifts the zoom anchor forward by ×1.5 so subsequent pops remain
// valid.
func popPrefetchFrame(w, h int) []float64 {
	if prefetchStale() {
		return nil
	}
	if atomic.LoadInt32(&prefetchReady) == 0 {
		return nil
	}

	prefetchMu.Lock()
	// Find the first entry with data (completed frame at lowest index).
	idx := -1
	for i, e := range prefetchFrames {
		if e.data != nil {
			idx = i
			break
		}
	}
	if idx < 0 {
		prefetchMu.Unlock()
		return nil
	}
	entry := prefetchFrames[idx]
	// Remove consumed entry — shift remaining down.
	prefetchFrames = prefetchFrames[idx+1:]
	prefetchMu.Unlock()

	atomic.AddInt32(&prefetchReady, -1)

	if entry.w != w || entry.h != h {
		invalidatePrefetch()
		return nil
	}
	return entry.data
}

// ─────────────────────────────────────────────
//  Background pre-computation
// ─────────────────────────────────────────────

// startPrefetch renders N zoom frames simultaneously using all CPU cores,
// then stores them in order for instant consumption by the 'z' key.
//
// KEY CHANGE vs the old serial approach: all N frames are rendered in
// parallel. The total worker budget (runtime.NumCPU()) is divided evenly
// across frames. Frame i gets workers[i*stride:(i+1)*stride].  All frames
// run concurrently so the total wall time ≈ time of one frame instead of N.
//
// Frames are inserted into prefetchFrames in index order (not completion
// order) using a pre-allocated slice so the consumer always pops frame[0].
func startPrefetch(n int) {
	if n < 1 { n = 1 }
	if n > 20 { n = 20 }
	atomic.StoreInt32(&prefetchAutoSize, int32(n))

	invalidatePrefetch()
	time.Sleep(10 * time.Millisecond) // let any lingering goroutine see abort

	// Use the precision needed for the DEEPEST frame we'll render (frame n),
	// not just the current zoom. Each frame is ×1.5 deeper than the last.
	baseExp  := zoom.MantExp(nil)
	deepExp  := baseExp + int(float64(n)*0.585+1) // log2(1.5^n) ≈ n*0.585
	prec     := precForExp(deepExp)
	if prec < 128 { prec = 128 }

	baseCx   := new(big.Float).SetPrec(prec).Copy(cx)
	baseCy   := new(big.Float).SetPrec(prec).Copy(cy)

	// Anchor is set to CURRENT zoom. popPrefetchFrame checks zoom == anchor
	// before the z-handler multiplies zoom, so this must match what the user
	// sees RIGHT NOW.
	prefetchAnchorCx   = cx.Text('g', 30)
	prefetchAnchorCy   = cy.Text('g', 30)
	prefetchAnchorZoom = zoom.Text('g', 30)
	prefetchPalette    = currentPaletteName
	prefetchMaxIter    = maxIter
	prefetchHistoEQ    = histoEQ
	prefetchJulia      = juliaMode

	prefetchMu.Lock()
	prefetchFrames = make([]prefetchEntry, n) // pre-sized so index writes are safe
	prefetchMu.Unlock()
	atomic.StoreInt32(&prefetchReady, 0)
	atomic.StoreInt32(&prefetchTotal, 0)
	atomic.StoreInt32(&prefetchAbort, 0)

	tw, trows := getTermSize()
	th := (trows - 2) * 2
	if th < 2 { th = 2 }

	// Divide worker budget across frames.
	// At least 1 worker per frame; shallow frames get more since they're faster.
	totalWorkers := runtime.NumCPU()
	if totalWorkers < 1 { totalWorkers = 1 }
	workersPerFrame := totalWorkers / n
	if workersPerFrame < 1 { workersPerFrame = 1 }

	// completedCh receives frame indices as they finish (unordered).
	completedCh := make(chan int, n)
	doneSig     := make(chan struct{})

	// Launch all N frames in parallel.
	// Compute each frame's zoom as 1.5^(baseSteps+i+1) fresh — no accumulation error.
	for i := 0; i < n; i++ {
		idx := i
		fz  := computeZoomForSteps(zoomSteps + i + 1, prec)
		fcx := new(big.Float).SetPrec(prec).Copy(baseCx)
		fcy := new(big.Float).SetPrec(prec).Copy(baseCy)
		wk  := workersPerFrame

		go func() {
			if atomic.LoadInt32(&prefetchAbort) != 0 { return }
			exp  := fz.MantExp(nil)
			data := renderMandelbrotWithWorkers(tw, th, fcx, fcy, fz, wk)
			if atomic.LoadInt32(&prefetchAbort) != 0 { return }

			prefetchMu.Lock()
			prefetchFrames[idx] = prefetchEntry{data: data, w: tw, h: th, zoomExp: exp}
			prefetchMu.Unlock()

			atomic.AddInt32(&prefetchTotal, 1)
			atomic.AddInt32(&prefetchReady, 1)
			completedCh <- idx
		}()
	}

	// ── Live progress bar ─────────────────────────────────────────────────
	restoreTerminal()
	fmt.Printf("\033[2J\033[H\033[0m")
	fmt.Printf("Pre-computing %d zoom frames in parallel…  (any key = cancel)\n\n", n)

	barWidth  := 30
	completed := 0

	keyCh := make(chan struct{}, 1)
	go func() {
		b := [1]byte{}
		for {
			select {
			case <-doneSig:
				return
			default:
			}
			ttyFile.SetDeadline(time.Now().Add(50 * time.Millisecond))
			nr, _ := ttyFile.Read(b[:])
			ttyFile.SetDeadline(time.Time{})
			if nr > 0 { keyCh <- struct{}{}; return }
		}
	}()

	for completed < n {
		select {
		case <-completedCh:
			completed++
			filled := completed * barWidth / n
			bar := ""
			for b := 0; b < barWidth; b++ {
				if b < filled { bar += "█" } else { bar += "░" }
			}
			pct      := completed * 100 / n
			exp      := zoom.MantExp(nil)
			frameExp := float64(exp) + float64(completed)*0.585
			depth10  := frameExp * 0.30103
			fmt.Printf("\r  [%s] %d/%d (%d%%)  ≈10^%.1f ",
				bar, completed, n, pct, depth10)
			os.Stdout.Sync()

		case <-keyCh:
			atomic.StoreInt32(&prefetchAbort, 1)
			fmt.Printf("\n\n  Cancelled — %d/%d frames ready.\n", completed, n)
			time.Sleep(700 * time.Millisecond)
			// Trim frames slice to only the ones actually completed.
			// Frames that didn't finish have zero-value entries.
			prefetchMu.Lock()
			good := make([]prefetchEntry, 0, completed)
			for _, e := range prefetchFrames {
				if e.data != nil { good = append(good, e) }
			}
			prefetchFrames = good
			prefetchMu.Unlock()
			initTerminal()
			return
		}
	}

	close(doneSig)
	fmt.Printf("\n\n  ✓ Done — %d frames cached.  Press z to fly.\n", n)
	time.Sleep(700 * time.Millisecond)
	initTerminal()
}

// ─────────────────────────────────────────────
//  Zoom-out prefetch + live terminal playback
// ─────────────────────────────────────────────

// zoomOutCache holds pre-rendered zoom-out frames for live terminal playback.
var zoomOutCache [][]byte
var zoomOutMu    sync.Mutex
var zoomOutReady int32
var zoomOutTotal int32
var zoomOutAbort int32

// startZoomOutPlayback pre-renders zoom-out frames from the current position
// back toward zoom=1, then plays them back in the terminal at the requested
// speed. Mirrors the M-button exportAnimation path but outputs to the
// terminal instead of PPM files.
func startZoomOutPlayback(multiplier float64, fps int) {
	if multiplier <= 1.0 || multiplier > 2.0 {
		multiplier = 1.02
	}
	if fps < 1  { fps = 10 }
	if fps > 60 { fps = 60 }

	atomic.StoreInt32(&zoomOutAbort, 1)
	zoomOutMu.Lock()
	zoomOutCache = zoomOutCache[:0]
	zoomOutMu.Unlock()
	atomic.StoreInt32(&zoomOutReady, 0)
	atomic.StoreInt32(&zoomOutTotal, 0)
	atomic.StoreInt32(&zoomOutAbort, 0)

	prec        := zoom.Prec()
	captureCx   := new(big.Float).SetPrec(prec).Copy(cx)
	captureCy   := new(big.Float).SetPrec(prec).Copy(cy)
	captureZoom := new(big.Float).SetPrec(prec).Copy(zoom)

	// Count frames by stepping zoom down until <= 1.
	nFrames := 0
	{
		cur  := new(big.Float).SetPrec(prec).Copy(captureZoom)
		mBig := new(big.Float).SetPrec(prec).SetFloat64(multiplier)
		one  := new(big.Float).SetPrec(prec).SetFloat64(1.0)
		for cur.Cmp(one) > 0 && nFrames < 5000 {
			cur.Quo(cur, mBig)
			nFrames++
		}
	}
	if nFrames == 0 {
		restoreTerminal()
		fmt.Println("Already at base zoom.")
		time.Sleep(700 * time.Millisecond)
		initTerminal()
		return
	}

	atomic.StoreInt32(&zoomOutTotal, int32(nFrames))

	tw, trows := getTermSize()
	th := (trows - 2) * 2
	if th < 2 { th = 2 }

	restoreTerminal()
	fmt.Printf("\033[2J\033[H\033[0m")
	fmt.Printf("Pre-rendering %d zoom-out frames (×÷%.4f, %d fps)…\n\n", nFrames, multiplier, fps)

	barWidth  := 30
	completed := 0
	startGate2 := make(chan struct{})
	doneCh2    := make(chan int)
	ackCh2     := make(chan struct{})
	doneSig2   := make(chan struct{})
	keyCh      := make(chan struct{}, 1)

	go func() {
		initTerminal()
		readCh := make(chan struct{}, 1)
		go func() { getChar(); readCh <- struct{}{} }()
		select {
		case <-readCh:
			restoreTerminal()
			keyCh <- struct{}{}
		case <-doneSig2:
			restoreTerminal()
		}
	}()

	go func() {
		<-startGate2
		mBig := new(big.Float).SetPrec(prec).SetFloat64(multiplier)
		one  := new(big.Float).SetPrec(prec).SetFloat64(1.0)

		zooms := make([]*big.Float, 0, nFrames)
		cur := new(big.Float).SetPrec(prec).Copy(captureZoom)
		for cur.Cmp(one) > 0 && len(zooms) < nFrames {
			zooms = append(zooms, new(big.Float).SetPrec(prec).Copy(cur))
			cur.Quo(cur, mBig)
		}

		lut        := luts[currentPaletteName]
		colorScale := colorDensity * 0.01
		lutSizeF   := float64(lutSize)

		for i, fz := range zooms {
			if atomic.LoadInt32(&zoomOutAbort) != 0 {
				return
			}
			fcx := new(big.Float).SetPrec(prec).Copy(captureCx)
			fcy := new(big.Float).SetPrec(prec).Copy(captureCy)
			data := renderMandelbrotWithWorkers(tw, th, fcx, fcy, fz,
				s20feWorkerCount(fz.MantExp(nil) > 43))
			if atomic.LoadInt32(&zoomOutAbort) != 0 {
				return
			}
			frame := colouriseToANSI(data, tw, th, lut, colorScale, lutSizeF)
			zoomOutMu.Lock()
			zoomOutCache = append(zoomOutCache, frame)
			zoomOutMu.Unlock()
			atomic.AddInt32(&zoomOutReady, 1)
			doneCh2 <- i
			<-ackCh2
		}
		close(doneCh2)
	}()

	// Draw UI then release render goroutine.
	close(startGate2)

	for {
		select {
		case _, ok := <-doneCh2:
			if !ok { goto zoDone }
			completed++
			filled := completed * barWidth / nFrames
			bar := ""
			for b := 0; b < barWidth; b++ {
				if b < filled { bar += "█" } else { bar += "░" }
			}
			fmt.Printf("\r  [%s] %d/%d (%d%%) ",
				bar, completed, nFrames, completed*100/nFrames)
			os.Stdout.Sync()
			ackCh2 <- struct{}{}
		case <-keyCh:
			atomic.StoreInt32(&zoomOutAbort, 1)
			go func() { ackCh2 <- struct{}{} }()
			fmt.Printf("\n\n  Cancelled — %d/%d frames ready.\n", completed, nFrames)
			time.Sleep(700 * time.Millisecond)
			initTerminal()
			return
		}
	}

zoDone:
	close(doneSig2)
	ready := int(atomic.LoadInt32(&zoomOutReady))
	fmt.Printf("\n\n  ✓ %d frames ready — playing at %d fps.  Any key stops.\n", ready, fps)
	time.Sleep(300 * time.Millisecond)
	restoreTerminal()

	frameInterval := time.Second / time.Duration(fps)
	stopCh := make(chan struct{}, 1)
	go func() {
		initTerminal()
		getChar()
		restoreTerminal()
		stopCh <- struct{}{}
	}()

	zoomOutMu.Lock()
	frames := make([][]byte, len(zoomOutCache))
	copy(frames, zoomOutCache)
	zoomOutMu.Unlock()

	ticker := time.NewTicker(frameInterval)
	defer ticker.Stop()

	for _, frame := range frames {
		select {
		case <-stopCh:
			goto playbackDone
		case <-ticker.C:
			os.Stdout.Write(frame)
		}
	}

playbackDone:
	fmt.Print("\033[0m")
	time.Sleep(300 * time.Millisecond)
	initTerminal()
}

// colouriseToANSI converts a float64 render buffer into a complete ANSI
// terminal frame ready to write to stdout. Same logic as drawTerminal's
// pixel loop, extracted so zoom-out frames can be pre-built off-screen.
func colouriseToANSI(data []float64, w, h int, lut *PaletteLUT,
	colorScale, lutSizeF float64) []byte {

	buf := make([]byte, 0, w*(h/2)*42+128)
	buf = append(buf, "\033[H\033[J"...)

	for y := 0; y < h; y += 2 {
		lastBG, lastFG := -2, -2
		for x := 0; x < w; x++ {
			vTop := data[y*w+x]
			var idxTop int
			if vTop < 0 { idxTop = -1 } else { idxTop = valToIdx(vTop, colorScale, lutSizeF) }
			idxBot := -1
			if y+1 < h {
				vBot := data[(y+1)*w+x]
				if vBot >= 0 { idxBot = valToIdx(vBot, colorScale, lutSizeF) }
			}

			var bgSeq, fgSeq []byte
			if idxTop < 0 { bgSeq = bgBlack } else { bgSeq = lut.BG[idxTop] }
			if idxBot < 0 { fgSeq = fgBlack } else { fgSeq = lut.FG[idxBot] }

			if idxTop != lastBG { buf = append(buf, bgSeq...); lastBG = idxTop }
			if idxBot != lastFG { buf = append(buf, fgSeq...); lastFG = idxBot }
			buf = append(buf, blockChar...)
		}
		if y+2 < h { buf = append(buf, '\r', '\n') }
	}
	buf = append(buf, resetSeq...)
	return buf
}
