// globals.go — Shared types, variables, and constants for the Mandelbrot renderer.
// Every other file in this package reads from or writes to these.
package main

import (
	"math/big"
	"runtime"
)

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────

// Color is an RGB triplet stored as uint8 for compact packing.
type Color struct{ R, G, B uint8 }

// PaletteLUT holds a pre-interpolated color table and its pre-built ANSI
// escape byte slices so we never call fmt.Sprintf inside the render loop.
type PaletteLUT struct {
	Colors []Color
	FG, BG [][]byte // foreground / background ANSI escape sequences
}

// Bookmark stores a named view position with full big.Float precision.
type Bookmark struct {
	Name    string
	Cx, Cy  string // serialised as big.Float text so precision survives round-trips
	ZoomExp int    // binary exponent of zoom value (zoom ≈ 2^ZoomExp)
	MaxIter int
}

// ─────────────────────────────────────────────
//  View state
// ─────────────────────────────────────────────

var (
	cx, cy, zoom *big.Float
	zoomSteps    int     // canonical zoom: zoom = 1.5^zoomSteps (no rounding error)
	maxIter      int     = 500
	colorDensity float64 = 2.0
)

// ─────────────────────────────────────────────
//  Feature flags
// ─────────────────────────────────────────────

var (
	showHelp  bool
	juliaMode bool
	juliaR    float64 = -0.7
	juliaI    float64 = 0.27015
	histoEQ   bool    // histogram equalization on/off
	orbitTrap bool    // orbit-trap coloring (reserved for future use)
	adaptIter bool    = true // auto-scale maxIter with zoom depth
)

// ─────────────────────────────────────────────
//  Palette globals
// ─────────────────────────────────────────────

var palettes = map[string][]Color{
	// Blue/Gold: deep navy → royal blue → ice white → rich gold → midnight.
	// More stops and richer midpoints than before for a wider colour arc.
	"Blue/Gold": {
		{0, 2, 30}, {0, 20, 100}, {10, 80, 200},
		{80, 180, 255}, {240, 255, 255}, {255, 220, 80},
		{255, 140, 0}, {180, 60, 0}, {10, 5, 20},
	},
	// Fire: true black-body radiation curve — black→deep red→orange→yellow→white.
	"Fire": {
		{0, 0, 0}, {80, 0, 0}, {200, 30, 0},
		{255, 100, 0}, {255, 210, 0}, {255, 255, 160}, {255, 255, 255},
	},
	// Grayscale: pure luminance ramp, perceptually linear via gamma correction.
	"Grayscale": {{0, 0, 0}, {128, 128, 128}, {255, 255, 255}},
	// Neon: vivid cyberpunk — black→violet→magenta→cyan→lime→white.
	"Neon": {
		{0, 0, 0}, {60, 0, 120}, {200, 0, 255},
		{0, 200, 255}, {0, 255, 120}, {200, 255, 0}, {255, 255, 255},
	},
	// Ocean: midnight black → deep navy → tropical teal → foam white → coral.
	"Ocean": {
		{0, 0, 15}, {0, 20, 80}, {0, 80, 160},
		{0, 180, 200}, {80, 230, 220}, {240, 255, 255},
		{255, 200, 120}, {200, 100, 40},
	},
	// Inferno: matplotlib-style perceptually uniform dark→purple→red→yellow.
	"Inferno": {
		{0, 0, 4}, {20, 5, 60}, {80, 10, 120},
		{160, 30, 100}, {220, 80, 50}, {250, 160, 20},
		{252, 230, 100}, {255, 255, 200},
	},
	// Ultra: the classic UltraFractal "ultra" palette, more saturated stops.
	"Ultra": {
		{0, 0, 0}, {80, 20, 5}, {30, 5, 40}, {10, 2, 70}, {3, 5, 100},
		{0, 10, 140}, {10, 55, 180}, {20, 100, 220}, {60, 148, 230},
		{140, 195, 245}, {215, 240, 252}, {245, 238, 200},
		{252, 210, 100}, {255, 175, 0}, {210, 130, 0}, {140, 75, 0},
	},
	// Sunset: dusk palette — deep purple → crimson → amber → pale gold.
	"Sunset": {
		{5, 0, 25}, {50, 0, 80}, {140, 10, 60},
		{220, 50, 20}, {255, 130, 0}, {255, 210, 80}, {255, 245, 200},
	},
	// Candy: pastel rainbow — every hue at high lightness, very smooth arcs.
	"Candy": {
		{255, 180, 200}, {255, 200, 120}, {200, 255, 150},
		{120, 220, 255}, {180, 150, 255}, {255, 160, 220},
	},
	// Ice: cold whites and blues, minimal saturation — great for deep zooms.
	"Ice": {
		{0, 0, 20}, {10, 30, 80}, {30, 90, 160},
		{100, 180, 230}, {200, 230, 255}, {240, 248, 255}, {255, 255, 255},
	},
}

var currentPaletteName = "Blue/Gold"
var paletteKeys = []string{
	"Blue/Gold", "Fire", "Grayscale", "Neon", "Ocean",
	"Inferno", "Ultra", "Sunset", "Candy", "Ice",
}

var luts map[string]*PaletteLUT

const lutSize = 4096

// Pre-built constant ANSI sequences used when a pixel is pure black (inside set).
var (
	fgBlack  = []byte("\033[38;2;0;0;0m")
	bgBlack  = []byte("\033[48;2;0;0;0m")
	resetSeq = []byte("\033[0m")
)

// ─────────────────────────────────────────────
//  Render buffers (allocated once, reused)
// ─────────────────────────────────────────────

// renderData is the main display buffer — filled by renderMandelbrot each frame.
var renderData []float64
var renderDataW, renderDataH int

// scratchData is a separate buffer used by the minibrot finder / iFeelLucky
// so low-res search renders never clobber the main display buffer.
var scratchData []float64
var scratchW, scratchH int

// termBuf is the terminal output buffer. Capacity grows and is retained across
// frames to avoid GC pressure (zero-allocation rendering).
var termBuf []byte

// termIdxBuf holds the precomputed LUT index for every pixel, computed once
// per frame before the byte-emit loop so valToIdx (trig math) runs separately
// from the ANSI-escape byte appends.
var termIdxBuf []int32

// prevIdxBuf stores the previous frame's LUT indices for cursor-skip delta rendering.
// Cells that haven't changed are skipped with \033[NC instead of re-emitting 41 bytes.
var prevIdxBuf []int32
var prevW, prevH int

// blockChar is the UTF-8 encoding of ▄ (LOWER HALF BLOCK, U+2584).
// Used to pack two pixel rows into one terminal character row.
var blockChar = []byte{0xe2, 0x96, 0x84}

// ─────────────────────────────────────────────
//  Concurrency / search
// ─────────────────────────────────────────────

// minibrotAbort is atomically set to 1 by the abort listener goroutine when
// the user presses 'q' during autoFindMinibrot or iFeelLucky.
var minibrotAbort int32

// animPaused is atomically set to 1 when the user presses Space during an
// animation export, and back to 0 on the next Space press.
var animPaused int32

// computeZoomForSteps computes 1.5^steps at the given precision using
// binary exponentiation. No accumulated error — safe for any step count.
func computeZoomForSteps(steps int, prec uint) *big.Float {
	if prec < 128 { prec = 128 }
	step := new(big.Float).SetPrec(prec).SetFloat64(1.5)
	result := new(big.Float).SetPrec(prec).SetFloat64(1.0)
	n := steps
	neg := n < 0
	if neg { n = -n }
	base := new(big.Float).SetPrec(prec).Copy(step)
	for n > 0 {
		if n&1 == 1 { result.Mul(result, base) }
		base.Mul(base, base)
		n >>= 1
	}
	if neg {
		one := new(big.Float).SetPrec(prec).SetFloat64(1.0)
		result.Quo(one, result)
	}
	return result
}

// recomputeZoom recomputes zoom from zoomSteps using high-precision arithmetic.
// zoom = 1.5^zoomSteps computed fresh — no accumulated rounding error.
func recomputeZoom() {
	estExp := int(float64(zoomSteps)*0.585) + 2
	if estExp < 1 { estExp = 1 }
	prec := precForExp(estExp)
	if prec < 128 { prec = 128 }
	cx.SetPrec(prec)
	cy.SetPrec(prec)
	zoom.Copy(computeZoomForSteps(zoomSteps, prec))
	zoom.SetPrec(prec)
}
// On the i7-12700H (6P+8E = 14 cores, 20 threads) we use all physical cores.
// On the S20 FE (8 cores) we cap at 5 to avoid thermal throttle.
// For deep zoom, big.Float is serial, so fewer workers helps cache locality.
func s20feWorkerCount(deep bool) int {
	n := runtime.NumCPU()
	if n <= 0 {
		n = 1
	}
	if deep {
		if n > 12 {
			return 12 // leave 2 cores for OS / UI on laptop
		}
		if n > 5 {
			return n - 1
		}
		return n
	}
	if n > 16 {
		return 16
	}
	return n
}

// ─────────────────────────────────────────────
//  Bookmarks
// ─────────────────────────────────────────────

// prefetchFrameReady is set to true by the 'z' key handler when a cached
// frame has been copied into renderData.  drawTerminal checks this flag and
// skips re-rendering for that one draw call.
var prefetchFrameReady bool

// bookmarks holds 10 save slots (accessed via keys '1'-'9' and shift+'1'-'9').
var bookmarks [10]Bookmark

// ─────────────────────────────────────────────
//  Render scratch — persistent allocations
// ─────────────────────────────────────────────

// globalBlaScratch avoids ~30 allocs per deep frame (was newBlaScratch each render).
var globalBlaScratch blaScratch

// globalBigSixteen / globalBigTwo: lazily initialized big.Float constants.
// Rebuilt only when precision changes (i.e., zoom depth changes tier).
var (
	globalBigSixteen *big.Float
	globalBigTwo     *big.Float
	globalBigPrec    uint
)

// ─────────────────────────────────────────────
//  Reference orbit cache
// ─────────────────────────────────────────────
//
// The reference orbit at cx,cy only changes when cx or cy changes (pan/reset).
// During a zoom-in sequence, the orbit at depth N+1 is a strict extension of
// depth N — we never need to recompute the prefix. This eliminates the serial
// big.Float bottleneck on every zoom step after the first deep frame.

var cachedRefX      []float64
var cachedRefY      []float64
var cachedRef2X     []float64
var cachedRef2Y     []float64
var cachedRefMag2   []float64
var cachedBLATable  []blaEntry
var cachedRefEscAt  int    = -1
var cachedRefCx     string // cx.Text('g',30) at build time
var cachedRefCy     string // cy.Text('g',30) at build time
var cachedRefMaxIter int   // maxIter at build time
var cachedRefPrec   uint   // precision at build time
var cachedRefZoomExp int   // binary zoom exponent at build time

// cachedBestCx/CY: world coordinates of the best reference point.
// Stored directly so cache hits don't need to reconstruct via pixelSize.
var cachedBestCxStr string
var cachedBestCyStr string

// workerRefData holds per-worker local reference orbit buffers.
// Pre-allocated once; avoids make([]float64, maxIter+2) inside each worker per frame.
type workerRefData struct {
	refX, refY   []float64
	ref2X, ref2Y []float64
	rowScratch   []float64 // reused per-row scratch for BLA output
}

// maxWorkers is an upper bound on concurrent render workers.
// Must be >= max(s20feWorkerCount) and >= GOMAXPROCS*2 for renderMandelbrotWithWorkers.
// S20 FE has 8 cores; GOMAXPROCS*2 = 16. Set 32 for headroom.
const maxWorkers = 32

var workerRefPool [maxWorkers]workerRefData
var workerPoolIdx int32 // reset to 0 before each render; each worker claims a slot
