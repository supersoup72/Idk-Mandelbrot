// finder.go — Minibrot search, "I Feel Lucky" deep zoom, abort listener.
package main

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// ─────────────────────────────────────────────
//  Frame analysis
// ─────────────────────────────────────────────

type frameStats struct {
	maxVal     float64
	blackCount int
	totalPix   int
}

func analyzeFrame(data []float64, w, h int) frameStats {
	var s frameStats
	s.totalPix = w * h
	for _, v := range data {
		if v < 0 {
			s.blackCount++
		} else if v > s.maxVal {
			s.maxVal = v
		}
	}
	return s
}

// ─────────────────────────────────────────────
//  Target selection — minibrot finder
// ─────────────────────────────────────────────

func findBestTarget(data []float64, w, h int) (tX, tY, zoomFactor float64) {
	s := analyzeFrame(data, w, h)
	halfW := float64(w) / 2
	halfH := float64(h) / 2
	blackFrac := float64(s.blackCount) / float64(s.totalPix)

	switch {
	case s.blackCount > 0 && blackFrac < 0.08:
		var sumX, sumY float64
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if data[y*w+x] < 0 {
					sumX += float64(x)
					sumY += float64(y)
				}
			}
		}
		tX = sumX / float64(s.blackCount)
		tY = sumY / float64(s.blackCount)
		zoomFactor = 1.8

	case blackFrac >= 0.08 && blackFrac < 0.60:
		threshold := s.maxVal * 0.85
		var sumX, sumY, wSum float64
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := data[y*w+x]
				if v >= threshold {
					dx := float64(x) - halfW
					dy := float64(y) - halfH
					dist := math.Sqrt(dx*dx + dy*dy)
					w2 := v / (1 + dist*0.05)
					sumX += float64(x) * w2
					sumY += float64(y) * w2
					wSum += w2
				}
			}
		}
		if wSum > 0 {
			tX = sumX / wSum
			tY = sumY / wSum
		} else {
			tX, tY = halfW, halfH
		}
		zoomFactor = 2.0

	case blackFrac >= 0.60:
		bestScore := -1e18
		tX, tY = halfW, halfH
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := data[y*w+x]
				if v <= 0 {
					continue
				}
				dx := float64(x) - halfW
				dy := float64(y) - halfH
				dist := math.Sqrt(dx*dx + dy*dy)
				score := v - dist*0.5
				if score > bestScore {
					bestScore = score
					tX = float64(x)
					tY = float64(y)
				}
			}
		}
		zoomFactor = 1.5

	case s.maxVal > 0:
		threshold := s.maxVal * 0.92
		var sumX, sumY, wSum float64
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := data[y*w+x]
				if v >= threshold {
					dx := float64(x) - halfW
					dy := float64(y) - halfH
					dist := math.Sqrt(dx*dx + dy*dy)
					w2 := v / (1 + dist*0.1)
					sumX += float64(x) * w2
					sumY += float64(y) * w2
					wSum += w2
				}
			}
		}
		if wSum > 0 {
			tX = sumX / wSum
			tY = sumY / wSum
		} else {
			tX, tY = halfW, halfH
		}
		zoomFactor = 2.5

	default:
		tX, tY = halfW, halfH
		zoomFactor = 0.5
	}
	return
}

// ─────────────────────────────────────────────
//  World-space step
// ─────────────────────────────────────────────

func stepPixelToWorld(w, h int, tX, tY float64) {
	exp := zoom.MantExp(nil)
	prec := precForExp(exp)
	minDim := w
	if h < minDim {
		minDim = h
	}
	pixelSizeBig := new(big.Float).SetPrec(prec)
	pixelSizeBig.Mul(zoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	pixelSizeBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), pixelSizeBig)
	dxBig := new(big.Float).SetPrec(prec).SetFloat64(tX - float64(w)/2)
	dyBig := new(big.Float).SetPrec(prec).SetFloat64(tY - float64(h)/2)
	dxBig.Mul(dxBig, pixelSizeBig)
	dyBig.Mul(dyBig, pixelSizeBig)
	cx.Add(cx, dxBig)
	cy.Add(cy, dyBig)
}

// ─────────────────────────────────────────────
//  Abort listener
// ─────────────────────────────────────────────

func startAbortListener() (stop func()) {
	atomic.StoreInt32(&minibrotAbort, 0)
	done := make(chan struct{})
	go func() {
		buf := [1]byte{}
		for {
			select {
			case <-done:
				return
			default:
			}
			ttyFile.SetDeadline(time.Now().Add(50 * time.Millisecond))
			n, _ := ttyFile.Read(buf[:])
			if n > 0 && (buf[0] == 'q' || buf[0] == 'Q' || buf[0] == 3) {
				atomic.StoreInt32(&minibrotAbort, 1)
				return
			}
		}
	}()
	return func() {
		close(done)
		ttyFile.SetDeadline(time.Time{})
		atomic.StoreInt32(&minibrotAbort, 0)
	}
}

// ─────────────────────────────────────────────
//  Auto find minibrot
// ─────────────────────────────────────────────

func autoFindMinibrot() {
	const (
		searchW     = 160
		searchH     = 160
		maxSteps    = 80
		miniSteps   = 12
		foundThresh = 0.05
	)
	stop := startAbortListener()
	defer stop()
	miniZoomSteps := 0
	for step := 0; step < maxSteps; step++ {
		if atomic.LoadInt32(&minibrotAbort) != 0 {
			fmt.Print("\033[H\033[J\033[0mMinibrot search aborted.\n")
			time.Sleep(600 * time.Millisecond)
			return
		}
		data := renderScratch(searchW, searchH, cx, cy, zoom)
		tX, tY, zoomFactor := findBestTarget(data, searchW, searchH)
		if adaptIter {
			exp := zoom.MantExp(nil)
			if s := suggestMaxIter(exp); s > maxIter {
				maxIter = s
			}
		}
		exp := zoom.MantExp(nil)
		s := analyzeFrame(data, searchW, searchH)
		blackPct := 100.0 * float64(s.blackCount) / float64(s.totalPix)
		fmt.Printf("\033[H\033[2K\033[0mSearching... Step %d | Zoom:10^%.1f | Black:%.1f%% | q=abort\n",
			step+1, float64(exp)*0.30103, blackPct)
		drawTerminal()
		if zoomFactor < 1.0 {
			zoom.Quo(zoom, big.NewFloat(4.0))
			miniZoomSteps = 0
			continue
		}
		blackFrac := float64(s.blackCount) / float64(s.totalPix)
		if blackFrac > 0 && blackFrac < foundThresh {
			miniZoomSteps++
			if miniZoomSteps >= miniSteps {
				break
			}
		} else {
			miniZoomSteps = 0
		}
		stepPixelToWorld(searchW, searchH, tX, tY)
		zoom.Mul(zoom, big.NewFloat(zoomFactor))
	}
	drawTerminal()
}

// ─────────────────────────────────────────────
//  I Feel Lucky
// ─────────────────────────────────────────────

// luckySpot is a mathematically verified interesting coordinate.
// cx/cy are the exact landing point; no random jitter needed because
// each coordinate is already known to produce rich detail at depth.
// zoomStart is the zoom level to begin the display render from
// (we jump straight there instead of iteratively steering).
type luckySpot struct {
	name      string
	cx, cy    string // stored as strings for full precision via big.Float
	zoomStart int    // log10 of starting zoom (we've already "arrived")
	desc      string // shown to user
}

// knownSpots are verified deep-zoom starting coordinates.
// Rules for every entry:
//   • cy must be clearly non-zero — real-axis points have hair-thin boundaries
//     that the steering logic can't lock onto at moderate zoom
//   • The coordinate must produce visible set boundary at zoom ~1e5
//   • Each entry produces structurally distinct geometry
var knownSpots = []luckySpot{
	{
		// Classic seahorse valley — dense spiralling arms, the most
		// reliable deep-zoom target in the entire set.
		"Seahorse Valley",
		"-0.7436438870371587", "0.1318259042053119",
		5, "Spiralling seahorse arms",
	},
	{
		// Interior of the seahorse valley — tighter spirals, different rhythm.
		"Inner Seahorse",
		"-0.7453294693392004", "0.1130261184900640",
		5, "Tight nested spirals",
	},
	{
		// Elephant valley — bulbous tendrils, completely different from seahorse.
		"Elephant Valley",
		"0.3023030226852533", "0.0193789750809900",
		5, "Bulbous elephant-trunk tendrils",
	},
	{
		// Triple spiral near the period-3 bulb — three-armed rotational symmetry.
		"Triple Spiral",
		"-0.1010963638456220", "0.9562865108091414",
		5, "Three-armed rotational symmetry",
	},
	{
		// Siegel disk boundary — spirals with a near-circular core.
		"Siegel Disk",
		"-0.3905408702129490", "-0.5867879073469687",
		5, "Near-circular irrational spirals",
	},
	{
		// Lightning structure near the period-4 bulb junction.
		"Lightning",
		"-0.1606982465411292", "1.0347693581293520",
		5, "Jagged fractal lightning",
	},
	{
		// Quad spiral — four-armed symmetry, very dense detail.
		"Quad Spiral",
		"-0.1588686290637270", "1.0322889390399967",
		5, "Four-armed dense spirals",
	},
	{
		// Cauliflower region — clustered bulbs, very organic-looking.
		"Cauliflower",
		"0.3753275406287958", "0.1737329919827063",
		5, "Cauliflower-like clusters",
	},
	{
		// Deep branch near the main antenna — long arching filaments.
		// Offset from real axis so there is a proper boundary to track.
		"Antenna Branch",
		"-1.9854527064572060", "0.0000242333500000",
		5, "Long arching filaments",
	},
	{
		// Period-4 island — banded concentric rings.
		"Period-4 Island",
		"-1.3107472472472470", "0.0660026002600260",
		5, "Concentric banded rings",
	},
	{
		// Parabolic bifurcation point — funnel-shaped spirals.
		"Parabolic Funnel",
		"-1.2532395990251797", "0.0449927900000000",
		5, "Funnel-shaped spirals",
	},
	{
		// Mini-Mandelbrot inside seahorse — a complete baby set with its own
		// seahorse valley, producing fractal self-similarity.
		"Baby Mandelbrot",
		"-0.7557860690498086", "0.1053600000000000",
		5, "Fractal self-similar baby set",
	},
}

// iFeelLucky teleports to a known interesting coordinate and zooms to
// the user-specified depth using the autoFindMinibrot steering logic.
// This guarantees we always land on the actual boundary, never in void.
func iFeelLucky() {
	restoreTerminal()

	// ── Ask for depth ─────────────────────────────────────────────────────
	fmt.Print("\033[2K\r\033[0mTarget depth (e.g. 20 for 10^20, 50 for 10^50): ")
	depthExp, err := strconv.Atoi(readLine())
	// readLine() restores raw mode via setRaw() before returning.
	if err != nil || depthExp < 5 {
		depthExp = 20
	}
	if depthExp > 300 {
		depthExp = 300
	}

	stop := startAbortListener()
	defer stop()

	// ── Pick a spot ───────────────────────────────────────────────────────
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	spot := knownSpots[r.Intn(len(knownSpots))]

	fmt.Printf("\033[2K\r\033[0mWarping to %s — %s (q to abort)\n", spot.name, spot.desc)

	// Set position to the exact known coordinate.
	// Use high precision from the start so we don't lose bits during zoom.
	prec := precForExp(depthExp * 4) // 4× headroom
	var ok bool
	cx, ok = new(big.Float).SetPrec(prec).SetString(spot.cx)
	if !ok {
		cx = new(big.Float).SetPrec(prec).SetFloat64(0)
	}
	cy, ok = new(big.Float).SetPrec(prec).SetString(spot.cy)
	if !ok {
		cy = new(big.Float).SetPrec(prec).SetFloat64(0)
	}
	// Start at zoom 1e5 — shallow enough that the boundary is visible and
	// the steering logic has something to lock onto.
	zoom = new(big.Float).SetPrec(prec).SetFloat64(1e5)
	maxIter = 1000

	// ── Validate starting frame has a visible boundary ──────────────────
	// If the first scratch render is all-black or all-escaped, the steering
	// logic has nothing to lock onto and we'll drift into void.
	// Try up to 3 times with a small upward nudge before giving up.
	{
		validated := false
		nudgeFrac := new(big.Float).SetPrec(prec).SetFloat64(0.0)
		for attempt := 0; attempt < 4; attempt++ {
			testCy := new(big.Float).SetPrec(prec).Add(cy, nudgeFrac)
			data := renderScratch(128, 128, cx, testCy, zoom)
			black, escaped := 0, 0
			for _, v := range data {
				if v < 0 { black++ } else { escaped++ }
			}
			total := black + escaped
			blackFrac := float64(black) / float64(total)
			// Good frame: has both black and escaped pixels, boundary fraction 5-90%
			if black > 0 && escaped > 0 && blackFrac > 0.05 && blackFrac < 0.90 {
				cy = testCy
				validated = true
				break
			}
			// Nudge cy upward by 0.5% of visible height each attempt.
			pixSize := new(big.Float).SetPrec(prec)
			pixSize.Quo(new(big.Float).SetPrec(prec).SetFloat64(4.0), zoom)
			nudgeFrac.Add(nudgeFrac, new(big.Float).SetPrec(prec).Mul(pixSize, new(big.Float).SetFloat64(64.0)))
		}
		if !validated {
			// Fallback: pick a different spot from the list and retry once.
			fallbackIdx := (r.Intn(len(knownSpots)-1) + 1 + r.Intn(len(knownSpots))) % len(knownSpots)
			spot = knownSpots[fallbackIdx]
			cx, _ = new(big.Float).SetPrec(prec).SetString(spot.cx)
			cy, _ = new(big.Float).SetPrec(prec).SetString(spot.cy)
			fmt.Printf("\r\033[2K\033[0mFalling back to %s...", spot.name)
		}
	}

	// ── Zoom loop using boundary steering ─────────────────────────────────
	// Each step renders a small frame, steers toward the highest-iter
	// non-black pixel adjacent to the set boundary, then zooms in ×2.
	// ×2 per step is conservative enough that the boundary stays in frame
	// even at difficult spots.
	//
	// Steps to reach 10^depthExp from zoom=1e5:
	// zoom × 2^n = 10^depthExp  →  n = (depthExp-5) / log10(2) ≈ (depthExp-5)*3.32
	zoomPerStep := big.NewFloat(2.0)
	stepsNeeded := int(math.Ceil(float64(depthExp-5) / math.Log10(2.0)))
	if stepsNeeded < 8 {
		stepsNeeded = 8
	}

	// Track how many consecutive steps had no valid boundary.
	// If we lose the boundary for too long, zoom out and re-acquire.
	lostBoundaryCount := 0

	for step := 0; step < stepsNeeded; step++ {
		if atomic.LoadInt32(&minibrotAbort) != 0 {
			initTerminal()
			return
		}

		// Slightly larger frame in the final stretch for accuracy.
		scrW, scrH := 96, 96
		if step >= stepsNeeded-10 {
			scrW, scrH = 160, 160
		}

		data := renderScratch(scrW, scrH, cx, cy, zoom)

		// Check if we've lost the boundary (all black or all escaped).
		hasBlack, hasEscaped := false, false
		for _, v := range data {
			if v < 0 { hasBlack = true } else { hasEscaped = true }
			if hasBlack && hasEscaped { break }
		}

		if !hasBlack || !hasEscaped {
			lostBoundaryCount++
			if lostBoundaryCount >= 4 {
				// Zoom out 8× to re-acquire the boundary then continue.
				zoom.Quo(zoom, big.NewFloat(8.0))
				lostBoundaryCount = 0
				fmt.Printf("\r\033[2K\033[0m%s | Re-acquiring boundary...  ", spot.name)
				continue
			}
		} else {
			lostBoundaryCount = 0
		}

		tX, tY := steerToBoundary(data, scrW, scrH)
		stepPixelToWorld(scrW, scrH, tX, tY)
		zoom.Mul(zoom, zoomPerStep)

		if adaptIter {
			if exp := zoom.MantExp(nil); suggestMaxIter(exp) > maxIter {
				maxIter = suggestMaxIter(exp)
			}
		} else {
			maxIter = int(float64(maxIter) * 1.08)
			if maxIter > 50000 {
				maxIter = 50000
			}
		}

		exp := zoom.MantExp(nil)
		depth10 := int(float64(exp) * 0.30103)
		fmt.Printf("\r\033[2K\033[0m%s | 10^%d → 10^%d  (q to abort)",
			spot.name, depth10, depthExp)
	}

	fmt.Println()
	initTerminal()
}

// steerToBoundary returns the screen pixel that best represents the
// Mandelbrot boundary — the escaped pixel with the highest iteration
// count that sits directly adjacent to at least one black (interior) pixel.
//
// This is the core fix for "always goes to black": we explicitly forbid
// targeting black pixels, and require adjacency to black so we don't
// drift into featureless escaped regions either.
//
// If no black pixels exist yet (early zoom), we target the highest-iter
// escaped pixel — that's where the boundary will appear as we zoom in.
// If the frame is overwhelmingly black (>85%) we've overshot — we target
// the escaped pixel nearest to the screen centre to re-anchor.
func steerToBoundary(data []float64, w, h int) (tX, tY float64) {
	halfW := float64(w) / 2
	halfH := float64(h) / 2

	// Count black pixels and find max escaped value.
	blackCount := 0
	maxEscaped := 0.0
	for _, v := range data {
		if v < 0 {
			blackCount++
		} else if v > maxEscaped {
			maxEscaped = v
		}
	}

	total := w * h
	blackFrac := float64(blackCount) / float64(total)

	// ── Overshot into interior — find nearest escaped pixel to centre ─────
	if blackFrac > 0.85 {
		bestDist := 1e18
		tX, tY = halfW, halfH
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if data[y*w+x] >= 0 {
					dx := float64(x) - halfW
					dy := float64(y) - halfH
					if d := dx*dx + dy*dy; d < bestDist {
						bestDist = d
						tX, tY = float64(x), float64(y)
					}
				}
			}
		}
		return
	}

	// ── No black yet — chase highest-iter pixel (boundary approaching) ────
	if blackCount == 0 {
		tX, tY = halfW, halfH
		best := -1.0
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := data[y*w+x]
				// weight by iter value and slight centre pull
				dx := float64(x) - halfW
				dy := float64(y) - halfH
				score := v / (1.0 + math.Sqrt(dx*dx+dy*dy)*0.05)
				if score > best {
					best = score
					tX, tY = float64(x), float64(y)
				}
			}
		}
		return
	}

	// ── Normal case: find highest-iter pixel adjacent to black ────────────
	// "Adjacent" = any of the 8 neighbours is black.
	// We avoid the full searchR neighbourhood scan (O(w*h*r²)) and instead
	// use a single neighbour pass (O(w*h)) — fast and sufficient.
	bestScore := -1.0
	tX, tY = halfW, halfH

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := data[y*w+x]
			if v < 0 {
				continue // never target interior
			}

			// Check 8-connected neighbours for black pixel.
			hasBlackNeighbour := false
		neighbourCheck:
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx, ny := x+dx, y+dy
					if nx >= 0 && nx < w && ny >= 0 && ny < h {
						if data[ny*w+nx] < 0 {
							hasBlackNeighbour = true
							break neighbourCheck
						}
					}
				}
			}

			// Also accept pixels within 3 pixels of black — catches thin
			// boundary regions where direct adjacency might be sparse.
			if !hasBlackNeighbour {
				const r = 3
			outerCheck:
				for dy := -r; dy <= r; dy++ {
					for dx := -r; dx <= r; dx++ {
						nx, ny := x+dx, y+dy
						if nx >= 0 && nx < w && ny >= 0 && ny < h {
							if data[ny*w+nx] < 0 {
								hasBlackNeighbour = true
								break outerCheck
							}
						}
					}
				}
			}

			if !hasBlackNeighbour {
				continue
			}

			// Score: iteration value normalised to [0,1], with a mild
			// pull toward screen centre to avoid hugging edges.
			normV := v / (maxEscaped + 1e-9)
			cdx := float64(x) - halfW
			cdy := float64(y) - halfH
			centrePull := 1.0 / (1.0 + math.Sqrt(cdx*cdx+cdy*cdy)*0.06)
			score := normV * centrePull

			if score > bestScore {
				bestScore = score
				tX, tY = float64(x), float64(y)
			}
		}
	}

	// If nothing scored (shouldn't happen but be safe), return centre.
	if bestScore < 0 {
		tX, tY = halfW, halfH
	}
	return
}

// ─────────────────────────────────────────────
//  parsePosition — parse a coordinate string for the 'B' (goto) command.
//
//  Accepted formats (whitespace and commas flexible):
//    -0.7436 0.1318                     → cx=-0.7436  cy=0.1318  zoom=current
//    -0.7436, 0.1318                    → same
//    -0.7436+0.1318i                    → complex notation
//    -0.7436-0.1318i                    → negative imaginary
//    cx=-0.7436 cy=0.1318               → labelled
//    cx=-0.7436 cy=0.1318 zoom=1e50     → with zoom
//    -0.7436 0.1318 zoom=1e50           → positional + zoom
//    -0.7436 0.1318 1e50                → positional + bare zoom
//
//  Returns (cx, cy, zoom, ok).  zoom is the current global if not specified.
// ─────────────────────────────────────────────
func parsePosition(s string) (newCx, newCy, newZoom *big.Float, ok bool) {
	prec := uint(200)
	s = strings.TrimSpace(s)

	// Helper: parse a single token as big.Float.
	parseBF := func(tok string) (*big.Float, bool) {
		tok = strings.TrimRight(tok, "i") // strip trailing 'i' from imaginary
		f, _, err := big.ParseFloat(tok, 10, prec, big.ToNearestEven)
		return f, err == nil
	}

	// Default zoom = current global zoom.
	newZoom = new(big.Float).SetPrec(prec).Copy(zoom)

	// ── cx=... cy=... labelled ────────────────────────────────────────────
	if strings.Contains(s, "cx=") || strings.Contains(s, "cy=") {
		var cxTok, cyTok, zTok string
		for _, tok := range strings.Fields(strings.ReplaceAll(s, ",", " ")) {
			kv := strings.SplitN(tok, "=", 2)
			if len(kv) != 2 { continue }
			switch strings.ToLower(kv[0]) {
			case "cx": cxTok = kv[1]
			case "cy": cyTok = kv[1]
			case "zoom", "z": zTok = kv[1]
			}
		}
		cx2, okX := parseBF(cxTok)
		cy2, okY := parseBF(cyTok)
		if !okX || !okY { return }
		newCx, newCy = cx2, cy2
		if zTok != "" {
			if z, okZ := parseBF(zTok); okZ { newZoom = z }
		}
		ok = true
		return
	}

	// ── Complex notation: a+bi or a-bi ───────────────────────────────────
	// Detect by presence of 'i' at the end.
	if strings.HasSuffix(s, "i") || strings.HasSuffix(s, "I") {
		// Remove trailing 'i'
		cs := strings.TrimRight(strings.ToLower(s), "i")
		// Find last '+' or '-' that is the imaginary separator (not exponent).
		// Walk backward to find the sign.
		sep := -1
		for j := len(cs) - 1; j > 0; j-- {
			if (cs[j] == '+' || cs[j] == '-') && cs[j-1] != 'e' && cs[j-1] != 'E' {
				sep = j
				break
			}
		}
		if sep > 0 {
			cx2, okX := parseBF(cs[:sep])
			cy2, okY := parseBF(cs[sep:])
			if okX && okY {
				newCx, newCy = cx2, cy2
				ok = true
				return
			}
		}
	}

	// ── Positional: two (or three) whitespace/comma-separated tokens ──────
	parts := strings.FieldsFunc(strings.ReplaceAll(s, ",", " "), func(r rune) bool {
		return r == ' ' || r == '\t'
	})

	// Filter out empty tokens.
	var toks []string
	for _, p := range parts {
		if p != "" { toks = append(toks, p) }
	}

	if len(toks) < 2 { return }

	cx2, okX := parseBF(toks[0])
	cy2, okY := parseBF(toks[1])
	if !okX || !okY { return }
	newCx, newCy = cx2, cy2
	ok = true

	// Optional third token: bare zoom value or zoom=...
	for _, tok := range toks[2:] {
		kv := strings.SplitN(tok, "=", 2)
		zStr := ""
		if len(kv) == 2 && (strings.ToLower(kv[0]) == "zoom" || strings.ToLower(kv[0]) == "z") {
			zStr = kv[1]
		} else if len(kv) == 1 {
			zStr = kv[0]
		}
		if zStr != "" {
			if z, okZ := parseBF(zStr); okZ { newZoom = z }
		}
	}
	return
}
