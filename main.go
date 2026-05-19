// main.go — Program entry point and main input loop.
//
// Build:  go build .
// Run:    ./mandelbrot
//
// All fractal math lives in render.go.
// All display logic lives in terminal.go.
// Palette/color code lives in palette.go.
// Minibrot search lives in finder.go.
// File export lives in export.go.
// Shared globals live in globals.go.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/big"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// readLine switches the tty to canonical mode, reads a line, then restores
// raw mode. Uses the same ttyFile fd as getChar() — no subprocess, no race.
func readLine() string {
    setCanonical()
    var out []byte
    b := make([]byte, 1)
    for {
        n, err := ttyFile.Read(b)
        if n > 0 {
            if b[0] == '\n' || b[0] == '\r' {
                break
            }
            out = append(out, b[0])
        }
        if err != nil {
            break
        }
    }
    setRaw()
    return strings.TrimSpace(string(out))
}

// shiftKeys maps shift+digit terminal bytes to bookmark slot numbers 1–9.
var shiftKeys = map[byte]int{
	'!': 1, '@': 2, '#': 3, '$': 4, '%': 5,
	'^': 6, '&': 7, '*': 8, '(': 9,
}

func main() {
	// ── --no-interactive mode ──────────────────────────────────────────────
	// Usage: ./mandelbrot --no-interactive --cx=-0.7435669 --cy=0.1314023 --zoom=1e20 [--iter=5000] [--width=160] [--height=90]
	noInteractive := flag.Bool("no-interactive", false, "Render a single frame and print timing, then exit")
	flagCx   := flag.String("cx",    "-0.743643887037151",  "Center X coordinate")
	flagCy   := flag.String("cy",    "0.131825904205330",   "Center Y coordinate")
	flagZoom := flag.String("zoom",  "1e20",                "Zoom level (e.g. 1e20)")
	flagIter := flag.Int("iter",     0,                     "Max iterations (0 = auto)")
	flagW    := flag.Int("width",    160,                   "Render width in pixels")
	flagH    := flag.Int("height",   90,                    "Render height in pixels")
	flagReps := flag.Int("repeat",   5,                     "Number of timed renders to average")
	flag1080p := flag.Bool("1080p",  false,                 "Shorthand for --width=1920 --height=1080")
	flag720p  := flag.Bool("720p",   false,                 "Shorthand for --width=1280 --height=720")
	flag480p  := flag.Bool("480p",   false,                 "Shorthand for --width=640 --height=480")
	flagPic  := flag.String("pic",   "",                    "Save rendered image to this PNG file path")
	flag.Parse()

	// Apply resolution presets (last one wins if multiple given)
	if *flag480p  { *flagW = 640;  *flagH = 480  }
	if *flag720p  { *flagW = 1280; *flagH = 720  }
	if *flag1080p { *flagW = 1920; *flagH = 1080 }

	if *noInteractive {
		runtime.GOMAXPROCS(runtime.NumCPU())
		initLUTs()

		prec := uint(200)
		cx = new(big.Float).SetPrec(prec)
		cy = new(big.Float).SetPrec(prec)
		zoom = new(big.Float).SetPrec(prec)

		posStr := *flagCx + " " + *flagCy + " zoom=" + *flagZoom
		newCx, newCy, newZoom, ok := parsePosition(posStr)
		if !ok {
			fmt.Fprintf(os.Stderr, "Failed to parse coordinates: cx=%s cy=%s zoom=%s\n", *flagCx, *flagCy, *flagZoom)
			os.Exit(1)
		}
		cx.Copy(newCx); cy.Copy(newCy); zoom.Copy(newZoom)

		if *flagIter > 0 {
			maxIter = *flagIter
		} else {
			exp := zoom.MantExp(nil)
			maxIter = suggestMaxIter(exp)
		}

		w, h := *flagW, *flagH
		reps := *flagReps
		if reps < 1 { reps = 1 }

		renderData = make([]float64, w*h)
		renderDataW, renderDataH = w, h

		exp := zoom.MantExp(nil)
		depth10 := float64(exp) * 0.30103
		fmt.Printf("cx=%s  cy=%s\n", cx.Text('g', 20), cy.Text('g', 20))
		fmt.Printf("zoom=10^%.1f  iter=%d  size=%dx%d  workers=%d\n",
			depth10, maxIter, w, h, s20feWorkerCount(exp > 43))

		// Cold render (builds reference orbit cache)
		coldStart := time.Now()
		data := renderMandelbrot(w, h, cx, cy, zoom)
		cold := time.Since(coldStart)

		escaped := 0
		for _, v := range data {
			if v >= 0 { escaped++ }
		}
		fmt.Printf("cold render:  %v  (cache build + frame)  — %d/%d pixels escaped\n",
			cold, escaped, w*h)

		// Warm renders (cache hit; only perturbation + C inner loop)
		var total time.Duration
		var best time.Duration = 1<<62
		for i := 0; i < reps; i++ {
			start := time.Now()
			renderMandelbrot(w, h, cx, cy, zoom)
			d := time.Since(start)
			total += d
			if d < best { best = d }
		}
		avg := total / time.Duration(reps)
		fmt.Printf("warm renders: best=%v  avg=%v  (%d runs)\n", best, avg, reps)
		fmt.Printf("target 0.1s:  %.1f%% of budget used (warm avg)\n",
			float64(avg.Nanoseconds())/float64((100*time.Millisecond).Nanoseconds())*100)

		// Save PNG if --pic was given
		if *flagPic != "" {
			fmt.Printf("saving image to %s ...\n", *flagPic)
			savePNG(*flagPic, w, h, data)
			fmt.Printf("saved %s (%dx%d)\n", *flagPic, w, h)
		}
		os.Exit(0)
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	openTTY()
	initLUTs()
	initTerminal()
	defer restoreTerminal()

	// Initial view: classic overview of the full Mandelbrot set.
	zoomSteps = 0
	cx   = new(big.Float).SetPrec(128).SetFloat64(-0.75)
	cy   = new(big.Float).SetPrec(128).SetFloat64(0)
	zoom = new(big.Float).SetPrec(128).SetFloat64(1)

	for {
		// ── Precision upkeep ────────────────────────────────────────────────
		// Recompute zoom precisely from the integer step count every frame.
		// This is the only correct approach — accumulated multiplications at
		// any precision will drift. Integer step count + fresh exponentiation
		// has zero accumulated error.
		recomputeZoom()

		// Auto-scale maxIter with zoom depth when enabled.
		if adaptIter {
			exp := zoom.MantExp(nil)
			if s := suggestMaxIter(exp); s > maxIter {
				maxIter = s
				cachedRefEscAt = -1 // maxIter grew — orbit may be too short
			}
		}

		drawTerminal()
		c := getChar()

		// ── Quit ────────────────────────────────────────────────────────────
		if c == 'q' || c == 3 { // 3 = ctrl-c
			fmt.Print("\r\n\033[0m\033[KQuit? (y/n): ")
			if ans := getChar(); ans == 'y' || ans == 'Y' {
					restoreTerminal()
					shutdownOpenCL()
					fmt.Print("\033[0m\033[H\033[J")
					os.Exit(0)
				}
			continue
		}

		// ── Bookmark load (keys 1–9) ─────────────────────────────────────────
		if c >= '1' && c <= '9' {
			loadBookmark(int(c - '0'))
			continue
		}
		// ── Bookmark save (shift+1 through shift+9) ──────────────────────────
		if slot, ok := shiftKeys[c]; ok {
			saveBookmark(slot)
			continue
		}

		// Movement step size: 20% of the visible width at the current zoom.
		// Precision is already guaranteed correct by the upkeep block above.
		moveAmount := new(big.Float).SetPrec(cx.Prec()).SetFloat64(0.2)
		moveAmount.Quo(moveAmount, zoom)

		switch c {

		// ── Navigation ────────────────────────────────────────────────────────
		case 'w':
			cy.Sub(cy, moveAmount)
			invalidatePrefetch()
		case 's':
			cy.Add(cy, moveAmount)
			invalidatePrefetch()
		case 'a':
			cx.Sub(cx, moveAmount)
			invalidatePrefetch()
		case 'd':
			cx.Add(cx, moveAmount)
			invalidatePrefetch()

		// ── Zoom ──────────────────────────────────────────────────────────────
		case 'z':
			zoomSteps++
			recomputeZoom()
			cachedRefEscAt = -1 // force fresh probe point selection at new depth
		case 'x':
			zoomSteps--
			recomputeZoom()
			invalidatePrefetch()
		case 'Z':
			zoomSteps += 5
			recomputeZoom()
			cachedRefEscAt = -1
			invalidatePrefetch()
		case 'X':
			zoomSteps -= 5
			recomputeZoom()
			invalidatePrefetch()

		// ── Random colour palette ───────────────────────────────────────────
		case 'r':
			fmt.Print("\033[2K\rHow many colors? (2-32, default 6): ")
			n := 6
			if nStr := readLine(); nStr != "" {
				if parsed, err := strconv.Atoi(nStr); err == nil {
					if parsed < 2 {
						parsed = 2
					} else if parsed > 32 {
						parsed = 32
					}
					n = parsed
				}
			}
			seed := time.Now().UnixNano()
			hexColors := generateRandomPalette(n, seed)
			pasteStr := strings.Join(hexColors, ",")
			fmt.Printf("\033[2K\rRandom palette applied — %d colors:\n", n)
			for _, h := range hexColors {
				fmt.Printf("  %s\n", h)
			}
			fmt.Printf("\nReady to paste (for 'n' key):\n  %s\n", pasteStr)
			fmt.Print("\nPress any key to continue.")
			getChar()

		// ── Reset view — requires confirmation so deep zooms aren’t lost ──────
		case 'R':
			exp := zoom.MantExp(nil)
			depth10 := int(float64(exp) * 0.30103)
			fmt.Printf("\033[2K\rReset view from 10^%d to start? (y/N): ", depth10)
			if ans := getChar(); ans == 'y' || ans == 'Y' {
				cx.SetFloat64(-0.75)
				cy.SetFloat64(0)
				zoomSteps = 0
				recomputeZoom()
				maxIter = 500
			}

		// ── Iteration count ───────────────────────────────────────────────────
		case 'i':
			maxIter *= 2
			invalidatePrefetch()
		case 'o':
			maxIter /= 2
			if maxIter < 50 {
				maxIter = 50
			}
			invalidatePrefetch()
		case 'I':
			maxIter *= 8
			invalidatePrefetch()
		case 'O':
			maxIter /= 8
			if maxIter < 50 {
				maxIter = 50
			}
			invalidatePrefetch()

		// ── Color density ─────────────────────────────────────────────────────
		case 'k':
			colorDensity *= 1.2
		case 'l':
			colorDensity /= 1.2
		case 'K':
			colorDensity *= 5
		case 'L':
			colorDensity /= 5

		// ── Palette ───────────────────────────────────────────────────────────
		case 'c':
			idx := 0
			for i, k := range paletteKeys {
				if k == currentPaletteName {
					idx = i
					break
				}
			}
			currentPaletteName = paletteKeys[(idx+1)%len(paletteKeys)]
			prevW = 0
			if use16Color { buildLut16Cache() }
			invalidatePrefetch()

		// ── Feature toggles ───────────────────────────────────────────────────
		case 'A':
			adaptIter = !adaptIter
		case 'e':
			histoEQ = !histoEQ
			prevW = 0
			invalidatePrefetch()

		// ── Pre-compute zoom cache ─────────────────────────────────────────────
		case 'C':
			fmt.Print("\033[2K\rPre-compute how many zooms? (1-20): ")
			nFrames := 5 // sensible default
			if line := readLine(); line != "" {
				if parsed, err := strconv.Atoi(line); err == nil {
					nFrames = parsed
				}
			}
			initTerminal()
			startPrefetch(nFrames)

		// ── Julia mode ────────────────────────────────────────────────────────
		case 'J':
			juliaMode = !juliaMode
		case ',':
			juliaR -= 0.01
		case '.':
			juliaR += 0.01
		case ';':
			juliaI -= 0.01
		case '\'':
			juliaI += 0.01

		// ── Search / warp ─────────────────────────────────────────────────────
		case 'v':
			autoFindMinibrot()
		case 'g':
			iFeelLucky()

		// ── Goto position ─────────────────────────────────────────────────────
		// Press 'G', paste coordinates in any of these formats, press Enter:
		//   -0.7436 0.1318
		//   -0.7436, 0.1318
		//   -0.7436+0.1318i
		//   -0.7436-0.1318i
		//   cx=-0.7436 cy=0.1318
		// Optionally append zoom:  ..., zoom=1e50  or  ..., z=1e50
		case 'B':
			fmt.Print("\033[2K\r\033[0mGoto position (cx cy [zoom]):\n> ")
			line := readLine()
			if newCx, newCy, newZoom, ok := parsePosition(line); ok {
				// Compute precision from the target zoom depth
				newExp  := newZoom.MantExp(nil)
				newPrec := precForExp(newExp)
				if newPrec < 128 { newPrec = 128 }
				cx   = new(big.Float).SetPrec(newPrec).Copy(newCx)
				cy   = new(big.Float).SetPrec(newPrec).Copy(newCy)
				zoom = new(big.Float).SetPrec(newPrec).Copy(newZoom)
				// Compute closest step count so zoom keys continue to work
				// correctly from here. zoomSteps = round(log(zoom)/log(1.5))
				zoomF64, _ := zoom.Float64()
				if zoomF64 > 0 {
					zoomSteps = int(math.Log(zoomF64) / math.Log(1.5))
				} else {
					zoomSteps = 0
				}
				// Don't call recomputeZoom() here — the user's exact value
				// is more precise than what 1.5^steps would give.
				// recomputeZoom() will be called next loop iteration but we
				// guard it: only recompute if zoomSteps actually changed.
				invalidatePrefetch()
				fmt.Printf("\033[2K\rJumped to cx=%s  cy=%s  zoom=10^%.1f\n",
					cx.Text('g', 20), cy.Text('g', 20),
					float64(zoom.MantExp(nil))*0.30103)
				fmt.Print("Press any key to continue.")
				getChar()
			} else {
				fmt.Print("\033[2K\rCould not parse coordinates. Format: '-0.7436 0.1318' or '-0.7436+0.1318i'\nPress any key.")
				getChar()
			}

		// ── Echo position ─────────────────────────────────────────────────────
		// Prints full-precision coordinates to stdout in a format that can be
		// pasted back into 'G' (goto) or saved in a text file.
		case 'E':
			cxStr := cx.Text('g', 40)
			cyStr := cy.Text('g', 40)
			zoomStr := zoom.Text('g', 40)
			exp := zoom.MantExp(nil)
			depth10 := float64(exp) * 0.30103
			fmt.Printf("\n\033[0m── Current position ──────────────────────────────\n")
			fmt.Printf("  cx   = %s\n", cxStr)
			fmt.Printf("  cy   = %s\n", cyStr)
			fmt.Printf("  zoom = %s  (≈10^%.2f)\n", zoomStr, depth10)
			fmt.Printf("  iter = %d\n", maxIter)
			fmt.Printf("\n  Paste-ready (for 'B' goto):\n")
			fmt.Printf("  %s %s zoom=%s\n", cxStr, cyStr, zoomStr)
			fmt.Printf("──────────────────────────────────────────────────\n")
			fmt.Print("Press any key to continue.")
			getChar()

		// ── OpenCL GPU toggle ─────────────────────────────────────────────────
		case 'G':
			if !useOpenCL {
				fmt.Print("\r\n\033[0mInitialising OpenCL GPU... ")
				if err := initOpenCL(); err != nil {
					fmt.Printf("FAILED: %v\n", err)
					fmt.Print("Press any key to continue (CPU mode).")
					getChar()
				} else {
					fmt.Printf("OK — %s\n", oclDeviceName())
					fmt.Print("Press any key to continue.")
					getChar()
					useOpenCL = true
				}
			} else {
				useOpenCL = false
			}

		// ── Help ──────────────────────────────────────────────────────────────
		case 'h':
			showHelp = !showHelp

		// ── Export ────────────────────────────────────────────────────────────
		case 'P':
			savePPM()
		case 'p':
			saveCurrentPNG()
		case 'M':
			exportAnimation(+1) // zoom-out
		case 'F':
			exportAnimation(-1) // zoom-in

		// ── Custom palette ────────────────────────────────────────────────────
		case 'n':
			fmt.Print("\033[2K\rHex colours (e.g. #ff0000,#00ff00,#0000ff): ")
			parseHexPalette(readLine())

		// ── Zoom-out terminal playback ────────────────────────────────────────
		// Pre-renders all frames from current zoom back to zoom=1, then plays
		// them back live in the terminal at the chosen FPS — same effect as
		// pressing M (export zoom-out MP4) but instant in-terminal playback.
		case 'W':
			mult := 1.02
			fps := 15
			fmt.Print("\033[2K\rZoom multiplier per frame (e.g. 0.02 = 1.02x, default): ")
			if mStr := readLine(); mStr != "" {
				if parsed, err := strconv.ParseFloat(mStr, 64); err == nil {
					if parsed > 0 && parsed < 1.0 {
						mult = 1.0 + parsed
					} else if parsed >= 1.0 && parsed <= 2.0 {
						mult = parsed
					}
				}
			}
			fmt.Print("\033[2K\rPlayback FPS (e.g. 10, 24, 30, default 15): ")
			if fStr := readLine(); fStr != "" {
				if parsed, err := strconv.Atoi(fStr); err == nil && parsed > 0 {
					fps = parsed
				}
			}
			startZoomOutPlayback(mult, fps)
		case 't', 'T':
    			use16Color = !use16Color
    			prevW = 0
			if use16Color { buildLut16Cache() }
		}
	}
}


