// export.go — File export (PNG, PPM, MP4 animation) and bookmark persistence.
package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/big"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// ─────────────────────────────────────────────
//  PNG
// ─────────────────────────────────────────────

func savePNG(filename string, w, h int, data []float64) {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	lut := luts[currentPaletteName]
	colorScale := colorDensity * 0.01
	lutSizeF := float64(lutSize)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := data[y*w+x]
			if v < 0 {
				img.SetRGBA(x, y, color.RGBA{A: 255})
			} else {
				idx := valToIdx(v, colorScale, lutSizeF)
				c := lut.Colors[idx]
				img.SetRGBA(x, y, color.RGBA{R: c.R, G: c.G, B: c.B, A: 255})
			}
		}
	}
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println("PNG error:", err)
		return
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 4<<20)
	png.Encode(bw, img)
	bw.Flush()
}

func saveCurrentPNG() {
	w, rows := getTermSize()
	h := (rows - 2) * 2
	fmt.Printf("\n\033[0mRendering %dx%d PNG... ", w, h)
	data := renderMandelbrot(w, h, cx, cy, zoom)
	filename := fmt.Sprintf("mandelbrot_%d.png", time.Now().Unix())
	savePNG(filename, w, h, data)
	fmt.Printf("Saved %s\n", filename)
	time.Sleep(1500 * time.Millisecond)
}

// ─────────────────────────────────────────────
//  PPM — binary P6 (fast)
// ─────────────────────────────────────────────

func savePPM() {
	w, h := 1920, 1080
	fmt.Printf("\n\033[0mRendering %dx%d PPM (binary P6)... ", w, h)
	data := renderMandelbrot(w, h, cx, cy, zoom)
	lut := luts[currentPaletteName]

	filename := fmt.Sprintf("mandelbrot_%d.ppm", time.Now().Unix())
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 4<<20)
	fmt.Fprintf(bw, "P6\n%d %d\n255\n", w, h)

	colorScale := colorDensity * 0.01
	lutSizeF := float64(lutSize)
	rowBuf := make([]byte, w*3)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := data[y*w+x]
			if v < 0 {
				rowBuf[x*3] = 0; rowBuf[x*3+1] = 0; rowBuf[x*3+2] = 0
			} else {
				idx := valToIdx(v, colorScale, lutSizeF)
				c := lut.Colors[idx]
				rowBuf[x*3] = c.R; rowBuf[x*3+1] = c.G; rowBuf[x*3+2] = c.B
			}
		}
		bw.Write(rowBuf)
	}
	bw.Flush()
	fmt.Printf("Saved %s\n", filename)
	time.Sleep(2 * time.Second)
}

// ─────────────────────────────────────────────
//  Animation
// ─────────────────────────────────────────────

type animFrame struct {
	idx int
	rgb []byte
}

var ppmFramePool sync.Pool

// exportAnimation renders a smooth zoom-out (+1) or zoom-in (-1) MP4.
//
// Glitch-free design:
//   Every frame is rendered with renderInto — the exact same path as the
//   interactive display (Mariani-Silver + C row functions + perturbation).
//   Using a stripped-down path produced pixel values that differed subtly
//   between frames, which the video codec interpreted as motion and turned
//   into blocking/glitch artifacts.
//
//   The multiplier is capped to 1.05 max by default suggestion so adjacent
//   frames share >95% of their content, giving the codec almost nothing to
//   compress as "motion".
//
//   ffmpeg settings: -crf 18 (near-lossless), -preset slow, yuv444p
//   (no chroma subsampling) so colour boundaries stay sharp.
func exportAnimation(direction int) {
	restoreTerminal()
	dirLabel := "zoom-out"
	if direction < 0 {
		dirLabel = "zoom-in"
	}

	fmt.Printf("\033[2K\rExport %s — zoom multiplier per frame (0.01–0.05 recommended, e.g. 0.02): ", dirLabel)
	input := readLine()

	// Accept both "1.02" and "0.02" style input.
	// If user types a number < 1 we treat it as the fractional part (e.g. 0.02 → multiplier 1.02).
	// If they type >= 1 it's the direct multiplier.
	rawVal, err := strconv.ParseFloat(input, 64)
	if err != nil {
		rawVal = 1.02
	}
	var multiplier float64
	if rawVal < 1.0 && rawVal > 0 {
		multiplier = 1.0 + rawVal // 0.02 → 1.02
	} else {
		multiplier = rawVal
	}
	if multiplier <= 1.0 || multiplier > 2.0 {
		multiplier = 1.02
	}

	if zoom.Cmp(big.NewFloat(1.0)) <= 0 {
		fmt.Println("Already at base zoom.")
		time.Sleep(2 * time.Second)
		initTerminal()
		return
	}

	os.MkdirAll("frames", 0755)

	exp := zoom.MantExp(nil)
	lnZoom := float64(exp) * math.Ln2
	totalFrames := int(math.Ceil(lnZoom/math.Log(multiplier))) + 1
	w, h := 640, 480
	fmt.Printf("Rendering %d frames at %dx%d (multiplier %.4f)...\n",
		totalFrames, w, h, multiplier)

	mBig := big.NewFloat(multiplier)

	// Pre-build all zoom levels.
	frameZooms := make([]*big.Float, totalFrames)
	cur := new(big.Float).Copy(zoom)
	if direction > 0 {
		for i := 0; i < totalFrames; i++ {
			frameZooms[i] = new(big.Float).Copy(cur)
			cur.Quo(cur, mBig)
		}
	} else {
		frameZooms[totalFrames-1] = new(big.Float).Copy(zoom)
		for i := totalFrames - 2; i >= 0; i-- {
			frameZooms[i] = new(big.Float).Quo(frameZooms[i+1], mBig)
		}
	}

	// ── CPU budget ────────────────────────────────────────────────────────
	// renderInto already uses all CPUs internally via its own worker pool,
	// so we render frames sequentially — parallelising would over-subscribe.
	// The write pipeline still overlaps I/O with the next render.
	ncpu := runtime.NumCPU()
	_ = ncpu // used below for pool sizing hint

	frameBytes := w * h * 3
	ppmFramePool = sync.Pool{New: func() interface{} {
		return make([]byte, frameBytes)
	}}

	lut := luts[currentPaletteName]
	colorScale := colorDensity * 0.01
	lutSizeF := float64(lutSize)
	captureCx := new(big.Float).Copy(cx)
	captureCy := new(big.Float).Copy(cy)

	// Buffered channel so the writer can work on frame N while frame N+1 renders.
	frameCh := make(chan animFrame, 4)

	// ── Writer goroutine ──────────────────────────────────────────────────
	var writeWg sync.WaitGroup
	var writtenCount int64
	writeWg.Add(1)
	go func() {
		defer writeWg.Done()
		ppmHdr := []byte(fmt.Sprintf("P6\n%d %d\n255\n", w, h))
		for frame := range frameCh {
			filename := fmt.Sprintf("frames/frame_%04d.ppm", frame.idx)
			f, ferr := os.Create(filename)
			if ferr == nil {
				bw := bufio.NewWriterSize(f, 4<<20)
				bw.Write(ppmHdr)
				bw.Write(frame.rgb)
				bw.Flush()
				f.Close()
			}
			ppmFramePool.Put(frame.rgb)
			n := atomic.AddInt64(&writtenCount, 1)
			fmt.Printf("\r  Frame %d/%d written...  ", n, totalFrames)
		}
	}()

	// ── Pause/resume listener ─────────────────────────────────────────────
	// Runs in background during animation.  Space toggles pause; 'q' aborts.
	atomic.StoreInt32(&animPaused, 0)
	animAbort := make(chan struct{})
	var listenerWg sync.WaitGroup
	listenerWg.Add(1)
	go func() {
		defer listenerWg.Done()
		for {
			select {
			case <-animAbort:
				return
			default:
			}
			c := getChar()
			if c == ' ' {
				if atomic.LoadInt32(&animPaused) == 0 {
					atomic.StoreInt32(&animPaused, 1)
					fmt.Print("\n  ⏸  Paused — press Space to resume, 'q' to cancel\n")
				} else {
					atomic.StoreInt32(&animPaused, 0)
					fmt.Print("\n  ▶  Resuming...\n")
				}
			} else if c == 'q' || c == 3 {
				atomic.StoreInt32(&animPaused, 0) // unblock render loop
				fmt.Print("\n  Cancelling animation...\n")
				// Signal abort by closing the abort channel.
				select {
				case <-animAbort: // already closed
				default:
					close(animAbort)
				}
				return
			}
		}
	}()

	// ── Render loop — sequential, uses full renderInto path ───────────────
	// renderInto is the same function used for interactive display.
	// It applies Mariani-Silver, perturbation theory, everything.
	// This guarantees pixel values are consistent frame-to-frame so the
	// video codec sees smooth motion rather than random pixel noise.
	aborted := false
	for i := 0; i < totalFrames; i++ {
		// ── Abort check ──────────────────────────────────────────────────
		select {
		case <-animAbort:
			aborted = true
		default:
		}
		if aborted {
			break
		}

		// ── Pause: spin until resumed or aborted ─────────────────────────
		for atomic.LoadInt32(&animPaused) != 0 {
			select {
			case <-animAbort:
				aborted = true
			default:
				runtime.Gosched()
			}
			if aborted {
				break
			}
		}
		if aborted {
			break
		}

		fmt.Printf("\r  Rendering frame %d/%d...  ", i+1, totalFrames)

		// Use the full render path — same as drawTerminal.
		// useScratch=true so animation frames never clobber the interactive display buffer.
		data := renderInto(w, h, captureCx, captureCy, frameZooms[i], true)

		// Colourmap float64 data → RGB bytes.
		rgb := ppmFramePool.Get().([]byte)
		for j := 0; j < w*h; j++ {
			v := data[j]
			o := j * 3
			if v < 0 {
				rgb[o] = 0; rgb[o+1] = 0; rgb[o+2] = 0
			} else {
				idx := valToIdx(v, colorScale, lutSizeF)
				c := lut.Colors[idx]
				rgb[o] = c.R; rgb[o+1] = c.G; rgb[o+2] = c.B
			}
		}
		frameCh <- animFrame{idx: i, rgb: rgb}
	}
	close(frameCh)
	writeWg.Wait()

	// Stop the keypress listener.
	select {
	case <-animAbort: // already closed by 'q' press
	default:
		close(animAbort)
	}
	listenerWg.Wait()
	atomic.StoreInt32(&animPaused, 0)

	if aborted {
		fmt.Println("\n  Animation cancelled.")
		time.Sleep(2 * time.Second)
		initTerminal()
		return
	}

	// ── ffmpeg encode ─────────────────────────────────────────────────────
	// -crf 18        near-lossless quality (0=lossless, 23=default, 51=worst)
	// -preset slow   better compression efficiency vs ultrafast
	// -pix_fmt yuv444p  no chroma subsampling — keeps colour edges sharp
	//                   (yuv420p halves colour resolution, causing colour fringing)
	// -vf scale      ensure dimensions are even (required by libx264)
	outFile := fmt.Sprintf("%s_%d.mp4", dirLabel, time.Now().Unix())
	fmt.Printf("\n  Encoding %s...\n", outFile)
	cmd := exec.Command("ffmpeg", "-y",
		"-framerate", "30",
		"-i", "frames/frame_%04d.ppm",
		"-c:v", "libx264",
		"-preset", "slow",
		"-crf", "18",
		"-pix_fmt", "yuv444p",
		"-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
		outFile,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err == nil {
		fmt.Println("\nVideo saved to", outFile)
		os.RemoveAll("frames")
	} else {
		fmt.Println("\nffmpeg error:", err)
	}
	time.Sleep(3 * time.Second)
	initTerminal()
}

// ─────────────────────────────────────────────
//  Bookmarks
// ─────────────────────────────────────────────

func saveBookmark(slot int) {
	if slot < 1 || slot > 9 {
		return
	}
	exp := zoom.MantExp(nil)
	bookmarks[slot] = Bookmark{
		Name:    fmt.Sprintf("Slot %d", slot),
		Cx:      cx.Text('e', 40),
		Cy:      cy.Text('e', 40),
		ZoomExp: exp,
		MaxIter: maxIter,
	}
	fmt.Printf("\033[H\033[J\033[0mBookmark saved to slot %d\n", slot)
	time.Sleep(500 * time.Millisecond)
}

func loadBookmark(slot int) {
	if slot < 1 || slot > 9 {
		return
	}
	b := bookmarks[slot]
	if b.Cx == "" {
		fmt.Printf("\033[H\033[J\033[0mSlot %d is empty.\n", slot)
		time.Sleep(500 * time.Millisecond)
		return
	}
	prec := precForExp(b.ZoomExp)
	cx, _ = new(big.Float).SetPrec(prec).SetString(b.Cx)
	cy, _ = new(big.Float).SetPrec(prec).SetString(b.Cy)
	zoom = new(big.Float).SetPrec(prec).SetMantExp(new(big.Float).SetFloat64(1), b.ZoomExp)
	maxIter = b.MaxIter
}
