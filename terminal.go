// terminal.go — Terminal setup, teardown, size detection, and the main
// display loop that renders each frame to stdout as a single write.
package main

import (
    "fmt"
    "os"
    "strconv"
    "strings"
    "sync/atomic"
    "syscall"
    "unsafe"
)

// ─────────────────────────────────────────────
//  ARM64 Linux termios constants (hardcoded stable ABI values).
//  Using raw numbers avoids missing syscall.TCGETS etc. on Android/Termux.
// ─────────────────────────────────────────────
const (
    tcgets      uintptr = 0x5401
    tcsets      uintptr = 0x5402
    icanon      uint32  = 0x2
    termEcho    uint32  = 0x8
    echoe       uint32  = 0x10
    echok       uint32  = 0x20
    echonl      uint32  = 0x40
    vmin                = 6
    vtime               = 5
    tiocgwinsz  uintptr = 0x5413
)

// termios mirrors the kernel struct termios for ARM64 Linux.
type termios struct {
    Iflag uint32
    Oflag uint32
    Cflag uint32
    Lflag uint32
    Line  uint8
    Cc    [19]uint8
    _     [3]byte
}

var ttyFd   int = -1
var ttyFile *os.File
var savedTermios termios

func openTTY() {
    f, err := os.OpenFile("/dev/tty", syscall.O_RDWR|syscall.O_NOCTTY, 0)
    if err != nil {
        ttyFd  = int(os.Stdin.Fd())
        ttyFile = os.Stdin
    } else {
        ttyFd  = int(f.Fd())
        ttyFile = f
    }
    syscall.Syscall(syscall.SYS_IOCTL, uintptr(ttyFd),
        tcgets, uintptr(unsafe.Pointer(&savedTermios)))
}

func setRaw() {
    t := savedTermios
    t.Lflag &^= icanon | termEcho | echoe | echok | echonl
    t.Cc[vmin]  = 1
    t.Cc[vtime] = 0
    syscall.Syscall(syscall.SYS_IOCTL, uintptr(ttyFd),
        tcsets, uintptr(unsafe.Pointer(&t)))
}

func setCanonical() {
    t := savedTermios
    t.Lflag |= icanon | termEcho
    t.Cc[vmin]  = 1
    t.Cc[vtime] = 0
    syscall.Syscall(syscall.SYS_IOCTL, uintptr(ttyFd),
        tcsets, uintptr(unsafe.Pointer(&t)))
}

func initTerminal()    { setRaw() }
func restoreTerminal() {
    syscall.Syscall(syscall.SYS_IOCTL, uintptr(ttyFd),
        tcsets, uintptr(unsafe.Pointer(&savedTermios)))
}

func getChar() byte {
    b := [1]byte{}
    ttyFile.Read(b[:])
    return b[0]
}

type winsize struct {
    Row, Col       uint16
    Xpixel, Ypixel uint16
}

func getTermSize() (w, h int) {
    var ws winsize
    _, _, errno := syscall.Syscall(syscall.SYS_IOCTL,
        uintptr(ttyFd), tiocgwinsz,
        uintptr(unsafe.Pointer(&ws)))
    if errno != 0 || ws.Col < 4 || ws.Row < 4 {
        return 80, 24
    }
    return int(ws.Col), int(ws.Row)
}

// ─────────────────────────────────────────────
//  16-Color Support
// ─────────────────────────────────────────────

// use16Color toggles the 16-color rendering mode (wire this to 'T' in your main.go input switch)
var use16Color = false

// lut16Cache caches the 16-color index mapping; rebuilt only when palette or mode changes.
var lut16Cache []int
var lut16PalName string // palette name when lut16Cache was last built

// buildLut16Cache rebuilds lut16Cache for the current palette. Call whenever
// use16Color is toggled on or the palette changes.
func buildLut16Cache() {
    lut := luts[currentPaletteName]
    if lut == nil {
        return
    }
    if len(lut16Cache) != lutSize {
        lut16Cache = make([]int, lutSize)
    }
    for i := 0; i < lutSize; i++ {
        r, g, b := extractRGBFromSeq(lut.BG[i])
        lut16Cache[i] = findClosest16Color(r, g, b)
    }
    lut16PalName = currentPaletteName
}

// standard16RGB defines the standard xterm 16-color palette RGB values
var standard16RGB = [16][3]int{
    {0, 0, 0},       // 0 Black
    {128, 0, 0},     // 1 Red
    {0, 128, 0},     // 2 Green
    {128, 128, 0},   // 3 Yellow
    {0, 0, 128},     // 4 Blue
    {128, 0, 128},   // 5 Magenta
    {0, 128, 128},   // 6 Cyan
    {192, 192, 192}, // 7 White
    {128, 128, 128}, // 8 Bright Black
    {255, 0, 0},     // 9 Bright Red
    {0, 255, 0},     // 10 Bright Green
    {255, 255, 0},   // 11 Bright Yellow
    {0, 0, 255},     // 12 Bright Blue
    {255, 0, 255},   // 13 Bright Magenta
    {0, 255, 255},   // 14 Bright Cyan
    {255, 255, 255}, // 15 Bright White
}

// color256ToRGB converts a standard 256-color terminal index to approximate RGB
func color256ToRGB(i int) (r, g, b int) {
    if i < 16 {
        return standard16RGB[i][0], standard16RGB[i][1], standard16RGB[i][2]
    }
    if i < 232 {
        i -= 16
        r = (i / 36) * 51
        g = ((i / 6) % 6) * 51
        b = (i % 6) * 51
        return
    }
    // Grayscale
    i -= 232
    v := i*10 + 8
    return v, v, v
}

// findClosest16Color finds the nearest standard 16-color ANSI index for a given RGB value
func findClosest16Color(r, g, b int) int {
    minDist := 1<<31 - 1
    best := 0
    for i, c := range standard16RGB {
        dr := r - c[0]
        dg := g - c[1]
        db := b - c[2]
        dist := dr*dr + dg*dg + db*db
        if dist < minDist {
            minDist = dist
            best = i
        }
    }
    return best
}

// extractRGBFromSeq parses the RGB values out of an ANSI escape sequence.
// It supports truecolor (\033[...;2;R;G;Bm) and 256-color (\033[...;5;Im).
func extractRGBFromSeq(seq []byte) (r, g, b int) {
    s := string(seq)
    // Look for truecolor ;2;R;G;Bm
    if idx := strings.Index(s, ";2;"); idx != -1 {
        parts := strings.Split(s[idx+3:len(s)-1], ";")
        if len(parts) >= 3 {
            r, _ = strconv.Atoi(parts[0])
            g, _ = strconv.Atoi(parts[1])
            b, _ = strconv.Atoi(parts[2])
        }
        return
    }
    // Look for 256-color ;5;Im
    if idx := strings.Index(s, ";5;"); idx != -1 {
        i, _ := strconv.Atoi(s[idx+3 : len(s)-1])
        return color256ToRGB(i)
    }
    // Fallback to black
    return 0, 0, 0
}

// fg16 and bg16 return classic ANSI escape sequences for the given 16-color index.
func fg16(idx int) []byte {
    if idx < 8 {
        return []byte(fmt.Sprintf("\033[%dm", 30+idx))
    }
    return []byte(fmt.Sprintf("\033[%dm", 90+idx-8))
}

func bg16(idx int) []byte {
    if idx < 8 {
        return []byte(fmt.Sprintf("\033[%dm", 40+idx))
    }
    return []byte(fmt.Sprintf("\033[%dm", 100+idx-8))
}

// ─────────────────────────────────────────────
//  Frame renderer
// ─────────────────────────────────────────────

// drawTerminal renders one complete frame and writes it to stdout in a
// single os.Stdout.Write call — required by Termux to prevent tearing.
func drawTerminal() {
    w, rows := getTermSize()
    h := (rows - 2) * 2
    if h < 2 {
        h = 2
    }

    // If the prefetch system already loaded a frame into renderData (via
    // popPrefetchFrame in the 'z' key handler), reuse it directly.
    // prefetchFrameReady is set to true by the z handler and cleared here
    // after one use, so subsequent redraws (e.g. help toggle) still re-render.
    var data []float64
    if prefetchFrameReady && renderDataW == w && renderDataH == h {
        data = renderData
        prefetchFrameReady = false
    } else {
        data = renderMandelbrot(w, h, cx, cy, zoom)
    }

    displayData := data
    if histoEQ {
        if mapped := buildHistoMap(data); mapped != nil {
            displayData = mapped
        }
    }

    lut := luts[currentPaletteName]
    colorScale := colorDensity * 0.01
    lutSizeF := float64(lutSize)

    // Each cell: up to 19 bytes BG + 19 bytes FG + 3 bytes blockChar = 41 bytes worst case.
    // Plus sync markers (16), cursor home (3), row resets (5*rows), status bar (200).
    need := w*(h/2)*42 + (h/2)*6 + 512
    termBuf = termBuf[:0]
    if cap(termBuf) < need {
        termBuf = make([]byte, 0, need)
    }
    buf := termBuf

    // ── Precompute 16-Color LUT if active ───────────────────────────────────
    if use16Color && (lut16PalName != currentPaletteName || len(lut16Cache) != lutSize) {
        buildLut16Cache()
    }

    // ── Precompute LUT indices for all pixels ───────────────────────────────
    pixCount := w * h
    if cap(termIdxBuf) < pixCount {
        termIdxBuf = make([]int32, pixCount)
    }
    idxBuf := termIdxBuf[:pixCount]
    for i, v := range displayData {
        if v < 0 {
            idxBuf[i] = -1
        } else {
            idxBuf[i] = int32(valToIdx(v, colorScale, lutSizeF))
        }
    }

    // ── Delta emit with cursor-skip ─────────────────────────────────────────
    // Compare idxBuf against prevIdxBuf. Unchanged cell-pairs are skipped
    // with \033[NC (cursor-forward N cols) — just 4-6 bytes vs 41 per cell.
    // On a zoom cache-hit, ~60-80% of cells change; on pan, all change.
    // prevIdxBuf is swapped at the end — no extra allocation.
    dimChanged := w != prevW || h != prevH
    if dimChanged {
        if cap(prevIdxBuf) < pixCount {
            prevIdxBuf = make([]int32, pixCount)
        }
        prevIdxBuf = prevIdxBuf[:pixCount]
        for i := range prevIdxBuf { prevIdxBuf[i] = -999 }
        prevW, prevH = w, h
    }

    buf = append(buf, "\033[?2026h"...)
    buf = append(buf, "\033[H"...)

    lastBG, lastFG := -2, -2
    rows = h / 2
    for row := 0; row < rows; row++ {
        y := row * 2
        skip := 0
        for x := 0; x < w; x++ {
            idxTop := int(idxBuf[y*w+x])
            idxBot := -1
            if y+1 < h { idxBot = int(idxBuf[(y+1)*w+x]) }
            prevTop := int(prevIdxBuf[y*w+x])
            prevBot := -999
            if y+1 < h { prevBot = int(prevIdxBuf[(y+1)*w+x]) }

            if idxTop == prevTop && idxBot == prevBot {
                skip++
                continue
            }
            if skip > 0 {
                if skip == 1 {
                    buf = append(buf, '\033', '[', 'C')
                } else {
                    buf = fmt.Appendf(buf, "\033[%dC", skip)
                }
                skip = 0
                lastBG, lastFG = -2, -2
            }
            if use16Color {
                dispTop := -1
                if idxTop >= 0 { dispTop = lut16Cache[idxTop] }
                dispBot := -1
                if idxBot >= 0 { dispBot = lut16Cache[idxBot] }
                if dispTop != lastBG {
                    if dispTop < 0 { buf = append(buf, bgBlack...) } else { buf = append(buf, bg16(dispTop)...) }
                    lastBG = dispTop
                }
                if dispBot != lastFG {
                    if dispBot < 0 { buf = append(buf, fgBlack...) } else { buf = append(buf, fg16(dispBot)...) }
                    lastFG = dispBot
                }
            } else {
                if idxTop != lastBG {
                    if idxTop < 0 { buf = append(buf, bgBlack...) } else { buf = append(buf, lut.BG[idxTop]...) }
                    lastBG = idxTop
                }
                if idxBot != lastFG {
                    if idxBot < 0 { buf = append(buf, fgBlack...) } else { buf = append(buf, lut.FG[idxBot]...) }
                    lastFG = idxBot
                }
            }
            buf = append(buf, blockChar...)
        }
        buf = append(buf, resetSeq...)
        buf = append(buf, '\n')
        lastBG, lastFG = -2, -2
    }

    termIdxBuf, prevIdxBuf = prevIdxBuf[:pixCount], idxBuf

    buf = append(buf, "\033[?2026l"...) // End synchronized update

    // ── Status bar ──────────────────────────────────────────────────────────
    zoomExp := zoom.MantExp(nil)
    log10Zoom := float64(zoomExp) * 0.30103
    cxF, _ := cx.Float64()
    cyF, _ := cy.Float64()

    modeStr := "Mandelbrot"
    if juliaMode {
        modeStr = fmt.Sprintf("Julia(%.4f%+.4fi)", juliaR, juliaI)
    }

    flags := ""
    if histoEQ {
        flags += " [EQ]"
    }
    if adaptIter {
        flags += " [AI]"
    }
    if useOpenCL && ocl != nil {
        flags += " [GPU]"
    } else if useOpenCL {
        flags += " [GPU-err]"
    }
    if use16Color {
        flags += " [16C]"
    }
    ready := int(atomic.LoadInt32(&prefetchReady))
    total := int(atomic.LoadInt32(&prefetchTotal))
    if total > 0 {
        flags += fmt.Sprintf(" [Cache:%d/%d]", ready, total)
    }

    // fmt.Sprintf allocates; append directly to avoid the intermediate string.
    buf = fmt.Appendf(buf,
        "%s | Pos:%.2e%+.2ei | Z:10^%.1f | It:%d | %s | D:%.2f%s | h:help",
        modeStr, cxF, cyF, log10Zoom, maxIter, currentPaletteName, colorDensity, flags,
    )

    // ── Help overlay ────────────────────────────────────────────────────────
    if showHelp {
        buf = append(buf, []byte(`
┌─ MOVEMENT ──────────────────────────────────────┐
│ w/a/s/d    Move          z/x    Zoom ×1.5/÷1.5  │
│ Z/X        Zoom ×10/÷10  R      Reset view       │
├─ QUALITY ───────────────────────────────────────┤
│ i/o        Iters ×2/÷2   I/O    Iters ×8/÷8     │
│ A          Toggle auto-iter scaling              │
│ e          Toggle histogram equalization         │
├─ COLOR ─────────────────────────────────────────┤
│ c          Cycle palette  n      Custom hex pal  │
│ r          Random palette R      Reset view      │
│ k/l        Density ×1.2  K/L    Density ×5/÷5   │
│ T          Toggle 16-color mode                 │
├─ MODES ─────────────────────────────────────────┤
│ J          Toggle Julia mode                     │
│ ,/.        Julia real ∓0.01  ;/'  Julia imag ∓  │
├─ NAVIGATION ────────────────────────────────────┤
│ v          Find Minibrot  (q to abort mid-search)│
│ g          I Feel Lucky   (random deep zoom)     │
│ B          Goto position  (paste cx cy [zoom])   │
│ E          Echo position  (print coords to copy) │
│ 1-9        Load bookmark  !-( (S+1-9) Save       │
├─ EXPORT ────────────────────────────────────────┤
│ P          1920×1080 PPM  p      Current-size PNG│
│ M          Zoom-out MP4   F      Zoom-in MP4     │
│ W          Zoom-out live terminal playback        │
│ Space      Pause/resume animation (during M/F)   │
├─ OTHER ─────────────────────────────────────────┤
│ C          Pre-compute N zooms (cache)           │
│ G          Toggle OpenCL GPU render              │
│ h          Toggle help    q      Quit            │
└─────────────────────────────────────────────────┘`)...)
    }

    termBuf = buf
    os.Stdout.Write(buf)
}
