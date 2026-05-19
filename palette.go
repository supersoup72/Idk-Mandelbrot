// palette.go — Color interpolation, LUT construction, histogram equalization,
// and custom hex palette parsing.
//
// Smoothness improvements over the original:
//
//  1. Linear-light interpolation: palette stops are gamma-decoded to linear
//     light before blending, then re-encoded.  Linear blending avoids the
//     muddy dark midpoints that RGB-space blending produces.
//
//  2. Cubic smoothstep easing: each segment between two palette stops uses
//     smoothstep(t) = 3t²-2t³ instead of linear t.  This gives a smooth
//     zero-derivative join at every stop — no colour "corners".
//
//  3. Richer palette stop colors: the built-in palettes have been tuned so
//     every stop is a saturated, visually distinct hue, maximising the colour
//     range across each cycle.
//
//  4. Higher cycle density default: colorDensity default raised to 2.0 so
//     you see more colour variation per zoom level without needing to adjust.
//
//  5. valToIdx uses a sine-smoothed fractional position so the LUT wrap point
//     (where t cycles from 1 back to 0) is a smooth cosine cross-fade rather
//     than a hard step.
//
package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// ─────────────────────────────────────────────
//  Gamma helpers
// ─────────────────────────────────────────────

// toLinear converts an sRGB uint8 channel to linear light [0,1].
// Uses the exact IEC 61966-2-1 piecewise function.
func toLinear(c uint8) float64 {
	s := float64(c) / 255.0
	if s <= 0.04045 {
		return s / 12.92
	}
	return math.Pow((s+0.055)/1.055, 2.4)
}

// toSRGB converts a linear-light [0,1] value back to sRGB uint8.
func toSRGB(lin float64) uint8 {
	if lin <= 0 {
		return 0
	}
	if lin >= 1 {
		return 255
	}
	var s float64
	if lin <= 0.0031308 {
		s = lin * 12.92
	} else {
		s = 1.055*math.Pow(lin, 1.0/2.4) - 0.055
	}
	v := int(s*255.0 + 0.5)
	if v > 255 {
		v = 255
	}
	return uint8(v)
}

// ─────────────────────────────────────────────
//  Smoothstep
// ─────────────────────────────────────────────

// smoothstep maps t ∈ [0,1] → [0,1] with zero first-derivative at both ends.
// Eliminates the "corner" visible at palette stop boundaries with linear lerp.
func smoothstep(t float64) float64 {
	return t * t * (3.0 - 2.0*t)
}

// ─────────────────────────────────────────────
//  Interpolation
// ─────────────────────────────────────────────

// interpolateColor linearly interpolates across a palette slice using
// linear-light blending and smoothstep easing between stops.
// t ∈ [0, 1] maps across the full palette.
func interpolateColor(palette []Color, t float64) Color {
	n := len(palette)
	if n == 0 {
		return Color{}
	}
	if n == 1 {
		return palette[0]
	}

	scaled := t * float64(n-1)
	i := int(scaled)
	frac := scaled - float64(i)

	if i >= n-1 {
		return palette[n-1]
	}
	if i < 0 {
		return palette[0]
	}

	// Apply smoothstep so transitions ease in/out at every stop.
	frac = smoothstep(frac)

	a, b := palette[i], palette[i+1]

	// Decode to linear light, blend, re-encode to sRGB.
	lr := toLinear(a.R)*(1-frac) + toLinear(b.R)*frac
	lg := toLinear(a.G)*(1-frac) + toLinear(b.G)*frac
	lb := toLinear(a.B)*(1-frac) + toLinear(b.B)*frac

	return Color{toSRGB(lr), toSRGB(lg), toSRGB(lb)}
}

// ─────────────────────────────────────────────
//  LUT construction
// ─────────────────────────────────────────────

func buildLUT(name string, pal []Color) {
	lut := &PaletteLUT{
		Colors: make([]Color, lutSize),
		FG:     make([][]byte, lutSize),
		BG:     make([][]byte, lutSize),
	}
	for i := 0; i < lutSize; i++ {
		c := interpolateColor(pal, float64(i)/float64(lutSize-1))
		lut.Colors[i] = c
		lut.FG[i] = []byte(fmt.Sprintf("\033[38;2;%d;%d;%dm", c.R, c.G, c.B))
		lut.BG[i] = []byte(fmt.Sprintf("\033[48;2;%d;%d;%dm", c.R, c.G, c.B))
	}
	if luts == nil {
		luts = make(map[string]*PaletteLUT)
	}
	luts[name] = lut
}

func initLUTs() {
	for name, pal := range palettes {
		buildLUT(name, pal)
	}
}

// ─────────────────────────────────────────────
//  Color index helper
// ─────────────────────────────────────────────

// valToIdx converts a smooth iteration value to a LUT index.
//
// The key smoothness improvement here: instead of a hard modulo wrap
// (which creates a sharp colour jump where t crosses an integer), we use
// a cosine-smoothed cycle:
//
//	t_cycle = 0.5 - 0.5*cos(2π * frac(t * colorScale))
//
// This maps the [0,1) fractional part through a cosine so the LUT is
// sampled with a smooth S-curve, and the wrap point at frac=0/1 is a
// smooth zero-derivative join rather than a discontinuity.
func valToIdx(val, colorScale, lutSizeF float64) int {
	t := val * colorScale
	frac := t - math.Floor(t)
	// Cosine remap: smooth the cycle boundary.
	smooth := 0.5 - 0.5*math.Cos(2.0*math.Pi*frac)
	idx := int(smooth * lutSizeF)
	if idx >= lutSize {
		idx = lutSize - 1
	}
	if idx < 0 {
		idx = 0
	}
	return idx
}

// ─────────────────────────────────────────────
//  Histogram equalization
// ─────────────────────────────────────────────

// histoEscaped and histoResult are persistent scratch buffers for buildHistoMap.
// Reused every frame when histogram equalization is enabled to avoid GC pressure.
var histoEscaped []float64
var histoResult  []float64

func buildHistoMap(data []float64) []float64 {
	// Reuse persistent slice, growing only when needed.
	if cap(histoEscaped) < len(data) {
		histoEscaped = make([]float64, 0, len(data))
	}
	escaped := histoEscaped[:0]
	for _, v := range data {
		if v >= 0 {
			escaped = append(escaped, v)
		}
	}
	histoEscaped = escaped
	if len(escaped) == 0 {
		return nil
	}
	sort.Float64s(escaped)
	total := float64(len(escaped))

	if cap(histoResult) < len(data) {
		histoResult = make([]float64, len(data))
	}
	result := histoResult[:len(data)]
	colorScale := colorDensity * 0.01
	inv := 1.0 / colorScale

	for i, v := range data {
		if v < 0 {
			result[i] = -1
			continue
		}
		rank := sort.SearchFloat64s(escaped, v)
		result[i] = (float64(rank) / total) * inv
	}
	return result
}

// ─────────────────────────────────────────────
//  Custom hex palette
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  Random palette generator
// ─────────────────────────────────────────────

// generateRandomPalette generates n completely random hex colours, installs them
// as the active "Random" palette, and returns the hex strings for display.
// The returned slice can be joined with "," to form a ready-to-paste string
// for the custom hex palette prompt ('n' key).
func generateRandomPalette(n int, seed int64) []string {
	// LCG — fast, no stdlib/rand import needed.
	rng := uint64(seed)
	nextByte := func() uint8 {
		rng = rng*6364136223846793005 + 1442695040888963407
		return uint8(rng >> 56)
	}

	stops := make([]Color, n)
	for i := range stops {
		stops[i] = Color{R: nextByte(), G: nextByte(), B: nextByte()}
	}

	palettes["Random"] = stops
	buildLUT("Random", stops)
	currentPaletteName = "Random"

	// Ensure "Random" is in paletteKeys.
	found := false
	for _, k := range paletteKeys {
		if k == "Random" {
			found = true
			break
		}
	}
	if !found {
		paletteKeys = append(paletteKeys, "Random")
	}

	hexColors := make([]string, n)
	for i, c := range stops {
		hexColors[i] = fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
	}
	return hexColors
}

func parseHexPalette(input string) {
	parts := strings.Split(input, ",")
	var parsed []Color
	for _, p := range parts {
		p = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(p), "#"))
		if len(p) != 6 {
			continue
		}
		rv, e1 := strconv.ParseUint(p[0:2], 16, 8)
		gv, e2 := strconv.ParseUint(p[2:4], 16, 8)
		bv, e3 := strconv.ParseUint(p[4:6], 16, 8)
		if e1 == nil && e2 == nil && e3 == nil {
			parsed = append(parsed, Color{uint8(rv), uint8(gv), uint8(bv)})
		}
	}
	if len(parsed) < 2 {
		return
	}
	palettes["Custom"] = parsed
	buildLUT("Custom", parsed)
	currentPaletteName = "Custom"
	for _, k := range paletteKeys {
		if k == "Custom" {
			return
		}
	}
	paletteKeys = append(paletteKeys, "Custom")
}
