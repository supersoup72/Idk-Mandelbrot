#!/bin/bash
# Build script for Linux x86-64 (Intel i7-12700H / Alder Lake)
# Requires: go, gcc
#
# Ubuntu/Debian:  sudo apt install golang-go gcc
# Arch:           sudo pacman -S go gcc
# Fedora:         sudo dnf install golang gcc

set -e
echo "=== Mandelbrot builder for Linux x86-64 ==="
echo "Go:  $(go version)"
echo "GCC: $(gcc --version | head -1)"
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo ""
echo "Building (AVX2 + FMA path active)..."
go build -o mandelbrot .
echo "Done! Binary: ./mandelbrot"
echo ""
echo "Benchmark (e20):"
echo "  ./mandelbrot --no-interactive --cx=-0.7269 --cy=0.1889 --zoom=1e20 --repeat=5"
echo ""
echo "Benchmark (your e32 deep zoom):"
echo "  ./mandelbrot --no-interactive --cx=-1.47872828219977616902121434692406329094 --cy=-0.002575590224323645168594890744282975494392 --zoom=127396267028484819415870936250763.276032 --iter=50000 --repeat=3"
echo ""
echo "Render a 1080p PNG:"
echo "  ./mandelbrot --no-interactive --cx=-0.7269 --cy=0.1889 --zoom=1e20 --1080p --pic out.png"
