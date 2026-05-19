#!/data/data/com.termux/files/usr/bin/bash
# Build script for Termux on Samsung S20 FE (ARM64)
# Run this inside Termux after extracting the source zip.
#
# One-time setup (if you haven't already):
#   pkg install golang clang
#
# Then just run:
#   bash build_termux.sh

set -e

echo "=== Mandelbrot builder for Termux ==="
echo ""

# Check dependencies
if ! command -v go &>/dev/null; then
    echo "Go not found. Installing..."
    pkg install golang -y
fi

if ! command -v clang &>/dev/null && ! command -v gcc &>/dev/null; then
    echo "C compiler not found. Installing..."
    pkg install clang -y
fi

echo "Go:  $(go version)"
echo "CC:  $(command -v clang || command -v gcc)"
echo "CPU: $(uname -m)"
echo ""

# Build — CGO picks up clang/gcc automatically in Termux
echo "Building..."
go build -o mandelbrot .

echo ""
echo "Done! Binary: ./mandelbrot"
echo ""
echo "Quick test:"
echo "  ./mandelbrot --no-interactive --cx=-0.7269 --cy=0.1889 --zoom=1e20 --repeat=5"
echo ""
echo "Interactive mode:"
echo "  ./mandelbrot"
