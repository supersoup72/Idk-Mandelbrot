// cflags_arm64.go — CGO compiler flags for ARM64 (Samsung S20 FE / Cortex-A77).
//go:build arm64

package main

// #cgo CFLAGS: -O3 -mcpu=cortex-a77 -march=armv8.2-a+fp16+dotprod -ffast-math -funroll-loops -ftree-vectorize -fomit-frame-pointer -fno-plt -fstrict-aliasing -falign-functions=64 -falign-loops=32 -fno-stack-protector -fprefetch-loop-arrays
import "C"
