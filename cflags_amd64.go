// cflags_amd64.go — CGO compiler flags for x86-64 (Intel i7-12700H / Alder Lake).
//go:build amd64

package main

// #cgo CFLAGS: -O3 -march=alderlake -mavx2 -mfma -ffast-math -funroll-loops -ftree-vectorize -fomit-frame-pointer -fno-stack-protector -fstrict-aliasing -falign-functions=64 -falign-loops=32
import "C"
