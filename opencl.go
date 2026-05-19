// opencl.go — Stub for cross-compilation on non-OpenCL hosts.
// The real opencl_orig.go requires CL/cl.h; this stub satisfies the linker
// so we can cross-compile. On the device, GPU support can be re-enabled.
package main

import (
	"fmt"
	"math/big"
	"sync"
)

var useOpenCL bool
var ocl       *oclState
var oclInitOnce sync.Once
var oclInitErr  error

type oclState struct{}

func initOpenCL() error {
	return fmt.Errorf("OpenCL not compiled in this build")
}

func renderOpenCL(w, h int, rcx, rcy, rzoom *big.Float) []float64 { return nil }
func shutdownOpenCL() {}
func oclDeviceName() string { return "none" }
