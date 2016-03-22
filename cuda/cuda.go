package cuda

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib -lcudart
// #include <cuda_runtime.h>
import "C"
import "unsafe"

func Malloc(n int) unsafe.Pointer {
	var p unsafe.Pointer
	r := C.cudaMalloc(&p, C.size_t(n))
	if r != C.cudaSuccess {
		panic("Failed allocate CUDA memory")
	}
	return p
}

func Free(p unsafe.Pointer) {
	C.cudaFree(p)
}

func MallocF32(n int) unsafe.Pointer {
	return Malloc(n * 4)
}
