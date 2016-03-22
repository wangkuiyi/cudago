package cublas

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib -lcublas
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
import "C"
import "unsafe"

type Handle struct {
	handle C.cublasHandle_t
}

func Create() Handle {
	var h Handle
	if C.cublasCreate(&h.handle) != C.CUBLAS_STATUS_SUCCESS {
		panic("CUBLAS initialization failed")
	}
	return h
}

func Destroy(h Handle) {
	C.cublasDestroy(h.handle)
}

// SetMatrixF32 simplifies the interface of cudaSetMatrix by removing
// the capability of copying part of a matrix.
func SetMatrixF32(rows, cols int, cpu []float32, gpu unsafe.Pointer) {
	r := C.cublasSetMatrix(C.int(rows), C.int(cols), C.int(4),
		unsafe.Pointer(&(cpu[0])), C.int(rows), gpu, C.int(rows))
	if r != C.CUBLAS_STATUS_SUCCESS {
		panic("CUBLAS download failed")
	}
}

// GetMatrixF32 simplifies the interface of cudaGetMatrix by removing
// the capability of copying part of a matrix.
func GetMatrixF32(rows, cols int, gpu unsafe.Pointer, cpu []float32) {
	r := C.cublasGetMatrix(C.int(rows), C.int(cols), C.int(4),
		gpu, C.int(rows), unsafe.Pointer(&(cpu[0])), C.int(rows))
	if r != C.CUBLAS_STATUS_SUCCESS {
		panic("CUBLAS upload failed")
	}
}
