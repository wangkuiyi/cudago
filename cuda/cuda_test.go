package cuda

import "testing"

func TestMallocFree(t *testing.T) {
	p := Malloc(100)
	defer Free(p)
}

func TestMallocFreeF32(t *testing.T) {
	p := MallocF32(100)
	defer Free(p)
}
