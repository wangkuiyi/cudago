package cublas

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/cudago/cuda"
)

func TestCreateDestroy(t *testing.T) {
	h := Create()
	defer Destroy(h)
}

func TestSetGetMatrixF32(t *testing.T) {
	h := Create()
	defer Destroy(h)

	M := 5
	N := 2

	gpu := cuda.MallocF32(M * N)
	defer cuda.Free(gpu)

	a := make([]float32, M*N)
	b := make([]float32, M*N)

	for i := range a {
		a[i] = float32(i)
	}

	SetMatrixF32(M, N, a, gpu)
	GetMatrixF32(M, N, gpu, b)

	assert.Equal(t, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, b)
}
