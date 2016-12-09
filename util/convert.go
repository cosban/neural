package util

import (
	"github.com/gonum/matrix/mat64"
	mnist "github.com/petar/GoMNIST"
)

// FromImage constructs a vector from an image
func FromImage(image mnist.RawImage) *mat64.Dense {
	m := mat64.NewDense(len(image), 1, nil)
	for i, v := range image {
		m.Set(i, 0, float64(v)/255.0)
	}
	return m
}

func FromLabel(label mnist.Label) *mat64.Dense {
	m := mat64.NewDense(10, 1, nil)
	m.Set(int(label), 0, 1.0)
	return m
}
