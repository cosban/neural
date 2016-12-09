package network

import (
	"math/rand"
	"time"

	"github.com/cosban/neural/set"
	"github.com/gonum/matrix/mat64"
)

var r = rand.New(rand.NewSource(time.Now().UnixNano()))

func buildBiases(sizes []int) []*mat64.Dense {
	var biases []*mat64.Dense
	for _, layer := range sizes[1:] {
		use := mat64.NewDense(layer, 1, nil)
		for i := 0; i < layer; i++ {
			use.Set(i, 0, r.NormFloat64())
		}
		biases = append(biases, use)
	}
	return biases
}

func buildWeights(sizes []int) []*mat64.Dense {
	var weights []*mat64.Dense
	for i := range sizes[:len(sizes)-1] {
		use := mat64.NewDense(sizes[i+1], sizes[i], nil)
		rows, cols := use.Caps()
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				use.Set(row, col, r.NormFloat64())
			}
		}
		weights = append(weights, use)
	}
	return weights
}

func buildBatches(training *set.Set, n, batchSize int) []*set.Set {
	var output []*set.Set
	for i := 0; i < n; i++ {
		k := i * batchSize
		lower := k % training.Count()
		upper := (k + batchSize) % training.Count()
		var images []*mat64.Dense
		var labels []*mat64.Dense
		// check for wrapping
		if lower < upper {
			images = training.Images[lower:upper]
			labels = training.Labels[lower:upper]
		} else {
			images = training.Images[lower:]
			labels = training.Labels[lower:]
			images = append(images, training.Images[:upper]...)
			labels = append(labels, training.Labels[:upper]...)
		}

		batch := set.Pack(images, labels)
		output = append(output, batch)
	}
	return output
}
