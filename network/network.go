package network

import (
	"log"

	"github.com/cosban/neural/matrix"
	"github.com/cosban/neural/set"
	"github.com/gonum/matrix/mat64"
)

// Network is made up from layers of nodes/cells known as sigmoids
type Network struct {
	// Layers is the number of layers in the network
	layers int
	// Sizes is slice containing the number of nodes in each respective layer
	sizes []int
	// Biases represents the bias for every node
	biases []*mat64.Dense
	// Weights is a slice of matrix.Matrixes which represent the weights on each input
	// coming to a node
	weights []*mat64.Dense
}

// New creates and initializes new network
func New(sizes []int) *Network {
	layers := len(sizes)
	return &Network{
		layers:  layers,
		sizes:   sizes,
		biases:  buildBiases(sizes),
		weights: buildWeights(sizes),
	}
}

func (n *Network) feedforward(a *mat64.Dense) *mat64.Dense {
	for i := range n.biases {
		w := n.weights[i]
		b := n.biases[i]
		d := mat64.NewDense(0, 0, nil)
		d.Mul(w, a)
		d.Add(b, d)
		d.Apply(matrix.Sigmoid, d)
		a = d
	}
	return a
}

// SGD uses stochastic gradient descent to train out neural network
func (n *Network) SGD(training, testing *set.Set, epochs, mBatchSize int, eta float64) {
	//num := training.Count()
	for j := 0; j < epochs; j++ {
		training = training.Shuffle()
		miniBatches := buildBatches(training, 1, 1)
		for i, batch := range miniBatches {
			n.update(batch, eta)
			if i > 0 && i%1000 == 0 {
				log.Printf("batch %d", i)
			}
		}
		log.Printf("Epoch %d: %d / %d\n", j, n.Evaluate(testing), testing.Count())
	}
}

func (n *Network) update(batch *set.Set, eta float64) {
	nablaB := matrix.Zeros(n.biases)
	nablaW := matrix.Zeros(n.weights)
	for {
		image, label, present := batch.Next()
		if !present {
			break
		}
		dNablaB, dNablaW := n.backprop(image, label)
		nablaB = matrix.Delta(nablaB, dNablaB)
		nablaW = matrix.Delta(nablaW, dNablaW)
	}
	n.weights = matrix.Adjust(n.weights, nablaW, float64(batch.Count()), eta)
	n.biases = matrix.Adjust(n.biases, nablaB, float64(batch.Count()), eta)
}

func (n *Network) backprop(image, y *mat64.Dense) ([]*mat64.Dense, []*mat64.Dense) {
	nablaB := matrix.Zeros(n.biases)
	nablaW := matrix.Zeros(n.weights)
	// feedforward.... notice how similar this is to the previous feed forward function
	activation := image
	activations := []*mat64.Dense{activation}
	var zs []*mat64.Dense
	for i := range n.biases {
		b := n.biases[i]
		w := n.weights[i]
		z := mat64.NewDense(0, 0, nil)
		z.Mul(w, activation)
		z.Add(b, z)
		zs = append(zs, z)
		activation = mat64.NewDense(0, 0, nil)
		activation.Apply(matrix.Sigmoid, z)
		activations = append(activations, activation)
	}

	//backward pass
	delta := mat64.NewDense(0, 0, nil)
	delta.Sub(activations[len(activations)-1], y)
	sp := mat64.NewDense(0, 0, nil)
	sp.Apply(matrix.SigmoidPrime, zs[len(zs)-1])
	delta.MulElem(delta, sp)
	nablaB[len(nablaB)-1] = delta
	nablaW[len(nablaW)-1].Mul(delta, activations[len(activations)-2].T())

	for l := 2; l < n.layers; l++ {
		z := zs[len(zs)-l]
		sp = mat64.NewDense(0, 0, nil)
		sp.Apply(matrix.SigmoidPrime, z)

		tDelta := mat64.NewDense(0, 0, nil)
		tDelta.Mul(n.weights[len(n.weights)-l+1].T(), delta)
		tDelta.MulElem(tDelta, sp)
		delta = tDelta

		nablaB[len(nablaB)-l] = delta

		t := mat64.NewDense(0, 0, nil)
		t.Mul(delta, activations[len(activations)-l-1].T())
		nablaW[len(nablaW)-l] = t
	}
	return nablaB, nablaW
}

func (n *Network) Evaluate(testing *set.Set) int {
	sum := 0
	for {
		i, l, p := testing.Next()
		if !p {
			break
		}
		m := n.feedforward(i)
		rr, rc := matrix.Max(m)
		er, ec := matrix.Max(l)

		if rr == er && rc == ec {
			sum++
		}
	}
	return sum
}
