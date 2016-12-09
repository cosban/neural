package util

import "math"

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*z))
}

func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1.0 - Sigmoid(z))
}
