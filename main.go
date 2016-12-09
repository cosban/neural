package main

import (
	"fmt"
	"log"

	"github.com/cosban/neural/network"
	"github.com/cosban/neural/set"

	mnist "github.com/petar/GoMNIST"
)

const (
	PIXELS = 784
	DIGITS = 10
)

func main() {

	net := network.New([]int{PIXELS, 100, DIGITS})
	fmt.Println("Loading MNIST DATA...")
	rawTraining, rawTesting, err := mnist.Load("./data")
	if err != nil {
		panic(err)
	}
	log.Println("Done... Converting Images..")
	training := set.Convert(rawTraining)
	testing := set.Convert(rawTesting)
	log.Println("Done... Evaluating Data..")
	net.SGD(training, testing, 30, 10, 3.0)
}
