package main

// Neuron is the fundamental element of a NN
type Neuron struct {
	NumberInputs int
	Weights      []*Node // of size `NumberInputs`
	Bias         float64
	Activation   func(in *Node) (out *Node)
}
