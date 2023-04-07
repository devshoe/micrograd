package main

import (
	"fmt"
	"math/rand"
)

// Neuron is the fundamental element of a NN
type Neuron struct {
	Label        string
	NumberInputs int
	Weights      []*Node // of size `NumberInputs`
	Bias         *Node

	//TODO modularize
	Activation func(in *Node) (out *Node)
}

// NewNeuron initializes a neuron with `inputsize` random floats
// if `label` is supplied, it is prefixed to the label of the weights
func NewNeuron(label string, inputsize int) *Neuron {
	weights := []*Node{}

	for i := 0; i < inputsize; i++ {
		weightLabel := fmt.Sprintf("%s_w%d", label, i)
		weight := NewNode(weightLabel, 1.0/float64(inputsize))
		weights = append(weights, weight)
	}

	biasLabel := fmt.Sprintf("%s_bias", label)
	bias := NewNode(biasLabel, rand.Float64())

	return &Neuron{
		Label:        label,
		NumberInputs: inputsize,
		Weights:      weights,
		Bias:         bias,
		Activation:   func(n *Node) *Node { return Tanh(fmt.Sprintf("%s_tanh", label), n) },
	}
}

// Forwards computes sum(xiwi) + b followed by calling the activation function.
// `input` should have len `Neuron.NumberInputs`, or will panic
func (n *Neuron) Forwards(input []*Node) *Node {
	if len(input) != n.NumberInputs {
		panic(fmt.Sprintf("mismatch in input dimensions: want %d, got %d", n.NumberInputs, len(input)))
	}

	var (
		neuronProductSumLabel           = fmt.Sprintf("%s_product_sum", n.Label)
		neuronPreActivationOutputLabel  = fmt.Sprintf("%s_biased", n.Label)
		neuronPostActivationOutputLabel = fmt.Sprintf("%s_output", n.Label)
		products                        = []*Node{}
		productSum                      = NewNode(neuronProductSumLabel, 0.0)
	)

	for i := range input {
		productLabel := fmt.Sprintf("%sin%dw%d", n.Label, i, i)
		product := Multiply(productLabel, input[i], n.Weights[i])
		products = append(products, product)
	}

	productSum = Add(neuronProductSumLabel, products...)

	preActivationOutput := Add(neuronPreActivationOutputLabel, productSum, n.Bias)
	postActivationOutput := Tanh(neuronPostActivationOutputLabel, preActivationOutput)

	return postActivationOutput
}

// Parameters returns the weights of this neuron
func (n *Neuron) Parameters() []*Node {
	return n.Weights
}
func (n *Neuron) String() string {
	return fmt.Sprintf("[Neuron %s | Input Size %d| Weights %v | Bias %v]", n.Label, n.NumberInputs, n.Weights, n.Bias)
}
