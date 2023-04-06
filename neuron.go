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
func NewNeuron(inputsize int, label ...string) *Neuron {
	l := ""
	if len(label) > 0 {
		l = label[0]
	}

	weights := []*Node{}

	for i := 0; i < inputsize; i++ {
		weightLabel := fmt.Sprintf("%s_weight-%d", l, i)
		weight := NewNode(1.0/float64(inputsize), weightLabel)
		weights = append(weights, weight)
	}

	biasLabel := fmt.Sprintf("%s_bias", l)
	bias := NewNode(rand.Float64(), biasLabel)

	return &Neuron{
		Label:        l,
		NumberInputs: inputsize,
		Weights:      weights,
		Bias:         bias,
		Activation:   func(n *Node) *Node { return n.Tanh(fmt.Sprintf("%s_tanh", l)) },
	}
}

// Forwards computes sum(xiwi) + b followed by calling the activation function.
// `input` should have len `Neuron.NumberInputs`
func (n *Neuron) Forwards(input []*Node) *Node {
	if len(input) != n.NumberInputs {
		panic(fmt.Sprintf("mismatch in input dimensions: want %d, got %d", n.NumberInputs, len(input)))
	}

	var (
		neuronProductSumLabel           = fmt.Sprintf("neuron_%s_product_sum", n.Label)
		neuronPreActivationOutputLabel  = fmt.Sprintf("neuron_%s_pre_activation_output", n.Label)
		neuronPostActivationOutputLabel = fmt.Sprintf("neuron_%s_post_activation_output", n.Label)
		products                        = []*Node{}
		productSum                      = NewNode(0.0, neuronProductSumLabel)
	)

	for i := range input {
		productLabel := fmt.Sprintf("%sin%dw%d", n.Label, i, i)
		product := input[i].Multiply(n.Weights[i], productLabel)
		products = append(products, product)
	}

	if len(products) == 1 {
		productSum = products[0].Clone(neuronProductSumLabel)
	} else {

		sum := 0.0
		for i := 0; i < len(products); i++ {
			sum += products[i].Data
		}
		productSum.Data = sum
		productSum.ProducedByChildren = products
		productSum.ProducedByOperation = OperationAddition
		productSum.GradientUpdater = func() {
			for i := range products {
				products[i].Gradient = 1.0 * productSum.Gradient
			}
		}
	}

	preActivationOutput := productSum.Add(n.Bias, neuronPreActivationOutputLabel)
	postActivationOutput := preActivationOutput.Tanh(neuronPostActivationOutputLabel)

	return postActivationOutput
}

func (n *Neuron) String() string {
	return fmt.Sprintf("[Neuron %s | Input Size %d| Weights %v | Bias %v]", n.Label, n.NumberInputs, n.Weights, n.Bias)
}
