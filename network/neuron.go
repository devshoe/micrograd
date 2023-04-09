package network

import (
	"fmt"
	"math/rand"
	"nn/network/exptree"
)

// Neuron is the fundamental element of a NN
type Neuron struct {
	Label        string
	NumberInputs int
	Weights      []*exptree.Node // of size `NumberInputs`
	Bias         *exptree.Node

	//TODO modularize
	Activation func(in *exptree.Node) (out *exptree.Node)
}

// NewNeuron initializes a neuron with `inputsize` random floats
// if `label` is supplied, it is prefixed to the label of the weights
func NewNeuron(label string, inputsize int) *Neuron {
	weights := []*exptree.Node{}

	for i := 0; i < inputsize; i++ {
		weightLabel := fmt.Sprintf("%s_w%d", label, i)
		weight := exptree.NewNode(weightLabel, 1.0/float64(inputsize))
		weights = append(weights, weight)
	}

	biasLabel := fmt.Sprintf("%s_bias", label)
	bias := exptree.NewNode(biasLabel, rand.Float64())

	return &Neuron{
		Label:        label,
		NumberInputs: inputsize,
		Weights:      weights,
		Bias:         bias,
		Activation:   func(n *exptree.Node) *exptree.Node { return exptree.Tanh(fmt.Sprintf("%s_tanh", label), n) },
	}
}

// Forwards computes sum(xiwi) + b followed by calling the activation function.
// `input` should have len `Neuron.NumberInputs`, or will panic
func (n *Neuron) Forwards(input []*exptree.Node) *exptree.Node {
	if len(input) != n.NumberInputs {
		panic(fmt.Sprintf("mismatch in input dimensions: want %d, got %d", n.NumberInputs, len(input)))
	}

	var (
		neuronProductSumLabel           = fmt.Sprintf("%s_product_sum", n.Label)
		neuronPreActivationOutputLabel  = fmt.Sprintf("%s_biased", n.Label)
		neuronPostActivationOutputLabel = fmt.Sprintf("%s_output", n.Label)
		products                        = []*exptree.Node{}
		productSum                      = exptree.NewNode(neuronProductSumLabel, 0.0)
	)

	for i := range input {
		productLabel := fmt.Sprintf("%sin%dw%d", n.Label, i, i)
		product := exptree.Multiply(productLabel, input[i], n.Weights[i])
		products = append(products, product)
	}

	productSum = exptree.Add(neuronProductSumLabel, products...)

	preActivationOutput := exptree.Add(neuronPreActivationOutputLabel, productSum, n.Bias)
	postActivationOutput := exptree.Tanh(neuronPostActivationOutputLabel, preActivationOutput)

	return postActivationOutput
}

// Parameters returns the weights of this neuron
func (n *Neuron) Parameters() []*exptree.Node {
	return append(n.Weights, n.Bias)
}
func (n *Neuron) String() string {
	return fmt.Sprintf("[Neuron %s | Input Size %d| Weights %v | Bias %v]", n.Label, n.NumberInputs, n.Weights, n.Bias)
}

func (n *Neuron) ToJSONMap() map[string]any {
	data := map[string]any{
		"label":          n.Label,
		"number_inputs":  n.NumberInputs,
		"number_outputs": 1,
		"weights":        n.getWeightsFloat64(),
		"bias":           n.Bias.Data,
	}

	return data
}

func (n *Neuron) getWeightsFloat64() []float64 {
	out := []float64{}
	for i := range n.Weights {
		out = append(out, n.Weights[i].Data)
	}

	return out
}
