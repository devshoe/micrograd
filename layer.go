package main

import "fmt"

// Layer is a set of neurons
type Layer struct {
	Label         string
	Neurons       []*Neuron
	NumberInputs  int
	NumberOutputs int
}

// NewLayer creates a new layer or set of neurons
// `numInputs` specifies the number of inputs for each neuron i.e also the number of weights
// `numOutputs` specifies the number of neurons this layer has
func NewLayer(label string, numInputs, numOutputs int) *Layer {

	neurons := []*Neuron{}

	for i := 0; i < numOutputs; i++ {
		neuronLabel := fmt.Sprintf("%s_n%d", label, i)
		neurons = append(neurons, NewNeuron(neuronLabel, numInputs))
	}

	return &Layer{
		Label:         label,
		Neurons:       neurons,
		NumberInputs:  numInputs,
		NumberOutputs: numOutputs,
	}
}

// Forwards computes the output of the layer.
func (l *Layer) Forwards(in []*Node) (output []*Node) {
	for _, neuron := range l.Neurons {
		output = append(output, neuron.Forwards(in))
	}
	return
}

// Parameters returns the weights of all nodes in this layer as a flattened array
func (l *Layer) Parameters() []*Node {
	n := []*Node{}
	for i := range l.Neurons {
		n = append(n, l.Neurons[i].Weights...)
	}
	return n
}

func (l *Layer) ToJSONMap() map[string]any {
	data := map[string]any{
		"name":           l.Label,
		"number_inputs":  l.NumberInputs,
		"number_outputs": l.NumberOutputs,
		"neurons":        []map[string]any{},
	}
	neurons := []map[string]any{}
	for i := range l.Neurons {
		neurons = append(neurons, l.Neurons[i].ToJSONMap())
	}

	data["neurons"] = neurons
	return data
}
