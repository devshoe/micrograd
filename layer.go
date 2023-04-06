package main

import "fmt"

// Layer is a set of neurons
type Layer struct {
	Label         string
	Neurons       []*Neuron
	NumberInputs  int
	NumberOutputs int
}

func NewLayer(numInputs, numOutputs int, label ...string) *Layer {
	l := ""
	if len(label) > 0 {
		l = label[0]
	}

	neurons := []*Neuron{}

	for i := 0; i < numOutputs; i++ {
		neuronLabel := fmt.Sprintf("layer_%s_neuron_%d", l, i)
		neurons = append(neurons, NewNeuron(numInputs, neuronLabel))
	}

	return &Layer{
		Label:         l,
		Neurons:       neurons,
		NumberInputs:  numInputs,
		NumberOutputs: numOutputs,
	}
}

func (l *Layer) Forwards(in []*Node) (output []*Node) {
	for _, neuron := range l.Neurons {
		output = append(output, neuron.Forwards(in))
	}
	return
}
