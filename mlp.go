package main

import "fmt"

type MultiLayerPerceptron struct {
	Label         string
	Layers        []*Layer
	NumberInputs  int
	NumberOutputs []int
}

func NewMultiLayerPerceptron(numIn int, numOut []int, label ...string) *MultiLayerPerceptron {
	l := ""
	if len(label) > 0 {
		l = label[0]
	}

	allLayers := append([]int{numIn}, numOut...)
	layers := []*Layer{}
	for i := 0; i < len(numOut); i++ {
		layerLabel := fmt.Sprintf("mlp_%s_layer_%d", l, i)
		layers = append(layers, NewLayer(allLayers[i], allLayers[i+1], layerLabel))
	}
	return &MultiLayerPerceptron{
		Label:         l,
		Layers:        layers,
		NumberInputs:  numIn,
		NumberOutputs: numOut,
	}
}

func (mlp MultiLayerPerceptron) Output(in []*Node) (out []*Node) {
	buf := in
	for i := range mlp.Layers {
		buf = mlp.Layers[i].Forwards(buf)
	}
	return buf
}
