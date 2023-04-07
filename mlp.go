package main

import (
	"fmt"
)

// MultiLayerPerceptron is the simplest kind of neural network
type MultiLayerPerceptron struct {
	Label         string
	Layers        []*Layer
	NumberInputs  int
	NumberOutputs []int
}

// NewMultiLayerPerceptron creates a new MLP. the num inputs is the number of inputs
// len of `numOut` specifies the number of layers this mlp has
// the value of index in `numOut` specifies the number of neurons the corresponding layer will have
func NewMultiLayerPerceptron(label string, numIn int, numOut []int) *MultiLayerPerceptron {
	allLayers := append([]int{numIn}, numOut...)
	layers := []*Layer{}
	for i := 0; i < len(numOut); i++ {
		layerLabel := fmt.Sprintf("l%d", i)
		layers = append(layers, NewLayer(layerLabel, allLayers[i], allLayers[i+1]))
	}
	return &MultiLayerPerceptron{
		Label:         label,
		Layers:        layers,
		NumberInputs:  numIn,
		NumberOutputs: numOut,
	}
}

// Forwards returns the final output of this neural network
func (mlp *MultiLayerPerceptron) Forwards(in []*Node) (out []*Node) {
	buf := in
	for i := range mlp.Layers {
		buf = mlp.Layers[i].Forwards(buf)
	}
	return buf
}

// Parameters returns the weights of all nodes in all layers as a flattened array
func (mlp *MultiLayerPerceptron) Parameters() []*Node {
	n := []*Node{}
	for i := range mlp.Layers {
		n = append(n, mlp.Layers[i].Parameters()...)
	}
	return n
}

// Train runs the training, performing backpropagation and gradient descent.
func (mlp *MultiLayerPerceptron) Train(inputs [][]float64, outputs [][]float64) error {

	if len(inputs) <= 0 {
		return fmt.Errorf("train: inputs must contain something")
	} else if len(outputs) != len(inputs) {
		return fmt.Errorf("train: mismatch b/w input and output, want %d got %d", len(inputs), len(outputs))
	} else if len(inputs[0]) != mlp.NumberInputs {
		return fmt.Errorf("train: mismatch b/w train set dimensions & mlp dimensions, want %d got %d", (mlp.NumberInputs), len(inputs[0]))
	} else if len(outputs[0]) != mlp.NumberOutputs[len(mlp.NumberOutputs)-1] {
		return fmt.Errorf("train: mismatch b/w train set dimensions & mlp dimensions, want %d got %d", (mlp.NumberOutputs[len(mlp.NumberOutputs)-1]), len(outputs[0]))
	}

	return nil
}

func (mlp *MultiLayerPerceptron) MeanSquaredLoss(inputs [][]float64, outputs [][]float64) *Node {
	trainIn, trainOut := toNodes(inputs, outputs)

	preds := [][]*Node{}
	losses := []*Node{}
	pow := NewNode("loss_pow", 2)
	for rowIndex, record := range trainIn {
		pred := mlp.Forwards(record)
		preds = append(preds, pred)

		for colIndex, elem := range trainOut[rowIndex] {
			diffLabel, powLabel := fmt.Sprintf("diff%d_%d", rowIndex, colIndex), fmt.Sprintf("local_loss%d_%d", rowIndex, colIndex)
			losses = append(losses, Power(powLabel, pow, Sub(diffLabel, elem, pred[colIndex])))
		}

	}

	return Add("loss", losses...)
}

func toNodes(inputs [][]float64, outputs [][]float64) ([][]*Node, [][]*Node) {
	inNodes, outNodes := [][]*Node{}, [][]*Node{}
	for i, in := range inputs {
		mlpIn := []*Node{}
		for idx, data := range in {
			mlpIn = append(mlpIn, NewNode(fmt.Sprintf("r%din%d", i, idx), data))
		}

		inNodes = append(inNodes, mlpIn)

	}

	for j, out := range outputs {
		mlpOut := []*Node{}
		for idx, data := range out {
			mlpOut = append(mlpOut, NewNode(fmt.Sprintf("r%dout%d", j, idx), data))
		}

		outNodes = append(outNodes, mlpOut)
	}
	return inNodes, outNodes
}

func Sum(vals ...float64) float64 {
	v := 0.0
	for i := range vals {
		v += vals[i]
	}
	return v
}
