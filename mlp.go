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
// the value of index of the previous layer in `numOut` specifies the number of inputs each neuron will have
// Basically, the way an MLP works is that all outputs of previous layer are feed to each neuron in the successive layer.
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

func (mlp *MultiLayerPerceptron) ToJSONMap() map[string]any {
	data := map[string]any{
		"name":             mlp.Label,
		"layer_dimensions": append([]int{mlp.NumberInputs}, mlp.NumberOutputs...),
		"layers":           map[string]any{},
	}
	layers := []map[string]any{}

	for _, layer := range mlp.Layers {
		layers = append(layers, layer.ToJSONMap())
	}
	data["layers"] = layers
	return data
	//fmt.Println(os.WriteFile(outfile, data, 0644))
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

// ZeroGradient sets gradients for all nodes in this mlp to 0
func (mlp *MultiLayerPerceptron) ZeroGradient() *MultiLayerPerceptron {
	for _, n := range mlp.Parameters() {
		n.Gradient = 0
	}
	return mlp
}

// Train runs the training, performing backpropagation and gradient descent.
func (mlp *MultiLayerPerceptron) Train(cycles int, learnrate float64, trainX [][]float64, trainY [][]float64) error {

	if len(inputs) <= 0 {
		return fmt.Errorf("train: inputs must contain something")
	} else if len(trainY) != len(trainX) {
		return fmt.Errorf("train: mismatch b/w input and output, want %d got %d", len(inputs), len(trainY))
	} else if len(trainX[0]) != mlp.NumberInputs {
		return fmt.Errorf("train: mismatch b/w train set dimensions & mlp dimensions, want %d got %d", (mlp.NumberInputs), len(trainX[0]))
	} else if len(trainY[0]) != mlp.NumberOutputs[len(mlp.NumberOutputs)-1] {
		return fmt.Errorf("train: mismatch b/w train set dimensions & mlp dimensions, want %d got %d", (mlp.NumberOutputs[len(mlp.NumberOutputs)-1]), len(trainY[0]))
	}

	var (
		trainx, trainy = toNodes(trainX, trainY)
	)

	for i := 0; i < cycles; i++ {
		netloss := mlp.MeanSquaredLoss(trainx, trainy)
		mlp.ZeroGradient()
		BackPropagate(netloss)
		fmt.Println(netloss)

		for _, node := range mlp.Parameters() {
			node.Data += -learnrate * node.Gradient
		}
	}

	// Graph("graph.png", netLoss)
	return nil
}

// MeanSquaredLoss returns the sum of (predicted - wanted) for each elem in trainy
func (mlp *MultiLayerPerceptron) MeanSquaredLoss(trainx [][]*Node, trainy [][]*Node) *Node {
	flattenedLosses := []*Node{}
	for i, inputSet := range trainx {
		pred := mlp.Forwards(inputSet)
		for j, elem := range trainy[i] {
			label := fmt.Sprintf("local_loss%d%d", i, j)
			loss := SquaredDifference(label, pred[j], elem)
			flattenedLosses = append(flattenedLosses, loss)
		}
	}

	return Add("loss_"+mlp.Label, flattenedLosses...)

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
