package main

import "fmt"

func main() {
	neuron, inputs := NewNeuron("neuron", 3), []*Node{NewNode("1", 0.5), NewNode("2", 0.213), NewNode("3", 0.9923)}
	want := NewNode("pred", 2.0)
	got := neuron.Forwards(inputs)
	loss := SquaredDiff("loss", got, want)
	BackPropagate(loss)
	Graph("pre.png", loss)
	Optimize(0.05, loss)
	got = neuron.Forwards(inputs)
	Graph("post.png", loss)

	newloss := SquaredDiff("loss", got, want)
	fmt.Printf("pre: %f, post: %f", loss.Data, newloss.Data)
}

/*
trainset := [][]float64{
		{255, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
	}

	outputset := [][]float64{
		{1},
		{2},
		{3},
	}

	mlp := NewMultiLayerPerceptron(3, []int{4, 4, 1})

	loss := mlp.MeanSquaredLoss(trainset, outputset)
	fmt.Println(loss.Graph("graph.png"))

*/
