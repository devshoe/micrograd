package main

import "fmt"

var (
	testX1, testW1, testX2, testW2, testB = NewNode("x1", 2), NewNode("w1", -3), NewNode("x2", 0), NewNode("w2", 1), NewNode("b", 6.8813735870195432)
	inputs                                = []*Node{NewNode("1", 0.5), NewNode("2", 0.213), NewNode("3", 0.9923)}
	neuron                                = NewNeuron("neuron", 3)

	trainY = NewNode("pred", 2.0)
)

func main() {

	trainX := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	trainY := [][]float64{
		{1.0}, {-0.0003}, {-0.01}, {1.0},
	}

	fmt.Println("train: x", trainX, " y ", trainY)
	mlp := NewMultiLayerPerceptron("is_odd", 3, []int{4, 4, 1})
	mlp.Train(1000, 0.1, trainX, trainY)

	d := mlp.ToJSONMap()

	for k, v := range d {
		fmt.Println(k, v)
	}
	fmt.Println(mlp.Forwards([]*Node{NewNode("in_base0", 2), NewNode("in_base1", 3), NewNode("in_base2", -1)}))   // expect 1
	fmt.Println(mlp.Forwards([]*Node{NewNode("in_base0", 3), NewNode("in_base1", -1), NewNode("in_base2", 0.5)})) // expect -1
}

func oddevenTrainset(size int) ([][]float64, [][]float64) {
	in, out := [][]float64{}, [][]float64{}
	for i := 0; i < size; i++ {
		in = append(in, []float64{float64(i)})
		isOdd := 0.0
		if i%2 != 0 {
			isOdd = 1
		}

		out = append(out, []float64{isOdd})
	}

	return in, out
}
func neuronTest() {
	neuronOp := Tanh("activated", Add("biased", Multiply("x1w1", testX1, testW1), Multiply("x2w2", testX2, testW2), testB))

	loss := SquaredDifference("loss", neuronOp, trainY)
	BackPropagate(loss)
	Optimize(0.05, loss)
	Graph("graph.png", loss)
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
