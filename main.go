package main

import "fmt"

func main() {
	n := NewNeuron(3, "neuron_1")
	o := n.Forwards([]*Node{NewNode(0.23, "input0"), NewNode(-3.0, "input1"), NewNode(0.77, "input2")})
	fmt.Println(o)
	o.Gradient = 1.0
	o.BackPropagate()
	o.Graph("graph.png")
}

func test() {
	x1, w1, x2, w2 := NewNode(2.0, "x1"), NewNode(-3.0, "w1"), NewNode(0.0, "x2"), NewNode(1.0, "w2")

	b := NewNode(6.8813735870195432, "b")

	x1w1, x2w2 := x1.Multiply(w1, "x1 * w1"), x2.Multiply(w2, "x2 * w2")

	sum := x1w1.Add(x2w2, "x1w1 + x2w2")

	preActivation := b.Add(sum, "result")

	postActivation := preActivation.Tanh("output")
	postActivation.Gradient = 1.0
	postActivation.BackPropagate()
	fmt.Println(postActivation.Graph("graph.png"))
}
