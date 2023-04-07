package main

import "math"

// Add computes the sum of data in supplied `nodes`. A fresh node with the result is returned and the operands are unchanged.
// `label` is the label of the output node.
// Sets the GradientUpdater function of output node.
// for c = a + b
// dc/da = 1.0
// dc/db = 1.0
func Add(label string, nodes ...*Node) *Node {
	sum := 0.0
	for i := range nodes {
		sum += nodes[i].Data
	}
	output := NewNode(label, sum)
	output.SetChildren(OperationAddition, nodes...)
	output.GradientUpdater = func() {
		for i := range nodes {
			nodes[i].Gradient = 1.0 * output.Gradient
		}
	}
	return output
}

// Sub computes the difference of data in supplied `nodes`.
// The first node is subtracted from by all other nodes.
// A fresh node with the result is returned and the operands are unchanged.
// `label` is the label of the output node.
// Sets the GradientUpdater function of output node.
// for d = a - b - c
// dc/da = 1.0
// dc/db = 1.0
func Sub(label string, nodes ...*Node) *Node {
	sub := 0.0
	if len(nodes) > 1 {
		sub = nodes[0].Data
		for _, node := range nodes[1:] {
			sub -= node.Data
		}
	}

	output := NewNode(label, sub)
	output.SetChildren(OperationSubtraction, nodes...)
	output.GradientUpdater = func() {
		for i := range nodes {
			nodes[i].Gradient = 1.0 * output.Gradient
		}
	}
	return output
}

// Multiply computes the product of data in supplied `nodes`. A fresh node with the result is returned and the operands are unchanged.
// `label` is the label of the output node.
// Sets the GradientUpdater function of output node.
// for d = a * b * c
// dd/da = b * c
// dd/db = a * c
func Multiply(label string, nodes ...*Node) *Node {
	product := 1.0
	for i := range nodes {
		product *= nodes[i].Data
	}
	output := NewNode(label, product)
	output.SetChildren(OperationMultiplication, nodes...)
	output.GradientUpdater = func() {
		for i := range nodes {
			gradient := 1.0
			for j := range nodes {
				if nodes[j] == nodes[i] {
					continue
				}
				gradient *= nodes[j].Data * output.Gradient
			}
			nodes[i].Gradient = gradient
		}
	}
	return output
}

// Tanh computes the tanh of the data in a single node. A fresh node with the result is returned and the operands are unchanged.
// `label` is the label of the output node.
// Sets the GradientUpdater function of output node.
// for b = tanh(a)
// db/da = 1 - (tanh(a) ^ 2)
func Tanh(label string, node *Node) *Node {
	tanh := math.Tanh(node.Data)
	output := NewNode(label, tanh)
	output.SetChildren(OperationTanh, node)
	output.GradientUpdater = func() {
		node.Gradient = (1 - math.Pow(tanh, 2)) * output.Gradient
	}
	return output
}

// Power computes the power of data in `node` to data in `power`. A fresh node with the result is returned and the operands are unchanged.
// `label` is the label of the output node.
// Sets the GradientUpdater function of output node taking the exponent as a constant, so:
// for c = a ** b
// dc/da = (b * a ** (b - 1))
func Power(label string, power *Node, node *Node) *Node {
	pow := math.Pow(node.Data, power.Data)
	output := NewNode(label, pow)
	op := OperationPowerOf
	if power.Data == 2 {
		op = OperationSquare
	} else if power.Data == 3 {
		op = OperationCube
	}
	output.SetChildren(op, node)
	output.GradientUpdater = func() {
		node.Gradient = power.Data * math.Pow(node.Data, power.Data-1) * output.Gradient
	}
	return output
}
