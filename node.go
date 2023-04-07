package main

import (
	"fmt"
)

// Node is a data holder
type Node struct {
	Label               string
	Data                float64
	Gradient            float64
	ProducedByChildren  []*Node
	ProducedByOperation Operation
	GradientUpdater     func()
}

// NewNode creates a new node. you can pass in an optional `label`
// To set other items, you have to call `SetChildren`
func NewNode(label string, data float64) *Node {

	return (&Node{
		Label:               label,
		Data:                data,
		Gradient:            0.0,
		ProducedByChildren:  []*Node{},
		ProducedByOperation: OperationNil,
		GradientUpdater:     func() { return },
	})
}

// SetChildren sets children that created this node
func (n *Node) SetChildren(operation Operation, operands ...*Node) *Node {
	n.ProducedByOperation = operation
	n.ProducedByChildren = operands

	return n
}

// SetLabel sets the label. Useful for printing
func (n *Node) SetLabel(label string) *Node {
	n.Label = label
	return n
}

// Clone copies the exact node, only updating the provided `label`
func (n *Node) Clone(label string) *Node {
	return &Node{
		Label:               label,
		Data:                n.Data,
		Gradient:            n.Gradient,
		ProducedByChildren:  n.ProducedByChildren,
		ProducedByOperation: n.ProducedByOperation,
		GradientUpdater:     n.GradientUpdater,
	}
}

func (n *Node) String() string {
	return fmt.Sprintf("[Node `%s` | Data `%.4f` | Gradient `%.4f`]", n.Label, n.Data, n.Gradient)
}
