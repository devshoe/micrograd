package main

import (
	"fmt"
	"math"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"github.com/google/uuid"
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
func NewNode(data float64, label ...string) *Node {
	l := ""
	if len(label) > 0 {
		l = label[0]
	}

	return (&Node{
		Label:               l,
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

// Add adds another `Node` to this one. A fresh node with the result is returned and the operands are unchanged.
// Sets the GradientUpdater function of output node.
// for c = a + b
// dc/da = 1.0
// dc/db = 1.0
func (n *Node) Add(other *Node, label ...string) *Node {
	output := NewNode(n.Data+other.Data, label...)
	output.SetChildren(OperationAddition, n, other)
	output.GradientUpdater = func() {
		n.Gradient = 1.0 * output.Gradient
		other.Gradient = 1.0 * output.Gradient
	}
	return output
}

// Multiply multiplies another `Node` to this one. A fresh node with the result is returned and the operands are unchanged.
// Sets the GradientUpdater function of output node.
// for c = a * b, dc/da = b & dc/db = a
func (n *Node) Multiply(other *Node, label ...string) *Node {
	output := NewNode(n.Data*other.Data, label...)
	output.SetChildren(OperationMultiplication, n, other)
	output.GradientUpdater = func() {
		n.Gradient = other.Data * output.Gradient
		other.Gradient = n.Data * output.Gradient
	}
	return output
}

// Tanh computes the tanh of the data in this node. A fresh node with the result is returned and the operands are unchanged.
// Sets the GradientUpdater function of output node.
// for c = a * b, dc/da = b & dc/db = a
func (n *Node) Tanh(label ...string) *Node {
	output := NewNode(math.Tanh(n.Data), label...)
	output.SetChildren(OperationTanh, n)
	output.GradientUpdater = func() {
		n.Gradient = (1 - math.Pow(math.Tanh(n.Data), 2)) * output.Gradient
	}

	return output
}

// BackPropagate traverses through the expression tree and calls the GradientUpdater functions for each node
func (n *Node) BackPropagate() {
	nodes, _ := n.Traverse()
	for i := 0; i < len(nodes); i++ {
		nodes[i].GradientUpdater()
	}

}

// Traverse recurses through the tree and returns a list of nodes and edges
// Uses preordering to accomplish traversal.
func (n *Node) Traverse() (nodes []*Node, edges [][]*Node) {
	var (
		isVisited func(n *Node) bool
		trace     func(n *Node)
	)

	isVisited = func(n *Node) bool {
		for i := range nodes {
			if nodes[i] == n {
				return true
			}
		}
		return false
	}

	trace = func(n *Node) { // trace builds a representation of the tree using preorder traversal.
		if !isVisited(n) {
			nodes = append(nodes, n)
			for _, child := range n.ProducedByChildren {
				edges = append(edges, []*Node{n, child})
				trace(child)
			}
		}
	}

	trace(n)

	return
}

// Graph graphs the complete tree taking the calling node as root
func (n *Node) Graph(outfile string) error {
	g := graphviz.New()
	graph, _ := g.Graph(graphviz.Directed)
	graph = graph.SetRankDir(cgraph.LRRank)
	nodes, edges := n.Traverse()

	for _, node := range nodes {
		elemnode, _ := graph.CreateNode(node.Label)
		elemnode = elemnode.SetLabel(node.String()).SetShape(cgraph.RectangleShape)

		if node.ProducedByOperation != OperationNil {
			opnode, _ := graph.CreateNode(node.Label + "_" + string(node.ProducedByOperation))
			opnode.SetLabel(string(node.ProducedByOperation))
			graph.CreateEdge(uuid.NewString(), opnode, elemnode)
		}
	}

	for _, edge := range edges {
		tonode, _ := graph.Node(edge[0].Label + "_" + string(edge[0].ProducedByOperation))
		fromnode, _ := graph.Node(edge[1].Label)

		graph.CreateEdge("", fromnode, tonode)
	}

	if f, err := os.Create(outfile); err != nil {
		return err
	} else {
		return g.Render(graph, graphviz.PNG, f)
	}

}

func (n *Node) String() string {
	return fmt.Sprintf("[Node `%s` | Data `%.4f` | Gradient `%.4f`]", n.Label, n.Data, n.Gradient)
}
