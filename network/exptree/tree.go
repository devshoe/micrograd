package exptree

import (
	"math"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"github.com/google/uuid"
)

// Optimize runs a single pass of optimization in order to minimize the loss function
func Optimize(learnrate float64, root *Node, passes ...int) {
	n := 1
	if len(passes) > 0 {
		n = passes[0]
	}

	for i := 0; i < n; i++ {
		BackPropagate(root)
		nodes := Topological(root, true)
		for _, node := range nodes {
			node.Data += -math.Abs(learnrate) * node.Gradient
		}
	}
}

// ZeroGradient zeroes all the gradients
func ZeroGradient(root *Node) {
	for _, node := range Topological(root, true) {
		node.Gradient = 0
	}
}

// BackPropagate traverses through the expression tree and calls the GradientUpdater function for each node
func BackPropagate(root *Node) {
	nodes := Topological(root, true)
	if len(nodes) > 0 {
		nodes[0].Gradient = 1.0
		for i := 0; i < len(nodes); i++ {
			nodes[i].GradientUpdater()
		}
	}

}

// Preorder recursively traverses through the tree and returns a list of nodes and edges
func Preorder(root *Node) (nodes []*Node, edges [][]*Node) {
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

	trace(root)

	return
}

// Topological traversal
func Topological(root *Node, reverse ...bool) (nodes []*Node) {
	var (
		visited   []*Node
		isVisited func(n *Node) bool
		trace     func(n *Node)
	)

	isVisited = func(n *Node) bool {
		for i := range visited {
			if visited[i] == n {
				return true
			}
		}
		return false
	}

	trace = func(n *Node) { // trace builds a representation of the tree using preorder traversal.
		if !isVisited(n) {
			visited = append(visited, n)
			for _, child := range n.ProducedByChildren {
				trace(child)
			}
			nodes = append(nodes, n)
		}
	}

	trace(root)

	if len(reverse) > 0 && reverse[0] {
		n := []*Node{}
		for i := len(nodes) - 1; i >= 0; i-- {
			n = append(n, nodes[i])
		}
		nodes = n
	}
	return

}

// Graph graphs the complete tree taking the calling node as root
func Graph(outfile string, root *Node) error {
	g := graphviz.New()
	graph, _ := g.Graph(graphviz.Directed)
	graph = graph.SetRankDir(cgraph.LRRank)
	nodes, edges := Preorder(root)

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
