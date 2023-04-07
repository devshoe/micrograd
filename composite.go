package main

func SquaredDiff(label string, nodes ...*Node) *Node {
	diff := Sub(label+"_diff", nodes...)
	pow := Power(label, NewNode(label+"const", 2), diff)
	return pow
}
