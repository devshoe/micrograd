package main

import "fmt"

func main() {
	a, b := NewNode(53, "a"), NewNode(22, "b")

	c := a.Add(b, "c")

	d := c.Multiply(a, "d")

	fmt.Println(d.Graph("graph.png"))
}
