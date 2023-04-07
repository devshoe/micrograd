package main

// Operation is the type of artihmetic that produces a given node
type Operation string

// Operation addition and others are the constrainted set of operations allowed for production
const (
	OperationAddition       Operation = "+"
	OperationSubtraction    Operation = "-"
	OperationMultiplication Operation = "*"
	OperationDivision       Operation = "/"
	OperationPowerOf        Operation = "**"
	OperationSquare         Operation = "^2"
	OperationCube           Operation = "^3"
	OperationExp            Operation = "exp"
	OperationTanh           Operation = "tanh"
	OperationNil            Operation = "_noop_"
)
