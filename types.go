package main

// Operation is the type of artihmetic that produces a given node
type Operation string

// Operation addition and others are the constrainted set of operations allowed for production
const (
	OperationAddition       = "+"
	OperationSubtraction    = "-"
	OperationMultiplication = "*"
	OperationDivision       = "/"
	OperationPowerOf        = "**"
	OperationExp            = "exp"
	OperationTanh           = "tanh"
	OperationNil            = "_noop_"
)
