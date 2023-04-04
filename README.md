# How a nn neuron works
    - The weights of a neuron can be one or many. Depends on the input size.
    - Say your input is a vector of size 3: you must build neurons of input vector size 3, 3 weights & a single bias
    - The bias of a neuron is only relevant for activation.
    - Activation is basically a function that asks something like `should I trigger?`. For micrograd, it is a tanh
    - Finally, during forward propagation, the output is 