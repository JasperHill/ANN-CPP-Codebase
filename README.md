# ANN-Cpp-Codebase
A (now defunct) attempt to create an independent codebase for constructing and training ANNs

Codebase was to allow for network configurations of arbitrary depth and dimension

The fundamental unit was the Node, which featured an activation function, a bias B, and a weight vector w.
The activation function could be either a hyperbolic tangent, a pi function (the elements of the input vector are multiplied), or a sigma function (the elements of the input vector are added).

The next higher unit was the Neuron, which contained at least one Node. The Neuron contained a vector of strings indicating the activation function of each Node. It also featured a null pointer pGate, which could be initialized as an LSTMGate pointer.

The Layer class contained an array of Neuron pointers. Inputs were passed to the Layer's function f, which allowed for the input of the previous Layer's output as layer_x as well as an optional input of the unmodified training data, training_x.

The entire collection of Layers was contained in a NeuralNet, which stored information about about the network's total squared error, entropy, and loss function (which could only be the squared error with or without entropy). The NeuralNet, also stored information about the number of parameters to be optimized as this could be used in the Brownian Simplex method of optimization.

Optimization could be achieved via gradient descent, however, an attempt was made to construct a Brownian NeuralNet Simplex that would stochastically arrive at a good extremum. The method of this class follows the description given by Hall et al. in Numerical Recipes. The class is incomplete.

The codebase is incomplete.
