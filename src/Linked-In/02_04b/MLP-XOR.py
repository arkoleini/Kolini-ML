import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))



class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias = 1.0):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object) #repesent number of neurons in each layer
        self.bias = bias
        self.network = [] # Initializes an empty list that will store the actual neuron objects
        self.values = []  # The list of lists of output values        
        
        # two nested loops to create neurons layer by layer
        # outer loop iterates over the layers
        # inner loop iterates over the number of neurons in that layer
        for i in range(len(layers)):   # Iterates through each layer index
            self.values.append([])  # Initialize the values for this layer``
            self.network.append([]) # Initialize the network for this layer
            self.values[i] = [0.0 for j in range(self.layers[i])]    # Creates placeholders for each neuron's output
            if i>0:  #network[0] is the input layer, so no neurons are created
                for j in range(self.layers[i]):  # Loop through each neuron in current layer
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))               
          
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        
    def set_weights(self, w_init):
        if (len(w_init) != len(self.network))-1:
            raise ValueError("Insufficient weight values provided.")
        for i in range(1, len(self.network)):  # start from 1 to skip the input layer
            layer = self.network[i]
            if len(w_init[i-1]) != len(layer):
                raise ValueError(f"Incorrect number of weight sets for layer {i}")
            for j, neuron in enumerate(layer):
                neuron.set_weights(w_init[i-1][j])
                
    def printweights(self):
        """Print the weights of the MLP."""
        print()
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
              print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()
                
                
    def run(self, x):
        """Run the MLP. x is a python list with the input values."""

        # Check if input size matches number of input neurons
        if len(x) != self.layers[0]:
            raise ValueError("Input size does not match number of input neurons.")

        # Set the input layer values to the provided input x
        self.values[0] = np.array(x)  
        
        # For each layer in the network (excluding input layer)
        for i in range(1, len(self.network)):
            layer = self.network[i]  # Get current layer (list of perceptrons)
            for j, neuron in enumerate(layer):
                # Run the perceptron using the previous layer's outputs as inputs
                neuron_output = neuron.run(self.values[i-1])
                # Store the output value in the current layer's values array
                self.values[i][j] = neuron_output
        # Return the final output layer's values (network's output)
        return self.values[-1]

mlp = MultiLayerPerceptron(layers=[2,2,1], bias=1.0)
mlp.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])
mlp.printweights()
print("MLP:")
print("0 XOR 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print("0 XOR 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print("1 XOR 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print("1 XOR 1 = {0:.10f}".format(mlp.run([1,1])[0]))
mlp.printvalues()