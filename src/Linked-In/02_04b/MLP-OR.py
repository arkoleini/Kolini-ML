import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.wieghts = (np.random.rand(inputs + 1)*2) -1  # Random weights for inputs and bias
        self.bias = bias
        pass
    
    def set_weights(self, w_inits):
        """Set the weight, w_init is a python list with the weights.s of the perceptron."""
        self.wieghts = np.array(w_inits)
    
    def sigmoid(self, x):
        """The sigmoid activation function."""
        return 1 / (1 + np.exp(x))
    

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x, self.bias), self.wieghts)
        return self.sigmoid(x_sum)
        
        
        
neuron = Perceptron(inputs=3)
neuron.set_weights([10, 10, -15])

print ("Gsyr:")
print ("0 0 = {0: .10f}".format((neuron.run([0, 0]))))
print ("0 1 = {0: .10f}".format((neuron.run([0, 1]))))
print ("1 0 = {0: .10f}".format((neuron.run([1, 0]))))
print ("1 1 = {0: .10f}".format((neuron.run([1, 1]))))

