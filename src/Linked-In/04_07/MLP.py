import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, layers, bias = 1.0, eta = 0.5):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []       # The list of lists of error terms (lowercase deltas)

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]   #*** hold error for each neuron
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias)) #each neuron created from previous call Perceptron

        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d],dtype=object) #*** errors return from array

    def set_weights(self, w_init):
        """Set the weights. 
           w_init is a 3D list with the weights for all but the input layer."""
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def print_weights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x,dtype=object)  
        self.values[0] = x   #first layer initialiaze with input self.values[0] = x ,   Example: if x=[0,1], then self.values[0] = [0,1].
        for i in range(1,len(self.network)):   #Loops over each layer after the input layer (hidden layers + output layer).  ## loop start from 1 since item zero already taken
            for j in range(self.layers[i]):    #Loops through each neuron in the current layer.
                self.values[i][j] = self.network[i][j].run(self.values[i-1])   #Take neuron j in layer i, ## Feed it all outputs from the previous layer (self.values[i-1]),  ##Get its output using run(),
        return self.values[-1] #After all layers are done, return the final layer’s outputs (the prediction).
    
    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x,dtype=object)
        y = np.array(y,dtype=object)

        # Challenge: Write the Backpropagation Algorithm. 
        # Here you have it step by step:

        # STEP 1: Feed a sample to the network 
        outputs = self.run(x)
                
        # STEP 2: Calculate the MSE
        mse = 0
        error = (y- outputs)
        mse = np.sum(error ** 2) / len(self.values[-1])
        

        # STEP 3: Calculate the output error terms
        
        # --- Step 3: Output-layer error terms (backprop) ---
        self.d[-1] = outputs * (1 - outputs) * (error) # This calculates the error term for the output layer neurons


        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1,len(self.network)-1)):   #This loops backward through hidden layers (hence reversed)    #i goes from the last hidden layer toward the first hidden layer`
            for h in range(len(self.network[i])):          #For each neuron h in layer i, it calculates fwd_error
                fwd_error = 0.0
                for k in range(self.layers[i+1]):    #The inner loop over k sums up error contributions from all neurons in the next layer
                    fwd_error += self.d[i+1][k] * self.network[i+1][k].weights[h]
                self.d[i][h] = fwd_error * self.values[i][h] * (1 - self.values[i][h])

        # STEPS 5 & 6: Calculate the deltas and update the weights
        for i in range(1,len(self.network)):  #goes though layers
            for j in range(self.layers[i]):   #goes though neurons
                for k in range(self.layers[i-1]+1):  #goes though inputs   loop from zero to number of neuron on that layer +1 because of bias weight
                    if k==self.layers[i-1]:  #bias
                                        # --- Bias weight branch ---
                                        # k points to the *extra* weight reserved for bias.
                                        # The "input" feeding that weight is NOT a neuron output;
                                        # it's the constant bias value (usually 1.0).
                                        # Delta rule: Δw = η * δ(neuron) * input_to_this_weight
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                                        # --- Regular input weight branch ---
                                        # k points to a real connection coming from the previous layer’s neuron k.
                                        # The "input" feeding that weight is the output of that previous neuron.
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta    # apply correction for weights from calculated delta
        return mse


#test code
mlp = MultiLayerPerceptron(layers=[2,2,1])
print("\nTraining Neural Network as an XOR Gate...\n")
mse_history = []   

for i in range(3000):
    mse = 0.0
    mse += mlp.bp([0,0],[0])
    mse += mlp.bp([0,1],[1])
    mse += mlp.bp([1,0],[1])
    mse += mlp.bp([1,1],[0])
    mse = mse / 4
    mse_history.append(mse)
    if(i%100 == 0):
        print (mse)

mlp.print_weights()

# Plot epochs vs MSE
plt.figure(figsize=(8,4))
plt.plot(range(1, len(mse_history)+1), mse_history, '-o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training: Epoch vs MSE')
plt.grid(True)
plt.tight_layout()
plt.show()
    
print("MLP:")
print ("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))
