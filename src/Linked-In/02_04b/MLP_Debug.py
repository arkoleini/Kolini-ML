import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias
        print(f"    Created Perceptron with {inputs} inputs, bias={bias}")
        print(f"    Initial weights: {self.weights}")

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
    """A multilayer perceptron class that uses the Perceptron class above."""

    def __init__(self, layers, bias = 1.0):
        """Return a new MLP object with the specified parameters.""" 
        print("=" * 60)
        print("STARTING MLP INITIALIZATION")
        print("=" * 60)
        print(f"Input parameters: layers={layers}, bias={bias}")
        print()
        
        # Initial assignments
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.network = []
        self.values = []
        
        print("After initial assignments:")
        print(f"self.layers = {self.layers}")
        print(f"self.bias = {self.bias}")
        print(f"self.network = {self.network}")
        print(f"self.values = {self.values}")
        print()
        
        # Main construction loop
        print("STARTING MAIN LOOP")
        print("-" * 40)
        
        for i in range(len(layers)):
            print(f"\n>>> ITERATION {i} (Layer {i}) <<<")
            print(f"Current layer size: {self.layers[i]} neurons")
            print(f"Loop variable i = {i}")
            
            # Before appending empty lists
            print(f"Before appending:")
            print(f"  self.values = {self.values}")
            print(f"  self.network = {self.network}")
            
            # Append empty lists
            self.values.append([])
            self.network.append([])
            
            print(f"After appending empty lists:")
            print(f"  self.values = {self.values}")
            print(f"  self.network = {self.network}")
            
            # Initialize values for this layer
            self.values[i] = [0.0 for j in range(self.layers[i])]
            
            print(f"After initializing values[{i}]:")
            print(f"  self.values[{i}] = {self.values[i]}")
            print(f"  Full self.values = {self.values}")
            
            # Create neurons (skip input layer)
            if i > 0:
                print(f"  Creating neurons for layer {i} (not input layer)")
                print(f"  Each neuron will have {self.layers[i-1]} inputs (from previous layer)")
                
                for j in range(self.layers[i]):
                    print(f"\n  >> Creating neuron {j} in layer {i}")
                    print(f"     Neuron {j} inputs: {self.layers[i-1]} (from layer {i-1})")
                    
                    # Create neuron
                    neuron = Perceptron(inputs=self.layers[i-1], bias=self.bias)
                    self.network[i].append(neuron)
                    
                    print(f"     self.network[{i}] now has {len(self.network[i])} neurons")
                    print(f"     Current network[{i}] length: {len(self.network[i])}")
                
                print(f"  Final network[{i}] has {len(self.network[i])} neurons")
            else:
                print(f"  Skipping neuron creation for layer {i} (input layer)")
            
            # Show current state
            print(f"\nEnd of iteration {i} state:")
            print(f"  self.layers = {self.layers}")
            print(f"  self.network lengths: {[len(layer) for layer in self.network]}")
            print(f"  self.values lengths: {[len(layer) for layer in self.values]}")
            
            # Show network structure so far
            for layer_idx, layer in enumerate(self.network):
                if layer_idx == 0:
                    print(f"  Layer {layer_idx}: {len(layer)} neurons (input layer - empty)")
                else:
                    print(f"  Layer {layer_idx}: {len(layer)} neurons")
        
        print("\n" + "=" * 60)
        print("BEFORE FINAL NUMPY CONVERSION")
        print("=" * 60)
        print(f"self.network structure:")
        for i, layer in enumerate(self.network):
            print(f"  Layer {i}: {len(layer)} elements, type: {type(layer)}")
        print(f"self.values structure:")
        for i, layer in enumerate(self.values):
            print(f"  Layer {i}: {len(layer)} elements, type: {type(layer)}")
        
        # Final conversion to numpy arrays
        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        
        print("\n" + "=" * 60)
        print("AFTER FINAL NUMPY CONVERSION")
        print("=" * 60)
        print(f"self.network structure:")
        for i, layer in enumerate(self.network):
            print(f"  Layer {i}: {len(layer)} elements, type: {type(layer)}")
        print(f"self.values structure:")
        for i, layer in enumerate(self.values):
            print(f"  Layer {i}: {len(layer)} elements, type: {type(layer)}")
        
        print("\n" + "=" * 60)
        print("FINAL NETWORK SUMMARY")
        print("=" * 60)
        print(f"Total layers: {len(self.network)}")
        for i in range(len(self.network)):
            if i == 0:
                print(f"Layer {i} (Input): {len(self.values[i])} input values")
            else:
                print(f"Layer {i}: {len(self.network[i])} neurons, each with {self.layers[i-1]} inputs + bias")
        print("=" * 60)

# Test the debug version
if __name__ == "__main__":
    print("Creating MLP with layers [2, 3, 1]")
    print("This means: 2 inputs -> 3 hidden neurons -> 1 output neuron")
    print()
    
    # Fix random seed for reproducible output
    np.random.seed(42)
    
    mlp = MultiLayerPerceptron([2, 3, 1], bias=1.0)
    
    print("\n" + "=" * 60)
    print("TESTING WITH DIFFERENT NETWORK SIZE")
    print("=" * 60)
    print("Creating MLP with layers [3, 4, 2, 1]")
    print("This means: 3 inputs -> 4 hidden -> 2 hidden -> 1 output")
    print()
    
    np.random.seed(42)  # Reset seed for consistent output
    mlp2 = MultiLayerPerceptron([3, 4, 2, 1], bias=1.0)