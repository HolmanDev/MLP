import numpy as np
from math import e as m_e

print("~ Welcome to MLP ~")

def main():
    program = Program()
 
    input_data_list = [0] * 100;
    n_datapoints = 100;
    program.scramble()
    program.train(input_data_list, n_datapoints)

    num = input("Pick a number between 0 and 2")
    input_layer = [0, 0, 0]
    input_layer[num] = 1
    result = program.execute(input_layer)
    
    print(result)

def sigmoid(x):
        return 1 / (1 + m_e**(-x))

# Main class
class Program():
    n_in = 3
    n_L1 = 10
    n_L2 = 10
    n_out = 2

    # Fill weights and biases with random data
    def scramble(self):
        self.weights_L1 = np.random.rand(self.n_L1, self.n_in) # Weights into layer 1
        self.bias_L1 = np.random.rand(self.n_L1, 1) # Biases into layer 1
        self.weights_L2 = np.random.rand(self.n_L2, self.n_L1) # Weights into layer 2
        self.bias_L2 = np.random.rand(self.n_L2, 1) # Biases into layer 2
        self.weights_out = np.random.rand(self.n_out, self.n_L2) # Weights into output layer
        self.bias_out = np.random.rand(self.n_out, 1) # Biases into output layer

    # Run the neural network as a function
    def execute(self, input_layer):
        layer1 = sigmoid(self.weights_L1.dot(input_layer) + self.bias_L1) # Neurons in layer 1
        layer2 = sigmoid(self.weights_L2.dot(layer1) + self.bias_L2) # Neurons in layer 2
        return sigmoid(self.weights_out.dot(layer2) + self.bias_out) # Neurons in output

    # Calculate cost for specified input in current configuration
    def cost(self, result, desired):
        diff = result - desired
        return np.dot(diff.T, diff)[0][0] # Calculate inner product of squared difference

    # Average cost over all input data
    def avg_cost(self, input_data_list, n_datapoints):
        total_cost = 0
        # Loop through all inputs
        for i in range(n_datapoints):
            n_in = 100;
            input_layer = input_data_list[i] # Select input data from list
            desired = np.c_[[0, 1]]
            output = self.execute(input_layer, n_in)
            total_cost += self.cost(output, desired)
        # Calculate average cost
        return total_cost / n_datapoints

    # Train until satisfaction
    def train(self, input_data_list, n_datapoints):
        iterations = 100;
        for _ in range(iterations):
            cost = self.avg_cost(input_data_list, n_datapoints)
            # Compute gradiant of cost function
            # Change weights and biases according to gradiant

if __name__ == "__main__":
    main()