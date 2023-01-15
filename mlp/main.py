import numpy as np
from math import e as m_e
from math import isnan

print("~ Welcome to MLP ~")

def sigmoid(x):
    return 1 / (1 + m_e**(-x))

def deriv_sigmoid(x):
    exp = m_e**x
    if(1 / (2 + exp + 1/exp) > 1): 
        print("WARNING 1")
    return 1 / (2 + exp + 1/exp) # sigmoid(x) * (1-sigmoid(x))

def main():
    program = Program()
    program.add_layer(Layer(10, 3))  # L1
    program.add_layer(Layer(10, 10)) # L2
    program.add_layer(Layer(3, 10))  # L3
    program.scramble()
 
    # Dataset
    input_data_list = [
        [np.array([0, 0, 0]), np.array([0, 0, 1])], # 1
        [np.array([0, 0, 1]), np.array([0, 1, 0])], # 2
        [np.array([0, 1, 0]), np.array([0, 1, 1])], # 3 
        [np.array([0, 1, 1]), np.array([1, 0, 0])], # 4
        [np.array([1, 0, 0]), np.array([1, 0, 1])], # 5
        [np.array([1, 0, 1]), np.array([1, 1, 0])], # 6
        [np.array([1, 1, 0]), np.array([1, 1, 1])]  # 7
    ]
    n_datapoints = 7;

    # Train the network
    iterations = 1000
    for i in range(iterations):
        program.train(input_data_list, n_datapoints)
        print(f"{round((i/iterations)*100,1)}% | Cost: {round(program.current_slope, 5)}, Slope: {round(program.current_slope, 5)}")
    
    # Use the trained network
    while(True):
        num = input("Write a 3 bit number: ")
        input_layer = np.array([int(num[0]), int(num[1]), int(num[2])])
        result = program.execute(input_layer)
        print(f"Result: {round(result[0])}{round(result[1])}{round(result[2])}")

class Layer():
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.linear_activation_values = np.zeros(size)
        self.weights = np.zeros((size, input_size))
        self.biases = np.zeros(size)
    
    def scramble(self):
        self.linear_activation_values = np.zeros(self.size)
        self.weights = 2*np.random.rand(self.size, self.input_size)-1
        self.biases = 2*np.random.rand(self.size)-1

# Main class
class Program():
    current_cost = 1

    def __init__(self):
        self.layers = []
        self.num_layers = 0

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_layers += 1
    
    def remove_layer(self, layer):
        self.layers.remove(layer)
        self.num_layers -= 1

    # Fill weights and biases with random data
    def scramble(self):
        for layer in self.layers:
            layer.scramble()

    # Run the neural network as a function
    def execute(self, input_layer):
        linear_activation_values = self.layers[0].weights.dot(input_layer) + self.layers[0].biases
        for i in range(1, self.num_layers):
            linear_activation_values = self.layers[i].weights.dot(sigmoid(linear_activation_values)) + self.layers[i].biases
        return sigmoid(linear_activation_values)

    # Calculate cost for specified input in current configuration
    def cost(self, result, desired):
        diff = np.linalg.norm(result - desired)
        return diff * diff

    # SHORTHAND "LAMBDA"-FUNCTIONS FOR NEATER CHAIN RULE CALCULATIONS
    def lambda_1_weight(self, L, j, k):
        if L == 0:
            return self.input_values[k] * deriv_sigmoid(self.layers[0].linear_activation_values[j])
        else:
            return sigmoid(self.layers[L-1].linear_activation_values[k]) * deriv_sigmoid(self.layers[L].linear_activation_values[j])

    def lambda_1_bias(self, L, j):
        return deriv_sigmoid(self.layers[L].linear_activation_values[j])

    def lambda_2(self, L, j, k):
        return self.layers[L].weights[j, k] * deriv_sigmoid(self.layers[L].linear_activation_values[j])

    def lambda_3(self, j):
        return 2 * (sigmoid(self.layers[-1].linear_activation_values[j]) - self.desired_values[j])

    # Recursive function used for calculating dc/da
    def f(self, n, q):
        if n == self.num_layers:
            return self.lambda_3(q)
        else:
            return sum([
                self.lambda_2(n, p, q) * self.f(n+1, p)
            for p in range(self.layers[n].size)])

    def train(self, input_data_list, n_datapoints):
        n = sum([l.size * (1 + l.input_size) for l in self.layers]) # Total number of weights and biases / length of gradient vector
        gradient_vec = np.zeros(n) # Whole gradient vector

        total_cost = 0
        # Loop through the dataset
        for data_index in range(n_datapoints):
            data = input_data_list[data_index]
            self.input_values = data[0] # Neurons in input layer, selected from an input data list
            self.desired_values = data[1] # What we will compare the output to

            # Execute neural network and save linear activation values of layers
            self.layers[0].linear_activation_values = self.layers[0].weights.dot(self.input_values) + self.layers[0].biases
            for i in range(1, self.num_layers):
                self.layers[i].linear_activation_values = self.layers[i].weights.dot(sigmoid(self.layers[i-1].linear_activation_values)) + self.layers[i].biases

            # Cost calculations
            cost = self.cost(sigmoid(self.layers[-1].linear_activation_values), self.desired_values)
            total_cost += cost

            # Propagate backwards using the chain rule (with "lambda"-functions)
            index = 0
            for i, layer in enumerate(self.layers):
                for j in range(layer.size):
                    dc_da = self.f(i+1, j) # Recursive function for calculating derivative of cost with respect to nonlinear activation values
                    # bias
                    gradient_vec[index] += self.lambda_1_bias(i, j) * dc_da
                    index += 1
                    # weights
                    for k in range(layer.input_size):
                        gradient_vec[index] += self.lambda_1_weight(i, j, k) * dc_da
                        index += 1
        gradient_vec = gradient_vec / n_datapoints # Now we have the final averaged gradient vector!

        # APPLY CHANGES
        # Format: b, w1, w2, ... , wn, b, w1, w2, ... , and so on
        offset = 0 # How far into the gradient vector are we
        for layer in self.layers:
            n_weights = layer.size * layer.input_size
            n_bias = layer.size
            n_total = n_weights + n_bias
            layer.weights -= gradient_vec[[i + offset for i in range(n_total) if i%(layer.input_size+1)]].reshape(layer.size, layer.input_size)
            layer.biases -= gradient_vec[[i + offset for i in range(n_total) if i%(layer.input_size+1)==0]]
            offset += n_total

        # Log important values
        self.current_cost = total_cost/n_datapoints
        self.current_slope = sum([abs(x) for x in gradient_vec])/n_datapoints

if __name__ == "__main__":
    main()