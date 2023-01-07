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

def deriv_sigmoid(x):
    return 1 / (m_e**x * (1 + m_e**(-x))**2)

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
            z = self.execute(input_layer, n_in)
            cost = self.cost(z, desired)
            total_cost += cost
        # Calculate average cost
        return total_cost / n_datapoints

    # SHORTHAND "LAMBDA"-FUNCTIONS FOR NEATER CHAIN RULE CALCULATIONS
    def lambda_1(self, L, j, k):
        if L == 1:
            return self.a_in[k] * deriv_sigmoid(self.z_L1[j])
        elif L == 2:
            return self.a_L1[k] * deriv_sigmoid(self.z_L2[j])
        elif L == 3:
            return self.a_L2[k] * deriv_sigmoid(self.z_out[j])
        else:
            print(f"Unknown layer '{L}'")

    def lambda_2(self, L, j, k):
        if L == 1:
            return self.weights_L1[j, k] * deriv_sigmoid(self.z_L1[j])
        elif L == 2:
            return self.weights_L2[j, k] * deriv_sigmoid(self.z_L2[j])
        elif L == 3:
            return self.weights_out[j, k] * deriv_sigmoid(self.z_out[j])
        else:
            print(f"Unknown layer '{L}'")

    #! Add lambda_b_1 through 3 for bias calculations?

    def lambda_3(self, j):
        return 2 * (self.a_out[j] - self.desired_values[j])

    def back_propagate(self, input_data_list, n_datapoints):
        n = self.n_L1 * (1 + self.n_in) + self.n_L2 * (1 + self.n_L1) + self.n_out * (1 + self.n_L2) # Number of weights and biases
        gradient_vec = np.empty(n, 1) # Whole gradient vector

        # ALL WE DO HERE IS COMPUTE A DERIVATIVE (THE GRADIENT)
        # DO THIS IN ONE PLACE
        # THERE WILL BE A LONG CHAIN RULE THAT WILL GIVE US THIS DERIVATIVE

        n_weights_out = self.n_out * self.n_L2
        n_bias_out = self.n_out
        #n_total_out = n_weights_out + n_bias_out
        avg_gradient_weight_out = 0
        avg_gradient_bias_out = 0
        for data_index in range(n_datapoints):
            # The following values are correct for the currently active trainging example (data point)
            self.input_values = input_data_list[data_index] # Neurons in input layer, selected from an input data list
            self.desired_values = np.c_[[0, 1]]
            self.a_in = sigmoid(self.input_values)
            self.z_L1 = self.weights_L1.dot(self.a_in) + self.bias_L1 # Neurons in layer 1
            self.a_L1 = sigmoid(self.z_L1)
            self.z_L2 = self.weights_L2.dot(self.a_L1) + self.bias_L2 # Neurons in layer 2
            self.a_L2 = sigmoid(self.z_L2)
            self.z_out = self.weights_out.dot(self.a_L2) + self.bias_out # Neurons in output layer
            self.a_out = sigmoid(self.z_out)

            # PROPAGATE BACKWARDS USING THE CHAIN RULE (with "lambda"-functions)
            #! Add support for biases
            index = 0
            # Out
            for j in range(self.n_out):
                for k in range(self.n_L2):
                    gradient_vec[index] += self.lambda_1(3, j, k) + self.lambda_3(j)
                    gradient_vec[index+1] += 0 #self.lambda_1(3, j, k) + self.lambda_3(j)
                    index += 2
            # Layer 2
            for j in range(self.n_L2):
                for k in range(self.n_L1):
                    gradient_vec[index] += self.lambda_1(2, j, k) * \
                        sum([
                            self.lambda_2(3, p, k) * self.lambda_3(p)
                        for p in range(self.n_out-1)])
                    gradient_vec[index+1] += 0 #self.lambda_1(3, j, k) + self.lambda_3(j)
                    index += 2
            # Layer 1
            for j in range(self.n_L1):
                for k in range(self.n_in):
                    gradient_vec[index] += self.lambda_1(1, j, k) * \
                        sum([
                            self.lambda_2(2, q, j) * 
                                sum([
                                    self.lambda_2(3, p, q) * self.lambda_3(p)
                                for p in range(self.n_out-1)])
                        for q in range(self.n_L2-1)])
                    gradient_vec[index+1] += 0 #self.lambda_1(3, j, k) + self.lambda_3(j)
                    index += 2
        gradient_vec /= n_datapoints # Now we have the final averaged gradient vector!

        # Apply changes

    # Train until satisfaction
    def train(self, input_data_list, n_datapoints):
        iterations = 100;
        for _ in range(iterations):
            pass
            #cost = self.avg_cost(input_data_list, n_datapoints)
            # Compute gradiant of cost function
            # z = output
            # dC/dw = dz/dw * da/dz * dc/da
            # Change weights and biases according to gradiant

if __name__ == "__main__":
    main()