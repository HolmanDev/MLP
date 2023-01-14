import numpy as np
from math import e as m_e
from math import isnan

print("~ Welcome to MLP ~")

def main():
    program = Program()
 
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
    program.scramble()
    iterations = 200
    for i in range(iterations):
        program.train(input_data_list, n_datapoints)
        print(f"{round((i/iterations)*100,1)}% | Cost: {round(program.current_slope, 5)}, Slope: {round(program.current_slope, 5)}")

    while(True):
        num = input("Write a 3 bit number: ")
        input_layer = np.array([int(num[0]), int(num[1]), int(num[2])])
        result = program.execute(input_layer)
        print(f"Result: {round(result[0])}{round(result[1])}{round(result[2])}")

def sigmoid(x):
    return 1 / (1 + m_e**(-x))

def deriv_sigmoid(x):
    exp = m_e**x
    if(1 / (2 + exp + 1/exp) > 1): 
        print("WARNING 1")
    return 1 / (2 + exp + 1/exp) # sigmoid(x) * (1-sigmoid(x))

# Main class
class Program():
    current_cost = 1

    n_in = 3
    n_L1 = 10
    n_L2 = 10
    n_out = 3

    # Fill weights and biases with random data
    def scramble(self):
        self.weights_L1 = 2*np.random.rand(self.n_L1, self.n_in)-1 # Weights into layer 1
        self.bias_L1 = 2*np.random.rand(self.n_L1)-1 # Biases into layer 1
        self.weights_L2 = 2*np.random.rand(self.n_L2, self.n_L1)-1 # Weights into layer 2
        self.bias_L2 = 2*np.random.rand(self.n_L2)-1 # Biases into layer 2
        self.weights_out = 2*np.random.rand(self.n_out, self.n_L2)-1 # Weights into output layer
        self.bias_out = 2*np.random.rand(self.n_out)-1 # Biases into output layer

    # Run the neural network as a function
    def execute(self, input_layer):
        layer1 = sigmoid(self.weights_L1.dot(input_layer) + self.bias_L1) # Neurons in layer 1
        layer2 = sigmoid(self.weights_L2.dot(layer1) + self.bias_L2) # Neurons in layer 2
        output = sigmoid(self.weights_out.dot(layer2) + self.bias_out) # Neurons in output
        return output

    # Calculate cost for specified input in current configuration
    def cost(self, result, desired):
        diff = np.linalg.norm(result - desired)
        return diff * diff

    # SHORTHAND "LAMBDA"-FUNCTIONS FOR NEATER CHAIN RULE CALCULATIONS
    def lambda_1_weight(self, L, j, k):
        if L == 1:
            return self.a_in[k] * deriv_sigmoid(self.z_L1[j])
        elif L == 2:
            return self.a_L1[k] * deriv_sigmoid(self.z_L2[j])
        elif L == 3:
            return self.a_L2[k] * deriv_sigmoid(self.z_out[j])
        else:
            print(f"Unknown layer '{L}'")

    def lambda_1_bias(self, L, j):
        if L == 1:
            return deriv_sigmoid(self.z_L1[j])
        elif L == 2:
            return deriv_sigmoid(self.z_L2[j])
        elif L == 3:
            return deriv_sigmoid(self.z_out[j])
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

    def lambda_3(self, j):
        return 2 * (self.a_out[j] - self.desired_values[j])
        

    def train(self, input_data_list, n_datapoints):
        n = self.n_L1 * (1 + self.n_in) + self.n_L2 * (1 + self.n_L1) + self.n_out * (1 + self.n_L2) # Number of weights and biases
        gradient_vec = np.zeros(n) # Whole gradient vector

        total_cost = 0
        for data_index in range(n_datapoints):
            # The following values are correct for the currently active trainging example (data point)
            data = input_data_list[data_index]
            self.input_values = data[0] # Neurons in input layer, selected from an input data list
            self.desired_values = data[1]
            self.a_in = self.input_values
            self.z_L1 = self.weights_L1.dot(self.a_in) + self.bias_L1 # Neurons in layer 1
            self.a_L1 = sigmoid(self.z_L1)
            self.z_L2 = self.weights_L2.dot(self.a_L1) + self.bias_L2 # Neurons in layer 2
            self.a_L2 = sigmoid(self.z_L2)
            self.z_out = self.weights_out.dot(self.a_L2) + self.bias_out # Neurons in output layer
            self.a_out = sigmoid(self.z_out)

            cost = self.cost(self.a_out, self.desired_values)
            total_cost += cost

            # PROPAGATE BACKWARDS USING THE CHAIN RULE (with "lambda"-functions)
            index = 0
            # Output layer
            for j in range(self.n_out):
                dc_da = self.lambda_3(j)
                # bias
                gradient_vec[index] += self.lambda_1_bias(3, j) * dc_da
                index += 1
                # weights
                for k in range(self.n_L2):                
                    gradient_vec[index] += self.lambda_1_weight(3, j, k) * dc_da
                    index += 1
            # Layer 2
            for j in range(self.n_L2):
                dc_da = sum([
                    self.lambda_2(3, p, j) * self.lambda_3(p)
                    for p in range(self.n_out)])
                #bias
                gradient_vec[index] += self.lambda_1_bias(2, j) * dc_da
                index += 1
                # weights
                for k in range(self.n_L1):
                    gradient_vec[index] += self.lambda_1_weight(2, j, k) * dc_da
                    index += 1
            # Layer 1
            for j in range(self.n_L1):
                dc_da = sum([
                    self.lambda_2(2, q, j) * 
                        sum([
                            self.lambda_2(3, p, q) * self.lambda_3(p)
                        for p in range(self.n_out)])
                    for q in range(self.n_L2)])
                # bias
                gradient_vec[index] += self.lambda_1_bias(1, j) * dc_da
                index += 1
                # weights
                for k in range(self.n_in):
                    gradient_vec[index] += self.lambda_1_weight(1, j, k) * dc_da
                    index += 1
        gradient_vec = gradient_vec / n_datapoints # Now we have the final averaged gradient vector!

        # APPLY CHANGES
        # Format: b, w1, w2, ... , wn, b, w1, w2, ... , and so on
        # Output layer
        n_weights_out = self.n_out * self.n_L2
        n_bias_out = self.n_out
        n_total_out = n_weights_out + n_bias_out
        self.bias_out -= gradient_vec[[i for i in range(n_total_out) if i%(self.n_L2+1)==0]]
        self.weights_out -= gradient_vec[[i for i in range(n_total_out) if i%(self.n_L2+1)]].reshape(self.n_out, self.n_L2)
        # Layer 2
        n_weights_L2 = self.n_L2 * self.n_L1
        n_bias_L2 = self.n_L2
        n_total_L2 = n_weights_L2 + n_bias_L2
        self.bias_L2 -= gradient_vec[[n_total_out + i for i in range(n_total_L2) if i%(self.n_L1+1)==0]]
        self.weights_L2 -= gradient_vec[[n_total_out + i for i in range(n_total_L2) if i%(self.n_L1+1)]].reshape(self.n_L2, self.n_L1)
        # Layer 1
        n_weights_L1 = self.n_L1 * self.n_in
        n_bias_L1 = self.n_L1
        n_total_L1 = n_weights_L1 + n_bias_L1
        self.bias_L1 -= gradient_vec[[n_total_out + n_total_L2 + i for i in range(n_total_L1) if i%(self.n_in+1)==0]]
        self.weights_L1 -= gradient_vec[[n_total_out + n_total_L2 + i for i in range(n_total_L1) if i%(self.n_in+1)]].reshape(self.n_L1, self.n_in)

        self.current_cost = total_cost/n_datapoints
        self.current_slope = sum([abs(x) for x in gradient_vec])/n_datapoints

        #print(f"Slope: {round(self.current_slope, 6)}, Cost:{round(self.current_cost, 6)}, Nan: {isnan(self.weights_L1[0,0])}")

if __name__ == "__main__":
    main()