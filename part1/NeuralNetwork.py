import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, 
                 hidden_layer_bias, output_layer_bias, learning_rate, bias_toggle):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        
        self.hidden_layer_bias = hidden_layer_bias
        self.output_layer_bias = output_layer_bias

        self.learning_rate = learning_rate
        self.bias_toggle = bias_toggle

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):

        # calculate the output for each hidden layer
        hidden_layer_outputs = []
        for hiddenNode in range(self.num_hidden):

            # calculate weighted sum
            weighted_sum = 0.
            for input in range(self.num_inputs):
                weighted_sum += inputs[input] * \
                    self.hidden_layer_weights[input][hiddenNode]
            # add bias
            weighted_sum += self.hidden_layer_bias[hiddenNode] * self.bias_toggle

            # apply activation function
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        # calculate the output for each output layer
        output_layer_outputs = []
        for outputNode in range(self.num_outputs):
            
            # calculate weighted sum
            weighted_sum = 0.
            for hiddenOutput in range(self.num_hidden):
                weighted_sum += hidden_layer_outputs[hiddenOutput] * \
                    self.output_layer_weights[hiddenOutput][outputNode]
            # add bias
            weighted_sum += self.output_layer_bias[outputNode] * self.bias_toggle
            
            # apply activation function
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        # Calculate output layer betas
        output_layer_betas = np.zeros(self.num_outputs)
        for outputNode in range(self.num_outputs):
            output_layer_betas[outputNode] = desired_outputs[outputNode] - \
                output_layer_outputs[outputNode]
        
        #print('OL betas: ', output_layer_betas)

        # Calculate hidden layer betas
        hidden_layer_betas = np.zeros(self.num_hidden)
        for hiddenNode in range(self.num_hidden):
            for outputNode in range(self.num_outputs):
                hidden_layer_betas[hiddenNode] += self.output_layer_weights[hiddenNode][outputNode] \
                    * output_layer_outputs[outputNode] * (1 - output_layer_outputs[outputNode]) \
                    * output_layer_betas[outputNode]

        #print('HL betas: ', hidden_layer_betas)

        # Calculate output layer weight changes
        delta_output_layer_weights = np.zeros(
            (self.num_hidden, self.num_outputs))
        
        for hiddenNode in range(self.num_hidden):
            for outputNode in range(self.num_outputs):
                delta_output_layer_weights[hiddenNode][outputNode] = \
                    self.learning_rate * hidden_layer_outputs[hiddenNode] \
                    * output_layer_outputs[outputNode] * (1 - output_layer_outputs[outputNode]) \
                    * output_layer_betas[outputNode]
           
        # Calculate hidden layer weight changes
        delta_hidden_layer_weights = np.zeros(
            (self.num_inputs, self.num_hidden))

        for input in range(self.num_inputs):
            for hiddenNode in range(self.num_hidden):
                delta_hidden_layer_weights[input][hiddenNode] = \
                    self.learning_rate * inputs[input] \
                    * hidden_layer_outputs[hiddenNode] * (1 - hidden_layer_outputs[hiddenNode]) \
                    * hidden_layer_betas[hiddenNode]
        
        # calculate delta bias for hidden and output layers
        delta_output_layer_bias = np.zeros(self.num_outputs)
        delta_hidden_layer_bias = np.zeros(self.num_hidden)
        if self.bias_toggle == 1:
            for outputNode in range(self.num_outputs):
                delta_output_layer_bias[outputNode] = self.learning_rate * output_layer_outputs[outputNode] \
                        * (1 - output_layer_outputs[outputNode]) \
                        * output_layer_betas[outputNode]
            for hiddenNode in range(self.num_hidden):
                delta_hidden_layer_bias[hiddenNode] = self.learning_rate * hidden_layer_outputs[hiddenNode] \
                        * (1 - hidden_layer_outputs[hiddenNode]) \
                        * hidden_layer_betas[hiddenNode]

        # Return the weights we calculated, so they can be used to update all the weights.
        # additionally, return the bias changes
        return delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_bias, delta_hidden_layer_bias

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_bias, delta_hidden_layer_bias):
        # Update the weights
        self.output_layer_weights += delta_output_layer_weights
        self.hidden_layer_weights += delta_hidden_layer_weights

        # Update the biases
        if self.bias_toggle == 1:
            self.output_layer_bias += delta_output_layer_bias
            self.hidden_layer_bias += delta_hidden_layer_bias

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch no. = ', epoch + 1)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(
                    instance)
                delta_output_layer_weights, delta_hidden_layer_weights, \
                    delta_output_layer_bias, delta_hidden_layer_bias = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = np.argmax(output_layer_outputs)
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights,
                                    delta_hidden_layer_weights, 
                                    delta_output_layer_bias, 
                                    delta_hidden_layer_bias)

            # Print accuracy achieved over this epoch
            acc = 0
            for i in range(len(desired_outputs)):
                if np.argmax(desired_outputs[i]) == predictions[i]:
                    acc += 1
            acc /= len(desired_outputs)
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(
                instance)
            predicted_class = np.argmax(output_layer_outputs)
            predictions.append(predicted_class)
        return predictions
