import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_first_layer = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
    
        self.weights_first_layer_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #Sigmoid calculation
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    

    def train(self, features, targets):
    
        records = features.shape[0]
        input_weights = np.zeros(self.weights_input_to_first_layer.shape)
        output_weights = np.zeros(self.weights_first_layer_to_output.shape)
        for X, y in zip(features, targets):
            # Forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # Backpropagation function
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, input_weights, output_weights)
        self.update_weights(input_weights, output_weights, records)


    def forward_pass_train(self, X):
     
        inputs = np.array(X, ndmin=2).T
        inputs = inputs.reshape((1, -1))
        hidden_inputs = np.dot(inputs, self.weights_input_to_first_layer)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = hidden_outputs.dot(self.weights_first_layer_to_output) 
        final_outputs = final_inputs 
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        X2 = np.array(X, ndmin=2)
        X2 = X2.reshape((1, -1))
        y2 = np.array(y, ndmin=2)
        y2 = y2.reshape((1, -1))
        
        error = final_outputs - y2 
        
        output_error_term =  error

        hidden_output_error_term = np.dot(output_error_term, self.weights_first_layer_to_output.T)
        hidden_input_error_term = hidden_output_error_term * hidden_outputs * (1 - hidden_outputs)
     
        delta_weights_i_h += np.dot(X2.T, hidden_input_error_term)
        
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, records
                      ):
        
        self.weights_first_layer_to_output += -self.lr * delta_weights_h_o / records 
        self.weights_input_to_first_layer += -self.lr * delta_weights_i_h / records 
        
    def run(self, features):

        hidden_inputs = np.dot(features, self.weights_input_to_first_layer)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_first_layer_to_output) 
        final_outputs = final_inputs 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2500
learning_rate = 0.5
hidden_nodes = 15
output_nodes = 1