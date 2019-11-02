import numpy as np

class NeuralNetwork(object):
        def __init__(self, inputLayerSize=4, outputLayerSize=1, hiddenLayerSize=4):
            # Set up Architecture of Neural Network
            self.inputs = inputLayerSize
            self.hiddenNodes = outputLayerSize
            self.outputNodes = hiddenLayerSize
        
        #setting random weights for each layer of the newtork
            self.weights1 = np.random.rand(self.inputs, self.hiddenNodes)
            self.weights2 = np.random.rand(self.hiddenNodes, self.outputNodes)
        
        def sigmoid(self, s):
            return 1 / (1+np.exp(-s))

        def sigmoidPrime(self, s):
            return s * (1 - s)

        def feed_forward(self, X):
            """
            Calculate the NN inference using feed forward.
            aka "predict"
            """

            # Weighted sum of inputs => hidden layer
            self.hidden_sum = np.dot(X, self.weights1)

            # Activations of weighted sum
            self.activated_hidden = self.sigmoid(self.hidden_sum)

            # Weight sum between hidden and output
            self.output_sum = np.dot(self.activated_hidden, self.weights2)

            # Final activation of output
            self.activated_output = self.sigmoid(self.output_sum)

            return self.activated_output

        def backward(self, X,y,o):
            """
            Backward propagate through the network
            """

            # Error in Output
            self.o_error = y - o

            # Apply Derivative of Sigmoid to error
            # How far off are we in relation to the Sigmoid f(x) of the output
            # ^- aka hidden => output
            self.o_delta = self.o_error * self.sigmoidPrime(o)

            # z2 error
            # z2 error is how much are the weights of the hidden layer responsible for the error
            self.z2_error = self.o_delta.dot(self.weights2.T)
            # How much of that "far off" can explained by the input => hidden
            self.z2_delta = self.z2_error * self.sigmoidPrime(self.activated_hidden)

            # Adjustment to first set of weights (input => hidden)
            self.weights1 += X.T.dot(self.z2_delta)
            # Adjustment to second set of weights (hidden => output)
            self.weights2 += self.activated_hidden.T.dot(self.o_delta)


        def train(self, X, y):
            o = self.feed_forward(X)
            self.backward(X,y,o)