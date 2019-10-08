import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split


class NeuralNetwork:

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, lr=0.01, momentum=0.9, iters=1000, seed=42):
        # Setup architecture of neural network
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Setup hyperparameters
        self.lr = lr
        self.momentum = momentum
        self.iters = iters

        # Set random seed fixed
        __set_seed(seed=seed)

        # Initial Weights
        # Input Layer Size x Hidden layer size Matrix Array for the First Layer
        self.Wh = self.__init_weights(
            self.input_layer_size, self.hidden_layer_size)

        # Hidden layer size x Output layer size Matrix Array for Hidden to Output
        self.Wo = np.random.randn(
            self.hidden_layer_size, self.output_layer_size)

        # Let's not forget the bias terms which allows to shift neuron's activation outputs
        self.Bh, self.Bo = __init_bias()

        # Stores loss per epoch
        self.loss = []

    def __set_seed(self, seed):
        np.random.seed(seed)

    def __sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def __sigmoid_prime(self, X):
        sx = self.__sigmoid(X)
        return sx * (1 - sx)

    def __tanh(self, X):
        return np.tanh(X)

    def __tanh_prime(self, X):
        return 1 - self.__tanh(X)**2

    def __init_weights(self, input_size, output_size, init_type='custom'):
        if init_type == 'uniform':
            W = np.random.random(input_size, output_size)
        elif init_type == 'normal':
            W = np.random.randn(input_size, output_size)
        elif init_type == 'custom':
            W = 2 * np.random.random((input_size, output_size)) - 1
        else:
            raise ValueError(
                'Parameter passed is wrong type, can only be "uniform", "normal", or "custom"')

    def __init_bias(self):
        Bh = np.full((1, self.hidden_layer_size), 0.1)
        Bo = np.full((1, self.output_layer_size), 0.1)
        return Bh, Bo

    def __feed_forward(self, X):
        """
        Calculate the NN inference using feed forward pass.
        X    - input matrix
        Zh   - hidden layer weighted input
        Zo   - output layer weighted input
        H    - hidden layer activation
        y    - output layer
        yHat - output layer predictions
        """

        # Weighted sum of inputs and hidden layer
        Z1 = (X @ self.Wh) + self.Bh
        H1 = __sigmoid(Z1)

        Z2 = (H1 @ self.Wo) + self.Bo
        y_hat = __sigmoid(Z2)
        return y_hat

    def __backprop(self, X):
        """
        Calculate weight adjustments
        """
        dW = np.dot(X.T, self.dO)
        db = np.sum(dO)
        return dW, db

    def fit(self, X, y):
        """
        Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """
        try:
            y.shape[1]
        except IndexError:
            y = y.reshape(-1, 1)

        for _ in range(iters):

            # Get predictions
            y_hat = self.__feed_forward(X)
            # Compute Error in output
            self.E = y_hat - y

            # Compute size of adjustments from hidden => output
            self.dO = self.E * self.__sigmoid_prime(y_hat)
            dW, db = self.__backprop(X)


def main():

    print('+'*80)
    url = 'https: // raw.githubusercontent.com/ryanleeallred/datasets/master/diabetes.csv'
    diabetes = pd.read_csv(url)
    print(diabetes.head())

    # Extract features
    features = list(diabetes)[:-1]
    target = 'Outcome'

    # Get df into X, y form
    X = diabetes[features].to_numpy()
    y = diabetes[target].to_numpy()

    # Split train, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print("---------- Shape ----------")
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit training set to Neural Network

    print('+'*80)


if __name__ == "__main__":
    main()
