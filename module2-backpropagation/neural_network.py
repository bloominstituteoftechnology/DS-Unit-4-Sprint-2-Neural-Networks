import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class SingleHiddenLayerNN:

    def __init__(self, input_layer_size=3, hidden_layer_size=4, output_layer_size=1, lr=0.01, momentum=0.9, iters=1000, seed=42):
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
        self.Wh1 = self.__init_weights(
            self.input_layer_size, self.hidden_layer_size)

        # Hidden layer size x Output layer size Matrix Array for Hidden to Output
        self.Wo = np.random.randn(
            self.hidden_layer_size, self.output_layer_size)

        # Let's not forget the bias terms which allows to shift neuron's activation outputs
        self.Bh1, self.Bo = __init_bias()

        # Stores loss per epoch
        self.loss = []

    def __set_seed(self, seed):
        np.random.seed(seed)

    def __sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def __sigmoid_prime(self, Z):
        sx = self.__sigmoid(Z)
        return sx * (1 - sx)

    def __tanh(self, Z):
        return np.tanh(Z)

    def __tanh_prime(self, Z):
        return 1 - self.__tanh(Z)**2

    def __relu(self, Z):
        return np.maximum(0, Z)

    def __relu_prime(self, Z):
        pass

    def __init_weights(self, input_size, output_size, init_type='custom'):
        if init_type == 'uniform':
            W = np.random.random(input_size, output_size) / np.sqrt(input_size)
        elif init_type == 'normal':
            W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        elif init_type == 'random_times_decimal':
            W = (np.random.randn(input_size, output_size)
                 * 0.01) / np.sqrt(input_size)
        elif init_type == 'custom':
            W = (2 * np.random.random((input_size, output_size)) - 1) / \
                np.sqrt(input_size)
        else:
            raise ValueError(
                'Parameter passed is wrong type, can only be "uniform", "normal", or "custom"')

    def __init_bias(self):
        Bh1 = np.full((1, self.hidden_layer_size), 0.1)
        Bo = np.full((1, self.output_layer_size), 0.1)
        return Bh1, Bo

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
        self.Z1 = (X @ self.Wh1) + self.Bh1
        self.A1 = __sigmoid(self.Z1)
        self.Z2 = (self.A1 @ self.Wo) + self.Bo
        cache = (self.Z1, self.A1, self.Z2)
        return cache

    def __backprop(self, X, y):
        """
        Calculate & Update weight adjustments

        X - input matrix
        y - true labels
        y_hat - predicted labels
        E - MAE

        """

        # dJdW2
        # Computes size of adjustments from hidden -> output
        self.D2 = self.E * self.__sigmoid_prime(self.Z2)
        self.dJdW2 = self.A1.T @ self.D2

        # dJdW1 (Z1) error
        # Computes size of adjustments from input -> hidden
        self.D1 = (self.D2 @ self.Wo.T) * self.sigmoidPrime(self.Z1)
        self.dJdW1 = X.T @ self.D1

        # dJdb1
        self.dJdb1 =

        # dJdb2
        self.dJdb2 =

        # Update weights and biases
        self.Wh1 +=
        self.Bh1 +=
        self.Wo += self.lr * self.dJdW2
        self.Bo

        cache = (self.Wh1, self.Bh1, self.Wo, self.Bo)
        return cache

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

        for _ in range(self.iters):
            # Get predictions
            self.__feed_forward(X)

            # Compute Error in output
            y_hat = __sigmoid(self.Z2)
            self.E = y_hat - y

            # Save the loss aka cost J
            self.loss.append(np.sum(np.square(self.E)))

            # Update weights & biases
            self.__backprop(X, y)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.loss)+1), np.array(self.loss),
                 marker='o', color='steelblue')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Error')
        plt.show()


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
