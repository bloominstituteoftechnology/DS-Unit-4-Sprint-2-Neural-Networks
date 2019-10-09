import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

    def __init__(self, input_size=2, output_size=1, lr=0.01, niter=10):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.niter = niter

        # Initialize weights once
        self.W = self.__init_weights(type_='custom')
        self.b = self.__init_bias()
        self.loss = []

    def __init_weights(self, type_='custom'):
        if type_ == 'uniform':
            W = np.random.random(self.input_size, self.output_size)
        elif type_ == 'normal':
            W = np.random.randn(self.input_size, self.output_size)
        elif type_ == 'custom':
            W = 2 * np.random.random((self.input_size, self.output_size)) - 1
        else:
            raise ValueError(
                'Parameter passed is wrong type, can only be "uniform", "normal", or "custom"')
        return W

    def __init_bias(self):
        return np.full((1, self.output_size), 0.1)

    def __sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def __sigmoid_derivative(self, X):
        sx = self.__sigmoid(X)
        return sx * (1-sx)

    def __feed_forward(self, X):
        """
        Computes Z values, which passing through sigmoid gives prediction probability
        """
        # Weighted sum of inputs / weights + bias
        Z = (X @ self.W) + self.b
        return Z

    def __backprop(self, dO, X):
        """
        Calculate weight adjustments
        """
        dW = np.dot(X.T, dO)
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

        for i in range(self.niter):
            # Forward Prop
            Z = self.__feed_forward(X)
            y_hat = self.__sigmoid(Z)

            # Calculate cost/error MAE
            E = (y - y_hat)
            self.loss.append(np.sum(E.T))

            # Calculate adjustments/gradient
            dO = E * self.__sigmoid_derivative(Z)
            dW, db = self.__backprop(dO, X)

            # Update the Weights with new gradient
            self.W += self.lr * dW
            self.b += self.lr * db

    def predict_proba(self, X):
        """Return prediction probabilites"""
        return self.__sigmoid(self.__feed_forward(X))

    def predict(self, X, threshold=0.5):
        """Return class labels"""
        pred_probas = self.predict_proba(X)
        return (pred_probas > threshold).astype('int32')

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.loss)+1), np.array(self.loss),
                 marker='o', color='steelblue')
        plt.xlabel('Epochs')
        plt.ylabel('MAE Error')
        plt.show()

    def accuracy_score(self, y_true, y_pred):
        return (y_true == y_pred).sum()


if __name__ == "__main__":

    data = {'x1': [0, 1, 0, 1],
            'x2': [0, 0, 1, 1],
            'y':  [1, 1, 1, 0]
            }

    df = pd.DataFrame.from_dict(data).astype('int')

    X = df[['x1', 'x2']].to_numpy()
    y = df['y'].to_numpy().reshape(-1, 1)

    print('*'*80)
    print('----- Shape of data ------')
    print(f'X.shape: {X.shape}')
    print(f'y.shape: {y.shape}')

    nand = Perceptron(input_size=2, output_size=1, lr=0.01, niter=1000)
    nand.fit(X, y)
    print('------- Weights & Biases -------')
    print('Weights after training')
    print(nand.W)
    print('Bias after training')
    print(nand.b)
    print('------- NAND Gate -------')
    # print(f'Predict X1: 0, X2: 0, y = {nand.predict(np.array([0, 0]))}')
    # print(f'Predict X1: 0, X2: 1, y = {nand.predict(np.array([0, 1]))}')
    # print(f'Predict X1: 1, X2: 0, y = {nand.predict(np.array([1, 0]))}')
    # print(f'Predict X1: 1, X2: 1, y = {nand.predict(np.array([1, 1]))}')
    test_X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    print(nand.predict(test_X))

    print('*'*80)
