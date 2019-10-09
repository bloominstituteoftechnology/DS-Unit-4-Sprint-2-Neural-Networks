from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class SingleHiddenLayerNN:

    def __init__(self, input_layer_size=3, hidden_layer_size=4, output_layer_size=1, lr=0.01, momentum=0.9, iters=1000, seed=42):
        """
        Initialize Single Hidden Layer Neural Network with input size, hidden layer size, output layer size, learning rate, momemtum, iterations, and random seed
        """
        # Setup architecture of neural network
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Setup hyperparameters
        self.lr = lr
        self.momentum = momentum
        self.iters = iters

        # Set random seed fixed
        self.__set_seed(seed=seed)

        # Initial Weights
        # Input Layer Size x Hidden layer size Matrix Array for the First Layer
        self.Wh1 = self.__init_weights(
            self.input_layer_size, self.hidden_layer_size)

        # Hidden layer size x Output layer size Matrix Array for Hidden to Output
        self.Wo = self.__init_weights(
            self.hidden_layer_size, self.output_layer_size)

        # Let's not forget the bias terms which allows to shift neuron's activation outputs
        self.Bh1, self.Bo = self.__init_bias()

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

    def __init_weights(self, input_size, output_size, init_type='normal'):
        """
        Initalize weights based on type passed int 
        """
        if init_type == 'uniform':
            W = np.random.random(input_size, output_size)
        elif init_type == 'normal':
            W = np.random.randn(input_size, output_size)
        elif init_type == 'random_times_decimal':
            W = (np.random.randn(input_size, output_size)
                 * 0.01) / np.sqrt(input_size)
        elif init_type == 'custom1':
            W = (2 * np.random.random((input_size, output_size)) - 1)
        else:
            raise ValueError(
                'Parameter passed is wrong type, can only be "uniform", "normal", or "custom"')
        return W

    def __init_bias(self):
        Bh1 = np.full((1, self.hidden_layer_size), 0.1)
        Bo = np.full((1, self.output_layer_size), 0.1)
        return Bh1, Bo

    def __feed_forward(self, X):
        """
        Calculate the NN inference using feed forward pass.

        Variable Names
        --------------
        X    - input matrix                 
        Z1   - hidden layer weighted input  
        Z2   - output layer weighted input  
        A1    - hidden layer activation

        """
        # Weighted sum of inputs and hidden layer
        self.Z1 = (X @ self.Wh1) + self.Bh1
        self.A1 = self.__sigmoid(self.Z1)
        self.Z2 = (self.A1 @ self.Wo) + self.Bo
        cache = (self.Z1, self.A1, self.Z2)
        return cache

    def __backprop(self, X):
        """
        Calculate & Update weight adjustments

        Variable Names
        --------------
        X        - input matrix             
        E        - MAE
        D2       - delta w.r.t Z2
        dJdWo    - gradient w.r.t Wo
        D1       - delta w.r.t Z1
        dJdWh1   - gradient w.r.t Wh1
        dJdbh1   - adjustments w.r.t bh1
        dJdbo    - adjustments w.r.t bo
        """

        # dJdWo (Z2) error
        # Computes size of adjustments from hidden -> output
        self.D2 = self.E * self.__sigmoid_prime(self.Z2)
        self.dJdWo = self.A1.T @ self.D2

        # dJdWh1 (Z1) error
        # Computes size of adjustments from input -> hidden
        self.D1 = (self.D2 @ self.Wo.T) * self.__sigmoid_prime(self.Z1)
        self.dJdWh1 = X.T @ self.D1

        # dJdbh1
        self.dJdbh1 = np.sum(self.D1)

        # dJdbo
        self.dJdbo = np.sum(self.D2)

        # Update weights and biases
        self.Wh1 += self.lr * self.dJdWh1
        self.Bh1 += self.lr * self.dJdbh1
        self.Wo += self.lr * self.dJdWo
        self.Bo += self.lr * self.dJdbo

        cache = (self.Wh1, self.Bh1, self.Wo, self.Bo)
        return cache

    def fit(self, X, y, verbose=True):
        """
        Fit training data

        X       - Training vectors, X.shape : [#samples, #features]
        y       - Target values, y.shape : [#samples]
        y_hat   - Predicted labels 
        """
        try:
            y.shape[1]
        except IndexError:
            y = y.reshape(-1, 1)

        for i in range(self.iters):
            # Get predictions
            self.__feed_forward(X)

            # Compute Error in output
            y_hat = self.__sigmoid(self.Z2)
            self.E = y_hat - y

            # Save the loss aka cost J
            cost_i = np.mean(np.square(self.E))
            self.loss.append(cost_i)

            # Update weights & biases
            self.__backprop(X)

            # Print loss
            if verbose:
                if (i+1 in [1, 2, 3, 4, 5]) or ((i+1) % 100 == 0):
                    print('+' + '---' * 3 + f'EPOCH {i+1}' + '---'*3 + '+')
                    print("Loss: \n", str(cost_i))

    def predict_proba(self, X):
        """Return prediction probabilites"""
        return self.__sigmoid(self.__feed_forward(X)[-1])

    def predict(self, X, threshold=0.5):
        """Return class labels"""
        pred_probas = self.predict_proba(X)
        return (pred_probas > threshold).astype('int32')

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.loss)+1), np.array(self.loss),
                 marker='o', color='steelblue')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Error')
        plt.show()


class TwoLayerNetSimple:
    """
    Simple two fully connected layers net with simple loss function loss = y - y_pred
    """

    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1, iterations=10000):
        self.iterations = iterations

        # initiailize weights
        self.W1 = np.random.rand(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.rand(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.loss = []
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def affine_forward(self, x, w, b):
        scores = x.dot(w) + b
        cache = (x, w, b)
        return scores, cache

    def affine_backward(self, dout, cache):
        x, w, b = cache
        dx = dout.dot(w.T)  # .reshape(x.shape)
        dw = x.T.dot(dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db

    def fit(self, x, y):
        for i in range(self.iterations):
            # forward prop
            a1, cache_l1 = self.affine_forward(x, self.W1, self.b1)
            out_l1 = self.sigmoid(a1)
            a2, cache_l2 = self.affine_forward(out_l1, self.W2, self.b2)

            # scoring
            out_l2 = self.sigmoid(a2)
            loss = y - out_l2
            self.loss.append(np.mean(np.sum(loss**2)))

            # back prop
            da2 = loss * self.dsigmoid(out_l2)
            dout_l2, dW2, db2 = self.affine_backward(da2, cache_l2)
            da1 = dout_l2 * self.dsigmoid(out_l1)
            _, dW1, db1 = self.affine_backward(da1, cache_l1)

            # gradient update
            self.W1 += dW1
            self.b1 += db1
            self.W2 += dW2
            self.b2 += db2
        pass

    def predict(self, x):
        x = np.array(x)
        a1, _ = self.affine_forward(x.reshape(1, -1), self.W1, self.b1)
        out1 = self.sigmoid(a1)
        scores = self.affine_forward(out1, self.W2, self.b2)
        out = self.sigmoid(scores[0])
        return out


class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(
            params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


def main():
    from sklearn.metrics import accuracy_score
    print('+'*80)
    url = 'https://raw.githubusercontent.com/ryanleeallred/datasets/master/diabetes.csv'
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
    nn = SingleHiddenLayerNN(
        input_layer_size=8, hidden_layer_size=16, output_layer_size=1, iters=1000)
    nn.fit(X_train_scaled, y_train)
    y_test_preds = nn.predict(X_test_scaled)
    y_train_preds = nn.predict(X_train_scaled)
    train_score = accuracy_score(y_train, y_train_preds)
    test_score = accuracy_score(y_test, y_test_preds)
    print('Train Accuracy is: ', train_score)
    print('Test Accuracy is: ', test_score)
    print('+'*80)


if __name__ == "__main__":
    main()
