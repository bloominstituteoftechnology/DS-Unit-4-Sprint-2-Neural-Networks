import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split

# Layer Dims
n_hidden = 10
n_in = 10

# Outputs
n_out = 10

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# non deterministic seeding
np.random.seed(0)



def main():
    diabetes = pd.read_csv(
        'https://raw.githubusercontent.com/ryanleeallred/datasets/master/diabetes.csv')
    print(diabetes.head())

    features = list(diabetes)[:-1]
    target = 'Outcome'

    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print("---------- Shape ----------")
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


if __name__ == "__main__":
    main()
