import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients

            dw = -2/n_samples * np.dot(X.T, (y-y_pred))
            db = -2/n_samples * np.sum(y - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):

        return np.mean((y_true - y_pred)**2)



        