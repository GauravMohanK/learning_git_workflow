import numpy as np

class LogisticRegressionScratch:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):

        m, n = X.shape

        self.W = np.zeros(n)
        self.b = 0

        for epoch in range(self.epochs):

            z = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (y_pred-y))
            db = (1/m) * np.sum(y_pred-y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:

                loss = -(1/m) * np.sum(
                    y*np.log(y_pred+1e-9)
                    +(1-y)*np.log(1-y_pred+1e-9)
                )

                print(f"Epoch {epoch}, Loss = {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)