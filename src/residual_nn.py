#Hunter Weston

#Learns NumPy MLP to r(x,u) in the physical state space: x_{k+1} ≈ C(Az_k + B u_k) + r(x_k, u_k)
#Learns r: R^{n+1} -> R^n (n = 3 states), where input is [x; u]

import numpy as np

class ReplayBuffer:
    def __init__(self, dim_in, dim_out, maxlen=5000):
        self.maxlen = maxlen
        self.X = np.zeros((0, dim_in))
        self.Y = np.zeros((0, dim_out))

    def push(self, x, y):
        self.X = np.vstack([self.X, x.reshape(1,-1)])
        self.Y = np.vstack([self.Y, y.reshape(1,-1)])
        if self.X.shape[0] > self.maxlen:
            excess = self.X.shape[0] - self.maxlen
            self.X = self.X[excess:]
            self.Y = self.Y[excess:]

class TinyMLP:
    def __init__(self, dim_in, dim_hidden=16, dim_out=3, lr=1e-3):
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.1, size=(dim_hidden, dim_in))
        self.b1 = np.zeros((dim_hidden,))
        self.W2 = rng.normal(0, 0.1, size=(dim_out, dim_hidden))
        self.b2 = np.zeros((dim_out,))
        self.lr = lr

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = np.tanh(Z1)
        Y = H @ self.W2.T + self.b2
        return Y, H

    def train_batch(self, X, Y):
        # forward
        Yhat, H = self.forward(X)
        # loss: MSE
        E = Yhat - Y
        loss = np.mean(np.sum(E**2, axis=1))
        # backprop
        dYhat = 2*E / X.shape[0]
        dW2 = dYhat.T @ H
        db2 = np.sum(dYhat, axis=0)
        dH = dYhat @ self.W2
        dZ1 = dH * (1 - H**2)
        dW1 = dZ1.T @ X
        db1 = np.sum(dZ1, axis=0)
        # update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return loss

    def predict(self, X):
        Y, _ = self.forward(X)
        return Y
