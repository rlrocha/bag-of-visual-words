import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def softmax(x):
    
    max = np.max(x, axis=1, keepdims=True) # Returns max of each row and keeps same dims
    e_x = np.exp(x - max) # Subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True) # Returns sum of each row and keeps same dims
    f_x = e_x / sum
    
    return f_x

class ELMClassifier(BaseEstimator, TransformerMixin):

    def __init__(self, L, random_state=None):
        
        self.L = L # number of hidden neurons
        self.random_state = random_state # random state

    def fit(self, X, y=None):

        M = np.size(X, axis=0) # Number of examples
        N = np.size(X, axis=1) # Number of features

        np.random.seed(seed=self.random_state) # set random seed

        self.w1 = np.random.uniform(low=-1, high=1, size=(self.L, N+1)) # Weights with bias

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Xa = np.concatenate((bias, X), axis=1) # Input with bias

        S = Xa.dot(self.w1.T) # Weighted sum of hidden layer
        H = np.tanh(S) # Activation function f(x) = tanh(x), dimension M X L

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Ha = np.concatenate((bias, H), axis=1) # Activation function with bias

        # One-hot encoding
        n_classes = len(np.unique(y))
        y = np.eye(n_classes)[y]

        self.w2 = (np.linalg.pinv(Ha).dot(y)).T # w2' = pinv(Ha)*D

        return self

    def predict(self, X):

        M = np.size(X, axis=0) # Number of examples
        N = np.size(X, axis=1) # Number of features

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Xa = np.concatenate((bias, X), axis=1) # Input with bias

        S = Xa.dot(self.w1.T) # Weighted sum of hidden layer
        H = np.tanh(S) # Activation function f(x) = tanh(x), dimension M X L

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Ha = np.concatenate((bias, H), axis=1) # Activation function with bias

        y_pred = softmax(Ha.dot(self.w2.T)) # Predictions
        
        # Revert one-hot encoding
        y_pred = np.argmax(y_pred, axis=1) # axis=1 means that we want to find the index of the maximum value in each row

        return y_pred

    def predict_proba(self, X):

        M = np.size(X, axis=0) # Number of examples
        N = np.size(X, axis=1) # Number of features

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Xa = np.concatenate((bias, X), axis=1) # Input with bias

        S = Xa.dot(self.w1.T) # Weighted sum of hidden layer
        H = np.tanh(S) # Activation function f(x) = tanh(x), dimension M X L

        bias = np.ones(M).reshape(-1, 1) # Bias definition
        Ha = np.concatenate((bias, H), axis=1) # Activation function with bias

        y_pred = softmax(Ha.dot(self.w2.T)) # Predictions

        return y_pred