import numpy as np
import pandas as pd

def train_test_split(X, y, test_prop=0.25):
    assert len(X) == len(y)

    train_indices = np.random.choice(len(X), int((1-test_prop) * len(X)))
    test_indices = np.setdiff1d(np.arange(0,len(X)), train_indices)

    train_X = [X.iloc[i] for i in train_indices]
    test_X = [X.iloc[i] for i in test_indices]

    train_y = [y.iloc[i] for i in train_indices]
    test_y = [y.iloc[i] for i in test_indices]

    return train_X, test_X, train_y, test_y

class Perceptron(object):

    classes = []
    isTrained = False
    coef = []

    def __init__(self, coef = []):
        self.coef = coef
    
    """ Given a list of features, classify the input """
    def classify_value(self, features):
        return 0
    
    def classify_all(self, X):
        pred = []
        for obs in X:
            pred.append(self.classify_value(obs))
        return pred
    
    """ Train a perceptron model on the given features and classes """
    def train(self, X: pd.DataFrame,y):
        # Resolve classes to numbers
        return 0
    
    """ Find the number of observations that are correctly classified """
    def test(self, X, y):
        # X and y must be the same length
        assert len(X) == len(y)

        n = 0
        for i in range(X):
            n += self.classify_value(X[i]) == y[i]

        return n / len(X)

