import pandas as pd
import numpy as np

class ConfusionMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def df(self):
        return pd.DataFrame(self.matrix, columns=["ActualTrue", "ActualFalse"], index=["PredTrue", "PredFalse"])

    def __str__(self):
        return str(self.df())

    def error(self):
        total = np.sum(self.matrix)
        return (total - np.trace(self.matrix)) / total

    def precision(self):
        return self.matrix[0][0] / sum(self.matrix[0])

    def recall(self):
        return self.matrix[0][0] / sum(self.matrix[:, 0])

    def updated(self, predicted, actual):
        unitvector = lambda boolean: np.array([int(boolean), int(not boolean)])
        m = np.outer(unitvector(predicted), unitvector(actual))
        return ConfusionMatrix(self.matrix + m)