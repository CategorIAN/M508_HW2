import pandas as pd
import numpy as np

class ConfusionMatrix:
    def __init__(self, matrix = None, file = None):
        '''
        :param matrix: A numpy array of the confusion matrix
        '''
        self.matrix = np.array(pd.read_csv(file, index_col=0)) if matrix is None else matrix

    def df(self):
        '''
        :return: A pandas dataframe representation of the confusion matrix
        '''
        return pd.DataFrame(self.matrix, columns=["ActualTrue", "ActualFalse"], index=["PredTrue", "PredFalse"])

    def updated(self, predicted, actual):
        '''
        :param predicted: Boolean of the predicted
        :param actual: Boolean of the actual
        :return: Confusion matrix with one entry added by one
        '''
        unitvector = lambda boolean: np.array([int(boolean), int(not boolean)])
        m = np.outer(unitvector(predicted), unitvector(actual))
        return ConfusionMatrix(self.matrix + m)

    def __str__(self):
        return str(self.df())

    def error(self):
        total = np.sum(self.matrix)
        return (total - np.trace(self.matrix)) / total

    def precision(self):
        return self.matrix[0][0] / sum(self.matrix[0])

    def recall(self):
        return self.matrix[0][0] / sum(self.matrix[:, 0])

