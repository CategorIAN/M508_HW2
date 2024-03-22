import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from NaiveBayes import NaiveBayes
from GDA import GDA
from ConfusionMatrix import ConfusionMatrix

class Analysis:
    def __init__(self, train, test, model):
        '''
        :param train: The training data to train the model on
        :param test: The test data to make predictions on
        :param model: The model to use (either 'NaiveBayes' or 'GDA')
        '''
        self.model = NaiveBayes(train) if model == "NaiveBayes" else GDA(train)
        self.test = test.transformed(self.model.transform_data)

    def finalConfMat(self):
        '''
        :return: The confusion matrix of the boolean model predictions
        '''
        def updatedConfMat(cf, i):
            predicted, actual = bool(self.model.predicted_class(self.test.sample(i))), bool(self.test.target(i))
            return cf.updated(predicted, actual)
        result = reduce(updatedConfMat, range(self.test.df.shape[0]), ConfusionMatrix(np.zeros((2, 2), dtype=int)))
        result.df().to_csv("\\".join([os.getcwd(), "Analysis", "{}ConfusionMatrix.csv".format(str(self.model))]))
        return result

    def timed(self, function, args = ()):
        print("Begin")
        start = time.time()
        result = function(*args)
        print("Time Elapsed: {} Seconds".format(time.time() - start))
        return result
