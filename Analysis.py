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
        self.model = NaiveBayes(train) if model == "NaiveBayes" else GDA(train)
        self.test = test

    def finalConfMat(self):
        def updatedConfMat(cf, i):
            predicted, actual = bool(self.model.predicted_class(self.test.value(i))), bool(self.test.target(i))
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
