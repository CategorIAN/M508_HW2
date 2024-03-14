import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from NaiveBayes import NaiveBayes
from ConfusionMatrix import ConfusionMatrix

class Analysis:
    def __init__(self, train, test):
        self.train = train
        self.nb = NaiveBayes(train)
        self.test = test

    def finalConfMat(self):
        pred_class_func = self.nb.predicted_class()
        def updatedConfMat(cf, i):
            print("i={}".format(i))
            predicted, actual = bool(pred_class_func(self.test.value(i))), bool(self.test.target(i))
            return cf.updated(predicted, actual)
        return reduce(updatedConfMat, range(self.test.df.shape[0]), ConfusionMatrix(np.zeros((2, 2), dtype=int)))
