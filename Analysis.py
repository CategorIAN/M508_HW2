import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes
from ConfusionMatrix import ConfusionMatrix

class Analysis:
    def __init__(self, train, test):
        self.train = train
        self.nb = NaiveBayes(train)
        self.test = test

    def finalConfMat(self, test_df):
        pred_class_func = self.nb.predicted_class()
        def updateMat(cf, i):
            x, actual = self.test.value(i), self.test.target(i)
            predicted = pred_class_func(x)
            if predicted == 1:
                if actual == 1:
                    return ConfusionMatrix(TP=cf.TP + 1, FP=cf.FP, FN=cf.FN, TN=cf.FN)
                else:
                    return ConfusionMatrix(TP=cf.TP, FP=cf.FP + 1, TN=cf.TN, FN=cf.FN)
            else:
                if actual == 1:
                    ConfusionMatrix(TP=cf.TP, FP=cf.FP, TN=cf.TN, FN=cf.TN)
                else:


