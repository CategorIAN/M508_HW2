import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MnistDataloader import MnistDataloader
from NaiveBayes import NaiveBayes
from ConfusionMatrix import ConfusionMatrix
from Analysis import Analysis
from GDA import GDA


def f(i):
    if i == 1:
        M = MnistDataloader()
        NB = NaiveBayes(M.zero_one_train)
        x = M.zero_one_train.value(0)
        print(NB.predicted_class(x))
    if i == 2:
        M = ConfusionMatrix(np.array([[2, 3], [4, 5]]))
        print(M.updated(False, False))
    if i == 3:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        A = Analysis(train, test, "GDA")
        print(A.timed(A.finalConfMat))
    if i == 4:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        gda = GDA(train)
        x = train.value(0)
        print("Prob is 0: {}".format(gda.class_prob(0, x)))
        print("Prob is 1: {}".format(gda.class_prob(1, x)))





if __name__ == '__main__':
    f(3)



