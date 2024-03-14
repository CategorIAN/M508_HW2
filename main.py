import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MnistDataloader import MnistDataloader
from NaiveBayes import NaiveBayes
from ConfusionMatrix import ConfusionMatrix
from Analysis import Analysis


def f(i):
    if i == 1:
        M = MnistDataloader()
        NB = NaiveBayes(M.zero_one_train)
        Q = NB.Q()
        f = NB.predicted_class()
    if i == 2:
        M = ConfusionMatrix(np.array([[2, 3], [4, 5]]))
        print(M.updated(False, False))
    if i == 3:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        print(M.zero_one_test.df.head())
        A = Analysis(train, test)
        print(A.timed(A.finalConfMat))


if __name__ == '__main__':
    f(3)



