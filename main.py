import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from MnistDataloader import MnistDataloader
from NaiveBayes import NaiveBayes


def f(i):
    if i == 1:
        M = MnistDataloader()
        NB = NaiveBayes(M.zero_one_train)
        Q = NB.Q()
        f = NB.predicted_class()
        x = NB.value()(1)
        print(f(x))
        """
        for j in range(len(NB.data.features)):
            F = F_func(j)
            print("------")
            print(j)
            print(F)
            print(F.loc[(0, x[j]), "Count"])
        j = 263
        print(j)
        print(x[j])
        print(NB.target(0))
        df = F_func(j)
        df.to_csv("df.csv")
        """

    if i == 2:
        pass

if __name__ == '__main__':
    f(1)



