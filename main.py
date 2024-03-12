import os
import random
import matplotlib.pyplot as plt
from MnistDataloader import MnistDataloader
import pandas as pd

def f(i):
    if i == 1:
        M = MnistDataloader()
        s = pd.Series([4, 7, 8, 9])
        predicate = lambda v: v == 4 or v == 8
        print(M.filter(s, predicate))
    if i == 2:
        M = MnistDataloader()
        M.toDataFrame("test")

if __name__ == '__main__':
    f(2)



