import numpy as np
from MnistDataloader import MnistDataloader
from NaiveBayes import NaiveBayes
from GDA import GDA
from ConfusionMatrix import ConfusionMatrix
from Analysis import Analysis
import os
import warnings
warnings.filterwarnings('ignore')


def f(i):
    if i == 1:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        A = Analysis(train, test, "NaiveBayes")
        print(A.timed(A.finalConfMat))
    if i == 2:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        A = Analysis(train, test, "GDA")
        print(A.finalConfMat())
    if i == 3:
        m = np.array([[1, 1], [1, 1]])
        M = ConfusionMatrix(m)
        print(M)
    if i == 4:
        file = "\\".join([os.getcwd(), "Analysis", "GDAConfusionMatrix.csv"])
        M = ConfusionMatrix(None, file)
        print(M)
        print(M.error())


if __name__ == '__main__':
    f(1)



