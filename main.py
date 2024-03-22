import numpy as np
from MnistDataloader import MnistDataloader
from NaiveBayes import NaiveBayes
from GDA import GDA
from ConfusionMatrix import ConfusionMatrix
from Analysis import Analysis


def f(i):
    if i == 1:
        M = MnistDataloader()
        NB = NaiveBayes(M.zero_one_train)
        x = M.zero_one_train.sample(0)
        print(NB.predicted_class(x))
    if i == 2:
        M = ConfusionMatrix(np.array([[2, 3], [4, 5]]))
        print(M.updated(False, False))
    if i == 3:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        #gda = GDA(train)
        A = Analysis(train, test, "GDA")
        print(A.timed(A.finalConfMat))
    if i == 4:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        gda = GDA(train)
        print(gda.data.df)
    if i == 5:
        M = MnistDataloader()
        train, test = M.zero_one_train, M.zero_one_test
        A = Analysis(train, test, "GDA")
        print(A.finalConfMat())


if __name__ == '__main__':
    f(5)



