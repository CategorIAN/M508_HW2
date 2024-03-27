from MnistDataloader import MnistDataloader
from Analysis import Analysis
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
        print(A.timed(A.finalConfMat))

if __name__ == '__main__':
    f(2)



