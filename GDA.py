import pandas as pd
import numpy as np
import math
from functools import reduce
from GenerativeModel import GenerativeModel


class GDA (GenerativeModel):
    def __init__(self, data):
        '''
        :param data: MLData object to train the model on
        '''
        super().__init__(data)
        self.X = np.array(self.data.df[data.features])
        self.n, self.d = self.X.shape
        self.mu_dict = dict([(cl, self.mu(cl)) for cl in self.data.classes])
        self.Sigma, self.components = self.covMat_Components()
        self.p = len(self.components)
        self.Sigma_det, self.Sigma_inv = np.linalg.det(self.Sigma), np.linalg.inv(self.Sigma)
        print("d: {}".format(self.d))
        print("p: {}".format(self.p))
        print("Sigma: \n{}".format(self.Sigma))
        print("Det: {}".format(self.Sigma_det))

    def __str__(self):
        return "GDA"

    def transform_data(self, df):
        return df.applymap(lambda x: x / 10)

    def mu(self, cl):
        '''
        :param cl: The class
        :return: The average feature value per feature within the class
        '''
        return np.array(self.data.df.loc[lambda df: df[self.data.target_name] == cl][self.data.features].mean())

    def filter(self, s, predicate):
        return pd.Series(dict(reduce(lambda l, i: l + [(i, s[i])] if predicate(s[i]) else l, s.index, [])))

    def covMat_Components(self):
        '''
        :return: The invertible covariance matrix based on nonzero-variance components along with those components
        '''
        def addMatrix(m, i):
            #print("-----------------------------")
            #print("i: {}".format(i))
            v = self.X[i] - self.mu_dict[self.data.target(i)]
            return m + np.outer(v, v)
        M = reduce(addMatrix, range(self.n), np.zeros((self.d, self.d))) / self.n
        alpha = 1
        nonzero_comps = list(self.filter(pd.Series(range(self.d)), lambda i: np.linalg.norm(M[i, :]) > (10 ** alpha)))
        return M[np.ix_(nonzero_comps, nonzero_comps)], nonzero_comps

    def cond_prob_func(self, cl, x):
        '''
        :param cl: The class
        :param x: The data features
        :return: The probability of getting the data features given the class
        '''
        v = np.array(x)[self.components] - self.mu_dict[cl][self.components]
        return np.exp(-0.5 * v @ self.Sigma_inv @ v) / ((2 * np.pi) ** (self.p / 2) * self.Sigma_det ** (1 / 2))


