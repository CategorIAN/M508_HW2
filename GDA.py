import pandas as pd
import numpy as np
from functools import reduce
from GenerativeModel import GenerativeModel

class GDA (GenerativeModel):
    def __init__(self, data):
        super().__init__(data)
        self.X = np.array(data.df[data.features])
        self.n, self.d = self.X.shape
        self.mu_dict = dict([(cl, self.mu(cl)) for cl in self.data.classes])
        self.Sigma = self.covMat()
        self.Sigma_det, self.Sigma_inv = np.linalg.det(self.Sigma), np.linalg.inv(self.Sigma)

    def mu(self, cl):
        return np.array(self.data.df.loc[lambda df: df[self.data.target_name] == cl][self.data.features].mean())

    def covMat(self):
        def addMatrix(m, i):
            v = self.X[i] - self.mu_dict[self.data.target(i)]
            return m + np.outer(v, v)
        return reduce(addMatrix, range(self.n), np.zeros((self.d, self.d))) / self.n

    def cond_prob_func(self, cl, x):
        v = x - self.mu_dict[cl]
        return np.exp(-0.5 * v @ self.Sigma_inv @ v) / ((2 * np.pi) ** (self.d / 2) * self.Sigma_det ** (1 / 2))


