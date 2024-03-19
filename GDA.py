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
        self.Sigma, self.components = self.covMat_Components()
        self.p = len(self.components)
        self.Sigma_det, self.Sigma_inv = np.linalg.det(self.Sigma), np.linalg.inv(self.Sigma)

    def __str__(self):
        return "GDA"

    def mu(self, cl):
        return np.array(self.data.df.loc[lambda df: df[self.data.target_name] == cl][self.data.features].mean())

    def filter(self, s, predicate):
        return pd.Series(dict(reduce(lambda l, i: l + [(i, s[i])] if predicate(s[i]) else l, s.index, [])))

    def covMat_Components(self):
        def addMatrix(m, i):
            v = self.X[i] - self.mu_dict[self.data.target(i)]
            return m + np.outer(v, v)
        M = reduce(addMatrix, range(self.n), np.zeros((self.d, self.d))) / self.n
        nonzero_comps = list(self.filter(pd.Series(range(self.d)), lambda i: np.any(M[i, :])))
        return M[np.ix_(nonzero_comps, nonzero_comps)], nonzero_comps

    def cond_prob_func(self, cl, x):
        v = np.array(x)[self.components] - self.mu_dict[cl][self.components]
        return np.exp(-0.5 * v @ self.Sigma_inv @ v) / ((2 * np.pi) ** (self.p / 2) * self.Sigma_det ** (1 / 2))


