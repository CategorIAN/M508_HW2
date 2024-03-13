import pandas as pd
from functools import reduce

class NaiveBayes:
    def __init__(self, data):
        self.data = data

    def Q(self, df = None):
        df = self.data.df if df is None else df
        Qframe = pd.DataFrame(df.groupby(by=["Class"])["Class"].agg("count")).rename(columns={"Class": "Count"})
        return pd.concat([Qframe, pd.Series(Qframe["Count"] / df.shape[0], name="Q")], axis=1)

    def F(self, Qframe, df = None):
        df = self.data.df if df is None else df
        def g(j):
            Fframe = pd.DataFrame(df.groupby(by=["Class", self.data.features[j]])["Class"].agg("count")).rename(
                columns={"Class": "Count"})
            Fcol = Fframe.index.to_series().map(lambda t: (Fframe["Count"][t] + 1) /
                                                   (Qframe.at[t[0], "Count"] + len(self.data.features)))
            return pd.concat([Fframe, pd.Series(Fcol, name = "F")], axis = 1)
        return g

    def Fs(self, Qframe, df = None):
        F_func = self.F(Qframe, df)
        return dict([(j, F_func(j)) for j in range(len(self.data.features))])

    def class_prob(self, Qframe, df = None):
        Fframes = self.Fs(Qframe, df)
        def f(cl, x):
            return reduce(lambda r, j: r * Fframes[j].to_dict()["F"].get((cl, x[j]), 0),
                          range(len(self.data.features)), Qframe.at[cl, "Q"])
        return f

    def predicted_class(self, df = None):
        Qframe = self.Q(df)
        class_prob_func = self.class_prob(df)
        def f(x):
            cl_probs = Qframe.index.map(lambda cl: (cl, class_prob_func(cl, x)))
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_probs, (None, None))[0]
        return f