import pandas as pd
from functools import reduce

class GenerativeModel:
    def __init__(self, data):
        self.data = data
        self.Q = self.getQ()

    def getQ(self):
        df, target = self.data.df, self.data.target_name
        Qframe = pd.DataFrame(df.groupby(by=[target])[target].agg("count")).rename(columns={target: "Count"})
        return pd.concat([Qframe, pd.Series(Qframe["Count"] / df.shape[0], name="Q")], axis=1)

    def cond_prob_func(self, cl, x):
        pass

    def class_prob(self, cl, x):
        cond_prob = self.cond_prob_func(cl, x)
        return cond_prob * self.Q.at[cl, "Q"]

    def predicted_class(self, x):
        cl_probs = self.Q.index.map(lambda cl: (cl, self.class_prob(cl, x)))
        return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_probs, (None, None))[0]