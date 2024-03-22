import pandas as pd
from functools import reduce
from GenerativeModel import GenerativeModel

class NaiveBayes (GenerativeModel):
    def __init__(self, data):
        '''
        :param data: MLData object to train the model on
        '''
        super().__init__(data)
        self.Fs = dict([(j, self.F(j)) for j in range(len(self.data.features))])

    def __str__(self):
        return "NaiveBayes"

    def transform_data(self, df):
        return df

    def F(self, j):
        '''
        :param j: The index of the feature
        :return: A dictionary that maps class-feature value tuples to probability of feature value given class
        '''
        target = self.data.target_name
        grouped_df = self.data.df.groupby(by=[target, self.data.features[j]])
        Fframe = pd.DataFrame(grouped_df[target].agg("count")).rename(columns={target: "Count"})
        Ffunc = lambda t: (Fframe["Count"][t] + 1) / (self.Q.at[t[0], "Count"] + len(self.data.features))
        Fcol = Fframe.index.to_series().map(Ffunc)
        return pd.concat([Fframe, pd.Series(Fcol, name = "F")], axis = 1).to_dict()["F"]

    def cond_prob_func(self, cl, x):
        '''
        :param cl: The class
        :param x: The data features
        :return: The probability of getting the data features given the class
        '''
        return reduce(lambda r, j: r * self.Fs[j].get((cl, x[j]), 0), range(len(self.data.features)), 1)

