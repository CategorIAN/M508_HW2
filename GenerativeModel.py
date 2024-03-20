import pandas as pd
from functools import reduce

class GenerativeModel:
    def __init__(self, data):
        '''
        :param data: MLData object to train the model on
        '''
        self.data = data
        self.Q = self.getQ()
        print("Q: \n{}".format(self.Q))

    def getQ(self):
        '''
        :return: Pandas dataframe that shows the probability of each class in the data
        '''
        df, target = self.data.df, self.data.target_name
        Qframe = pd.DataFrame(df.groupby(by=[target])[target].agg("count")).rename(columns={target: "Count"})
        return pd.concat([Qframe, pd.Series(Qframe["Count"] / df.shape[0], name="Q")], axis=1)

    def cond_prob_func(self, cl, x):
        '''
        :param cl: The class
        :param x: The data features
        :return: The probability of getting the data features given the class
        '''
        pass

    def class_prob_2(self, cl, x):
        cond_prob = self.cond_prob_func(cl, x)
        return cond_prob * self.Q.at[cl, "Q"]

    def class_prob(self, cl, x):
        '''
        :param cl: The class
        :param x: The data features
        :return: The probability of getting the data features and the class
        '''
        return self.printed([None, None], "Prob\n", self.class_prob_2, (cl, x))

    def predicted_class(self, x):
        '''
        :param x: The data features.
        :return: The class most likely having the given features
        '''
        cl_probs = self.Q.index.map(lambda cl: (cl, self.class_prob(cl, x)))
        return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, cl_probs, (None, None))[0]

    def printed(self, arg_names, result_name, function, args = ()):
        for arg_name, arg in zip(arg_names, args):
            if arg_name is None:
                pass
            else:
                print("{}: {}".format(arg_name, arg))
        result = function(*args)
        print("{} {}".format(result_name, result))
        return result