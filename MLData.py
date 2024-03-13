import pandas as pd

class MLData:
    def __init__(self, name, file, features, target):
        self.name, self.file, self.features, self.target = name, file, features, target
        self.df = pd.read_csv(self.file, index_col=0)
        self.classes = pd.Index(list(set(self.df[target])))

    def __str__(self):
        return self.name

    def value(self, df = None):
        df = self.df if df is None else df
        return lambda i: df.iloc[i][self.features]

    def target(self, i):
        return self.df.iloc[i]["Class"]
