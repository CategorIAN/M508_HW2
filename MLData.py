import pandas as pd

class MLData:
    def __init__(self, name, file, features, target_name):
        self.name, self.file, self.features, self.target_name = name, file, features, target_name
        self.df = pd.read_csv(self.file, index_col=0)
        self.classes = pd.Index(list(set(self.df[target_name])))

    def __str__(self):
        return self.name

    def value(self, i):
        return self.df.iloc[i][self.features]

    def target(self, i):
        return self.df.iloc[i][self.target_name]
