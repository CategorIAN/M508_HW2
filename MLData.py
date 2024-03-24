import pandas as pd

class MLData:
    def __init__(self, name, file, features, target_name, transform = None):
        '''
        :param name: The name of the data set
        :param file: The file location
        :param features: The features of the data set
        :param target_name: The name of the target
        :param transform: Function that transforms the features of the original data
        '''
        self.name, self.file, self.features, self.target_name = name, file, features, target_name
        df = pd.read_csv(self.file, index_col=0)
        self.df = df if transform is None else pd.concat([transform(df[features]), df[target_name]], axis=1)
        self.classes = pd.Index(list(set(self.df[target_name])))

    def __str__(self):
        return self.name

    def sample(self, i):
        '''
        :param i: Index of the data set
        :return: The data sample by its features with the given index in the data set
        '''
        return self.df.iloc[i][self.features]

    def target(self, i):
        '''
        :param i: Index of the data set
        :return: The target value of the sample with the given index in the data set
        '''
        return self.df.iloc[i][self.target_name]

    def transformed(self, f):
        '''
        :param f: Function to transform the features
        :return: MLData object that has its features transformed by f.
        '''
        return MLData(self.name, self.file, self.features, self.target_name, transform = f)
