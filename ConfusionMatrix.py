import pandas as pd

class ConfusionMatrix:
    def __init__(self, TP, FP, FN, TN):
        self.TP, self.FP, self.FN, self.TN,  = TP, FP, FN, TN

    def __repr__(self):
        return pd.DataFrame({"ActualTrue":[self.TP, self.FN], "ActualFalse":[self.FP, self.TN]},
                            index=["PredTrue", "PredFalse"])

    def error(self):
        return (self.FP + self.FN) / (self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)