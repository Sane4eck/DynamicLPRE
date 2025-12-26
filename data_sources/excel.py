import numpy as np
import pandas as pd


class TimeSeries1D:
    def __init__(self, t, y):
        self.t = np.asarray(t)
        self.y = np.asarray(y)

    @classmethod
    def from_excel(cls, path, t_col, y_col):
        df = pd.read_excel(path)
        return cls(df[t_col], df[y_col])

    def __call__(self, t):
        return float(np.interp(t, self.t, self.y))
