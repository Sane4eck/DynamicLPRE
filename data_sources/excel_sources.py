# data_sources/excel_sources.py

import numpy as np
import pandas as pd


class TimeSeries1D:
    """
    Одновимірна таблична функція y(t), зчитана з Excel.

    Використання:
        ts = TimeSeries1D.from_excel(...)

        y_val = ts(t)   # лінійна інтерполяція
    """

    def __init__(self, t, y):
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, t: float) -> float:
        # Лінійна інтерполяція по часу
        return float(np.interp(t, self.t, self.y))

    @classmethod
    def from_excel(cls, path: str, sheet_name: str, t_col: str, y_col: str) -> "TimeSeries1D":
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls(df[t_col].to_numpy(), df[y_col].to_numpy())
