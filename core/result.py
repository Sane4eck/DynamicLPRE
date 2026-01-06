# core/result.py
import numpy as np

class Result:
    def __init__(self, t, y, p0, dy):
        self.t = t                  # (N,)
        self.y = y                  # (N, 4) -> m01,m12,m13,p1 [Pa]
        self.p0 = p0                # (N,)   -> p0 [Pa]
        self.dy = dy                # (N, 4) -> похідні

    def to_records_bar(self):
        # сумісність з твоїм plot.py (list[dict])
        out = []
        for i in range(self.t.size):
            out.append({
                "time": float(self.t[i]),
                "m01":  float(self.y[i, 0]),
                "m12":  float(self.y[i, 1]),
                "m13":  float(self.y[i, 2]),
                "p1":   float(self.y[i, 3]) * 1e-5,  # bar
                "p0":   float(self.p0[i]) * 1e-5     # bar
            })
        return out
