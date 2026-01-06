# core/result.py
class Result:
    def __init__(self, t, y, dy, aux):
        self.t = t
        self.y = y
        self.dy = dy
        self.aux = aux

    @property
    def data(self):
        # plot.py очікує list[dict] з p0/p1 в bar
        out = []
        for i in range(len(self.t)):
            out.append({
                "time": float(self.t[i]),
                "m01":  float(self.y[i, 0]),
                "m12":  float(self.y[i, 1]),
                "m13":  float(self.y[i, 2]),
                "p1":   float(self.y[i, 3]) * 1e-5,   # bar
                "p0":   float(self.aux[i, 0]) * 1e-5, # bar
                "dm01": float(self.dy[i, 0]),
                "dm12": float(self.dy[i, 1]),
                "dm13": float(self.dy[i, 2]),
                "dp1":  float(self.dy[i, 3]),
            })
        return out
