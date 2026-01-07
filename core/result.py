# core/result.py
from core.system import VAR_NAMES, AUX_NAMES

class Result:
    def __init__(self, t, y, dy, aux):
        self.t = t
        self.y = y
        self.dy = dy
        self.aux = aux

    @property
    def data(self):
        dy_names = tuple("d" + n for n in VAR_NAMES)
        out = []
        for i in range(len(self.t)):
            row = {"time": float(self.t[i])}

            # y
            for j, name in enumerate(VAR_NAMES):
                val = float(self.y[i, j])
                row[name] = val * 1e-5 if name.startswith("p") else val  # p* -> bar

            # aux
            for j, name in enumerate(AUX_NAMES):
                val = float(self.aux[i, j])
                row[name] = val * 1e-5 if name.startswith("p") else val  # p* -> bar

            # dy
            for j, name in enumerate(dy_names):
                row[name] = float(self.dy[i, j])

            out.append(row)
        return out
