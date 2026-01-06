# core/model.py
import numpy as np
from math import fabs

class HydraulicModel:

    def linear_law(self, t):
        if t <= 0.0:
            return 2.24e5
        elif t <= 0.1:
            return 2.24e5 + (218.602e5 - 2.24e5)/0.1 * t
        else:
            return 218.602e5

    # --- універсальний інтерфейс ---
    def pack(self, s):
        return np.array([s.m01, s.m12, s.m13, s.p1])

    def unpack(self, s, y):
        s.m01, s.m12, s.m13, s.p1 = y

    def rhs_vec(self, s, params):
        p0 = self.linear_law(s.time)
        s.p0 = p0

        dm01 = (p0 - s.p1 - params.a01/params.rhoFu*abs(s.m01)*s.m01) / params.j01
        dm12 = (s.p1 - params.p2 - params.a12/params.rhoFu*abs(s.m12)*s.m12) / params.j12
        dm13 = (s.p1 - params.p3 - params.a13/params.rhoFu*abs(s.m13)*s.m13) / params.j13
        dp1  = (s.m01 - s.m12 - s.m13) / params.C1

        return np.array([dm01, dm12, dm13, dp1])

    def clamp(self, s):
        LIMIT = 1e6
        s.m01 = max(min(s.m01, LIMIT), -LIMIT)
        s.m12 = max(min(s.m12, LIMIT), -LIMIT)
        s.m13 = max(min(s.m13, LIMIT), -LIMIT)
        s.p1  = max(min(s.p1, 300e5), 1e5)
