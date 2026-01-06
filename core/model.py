# core/model.py
from math import fabs

class HydraulicModel:

    def linear_law(self, t):
        if t <= 0.0:
            return 2.24e5
        elif t <= 0.1:
            return 2.24e5 + (218.602e5 - 2.24e5)/0.1 * t
        else:
            return 218.602e5

    def rhs(self, s, p):
        s.p0 = self.linear_law(s.time)

        s.dm01 = (s.p0 - s.p1 - p.a01/p.rhoFu * fabs(s.m01)*s.m01) / p.j01
        s.dm12 = (s.p1 - p.p2 - p.a12/p.rhoFu * fabs(s.m12)*s.m12) / p.j12
        s.dm13 = (s.p1 - p.p3 - p.a13/p.rhoFu * fabs(s.m13)*s.m13) / p.j13
        s.dp1  = (s.m01 - s.m12 - s.m13) / p.C1

    def clamp(self, s):
        LIMIT = 1e6
        s.m01 = max(min(s.m01, LIMIT), -LIMIT)
        s.m12 = max(min(s.m12, LIMIT), -LIMIT)
        s.m13 = max(min(s.m13, LIMIT), -LIMIT)
        s.p1  = max(min(s.p1, 300e5), 1e5)
