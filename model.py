def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0)/(t2 - t1)*(t - t1)
    else:
        return valN

import numpy as np
from math import fabs

rhoFu = 814.41
C1 = 1.27465e-9

a01, a12, a13 = 7.15292e8, 5.8531e12, 1.96184e12
j01, j12, j13 = 10115.2, 32228.9, 37852.4
p2, p3 = 1e5, 1e5

def prav(t, y):
    m01, m12, m13, p1 = y

    LIMIT = 1e6  # підібрати, але достатньо великий

    m01 = max(min(m01, LIMIT), -LIMIT)
    m12 = max(min(m12, LIMIT), -LIMIT)
    m13 = max(min(m13, LIMIT), -LIMIT)
    p1 = max(min(p1, 300e5), 1e5)

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)

    Dp1  = (m01 - m12 - m13) / C1
    Dm01 = (p0 - p1 - a01/rhoFu * fabs(m01)*m01) / j01
    Dm12 = (p1 - p2 - a12/rhoFu * fabs(m12)*m12) / j12
    Dm13 = (p1 - p3 - a13/rhoFu * fabs(m13)*m13) / j13

    return np.array([Dm01, Dm12, Dm13, Dp1])
