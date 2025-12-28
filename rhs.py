# rhs.py
import numpy as np

def Prav(t, y, pars):
    # y = {m01, m12, m13}
    m01, m12, m13 = y

    # -------- p0 : defLinearLaw --------
    if t < 0.0:
        p0 = 2.24e5
    elif t > 0.1:
        p0 = 218.602e5
    else:
        p0 = 2.24e5 + (218.602e5 - 2.24e5) * (t / 0.1)

    # -------- p1 : defPNod (КОРЕКТНО) --------
    num = (
            p0 * pars["a01"] / pars["rho"] * abs(m01) * m01 / pars["j01"] +
            pars["p2"] * pars["a12"] / pars["rho"] * abs(m12) * m12 / pars["j12"] +
            pars["p3"] * pars["a13"] / pars["rho"] * abs(m13) * m13 / pars["j13"]
    )

    den = (
            1.0 / pars["j01"] +
            1.0 / pars["j12"] +
            1.0 / pars["j13"]
    )

    p1 = num / den

    # -------- праві частини --------
    dm01 = (p0 - p1) * pars["a01"] / pars["rho"] * abs(m01) * m01 / pars["j01"]
    dm12 = (p1 - pars["p2"]) * pars["a12"] / pars["rho"] * abs(m12) * m12 / pars["j12"]
    dm13 = (p1 - pars["p3"]) * pars["a13"] / pars["rho"] * abs(m13) * m13 / pars["j13"]

    return np.array([dm01, dm12, dm13]), p1, p0
