# physics.py
import numpy as np


def linear_law(t, v0, v1, t1, t2):
    if t < t1:
        return v0
    if t > t2:
        return v1
    return v0 + (v1 - v0) * (t - t1) / (t2 - t1)


def dm_for_pnode(p, m, a, J, rho):
    return p * a / rho * abs(m) * m / J


def p_node(list_in, list_out):
    num = 0.0
    den = 0.0
    for p, m, a, J, rho in list_in:
        num += dm_for_pnode(p, m, a, J, rho)
        den += 1.0 / J
    for p, m, a, J, rho in list_out:
        num += dm_for_pnode(p, m, a, J, rho)
        den += 1.0 / J
    return num / den


def eval_aux(t, y, pars):
    """Все вспомогательные величины в одном месте"""
    m01, m12, m13 = y

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)

    p1 = p_node(
        [(p0, m01, pars["a01"], pars["j01"], pars["rho"])],
        [
            (pars["p2"], m12, pars["a12"], pars["j12"], pars["rho"]),
            (pars["p3"], m13, pars["a13"], pars["j13"], pars["rho"]),
        ],
    )

    return p0, p1


# physics.py
import numpy as np


def eval_all(t, y, pars):
    """АНАЛОГ Reap/Sow: рахує ВСЕ"""
    m01, m12, m13 = y

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)

    p1 = p_node(
        [(p0, m01, pars["a01"], pars["j01"], pars["rho"])],
        [
            (pars["p2"], m12, pars["a12"], pars["j12"], pars["rho"]),
            (pars["p3"], m13, pars["a13"], pars["j13"], pars["rho"]),
        ],
    )

    dm01 = (p0 - p1) * pars["a01"] / pars["rho"] * abs(m01) * m01 / pars["j01"]
    dm12 = (p1 - pars["p2"]) * pars["a12"] / pars["rho"] * abs(m12) * m12 / pars["j12"]
    dm13 = (p1 - pars["p3"]) * pars["a13"] / pars["rho"] * abs(m13) * m13 / pars["j13"]

    return {
        "t": t,
        "m01": m01,
        "m12": m12,
        "m13": m13,
        "p0": p0,
        "p1": p1,
        "dm01": dm01,
        "dm12": dm12,
        "dm13": dm13,
    }


def Prav(t, y, pars):
    """СИГНАТУРА СТАБІЛЬНА"""
    d = eval_all(t, y, pars)
    return np.array([d["dm01"], d["dm12"], d["dm13"]])

