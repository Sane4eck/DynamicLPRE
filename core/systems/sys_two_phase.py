# core/system.py
import numpy as np
from numba import njit
from dataclasses import dataclass, fields, astuple

from core.physics import linear_law
from core.systems.sys_two_phase import *

Y_ORDER = ("ppin", "p1", "p2", "mtp", "mp1", "m12", "m1t", "m2t")  # інтегровані (y)
AUX_ORDER = ("ppout", "pt", "a12", "a1t")  # допоміжні (aux)

NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)


@dataclass(frozen=True)
class Params:
    t1_valve_1: float = 0.3
    t2_valve_1: float = 0.4
    pt: float = 15.e5
    ppin: float = 14.e5
    ppout: float = 18.e5
    p1: float = 17.e5
    pvi: float = p1
    pvo2: float = 16.9e5
    pvot: float = 16.9e5
    p2: float = 16.e5

    mtp: float = 0.1
    mp1: float = 0.1
    m12: float = 0.1
    m2t: float = 0.1
    m1t: float = 0.1

    rho: float = 814.41

    Cpin: float = 1.27465e-9
    C1: float = 1.27465e-9
    C2: float = 1.27465e-9

    atp: float = ((pt - ppin) * rho) / mtp ** 2
    ap1: float = ((ppout - p1) * rho) / mp1 ** 2
    a12: float = ((p1 - p2) * rho) / m12 ** 2
    a2t: float = ((p2 - pt) * rho) / m2t ** 2
    a1t: float = (((p1 - pt) * rho) / m1t ** 2)

    aval12: float = (((pvi - pvo2) * rho) / m12 ** 2)
    aval1t: float = (((pvi - pvot) * rho) / m1t ** 2)
    av2: float = (((pvo2 - p2) * rho) / m12 ** 2)
    avt: float = (((pvot - pt) * rho) / m1t ** 2)


    jtp: float = 10115.2
    jp1: float = 32228.9
    j12: float = 37852.4
    j2t: float = 37852.4
    j1t: float = 378520.4

    Hp: float = 4e5

    def as_tuple(self):
        # порядок = порядок полів dataclass
        return astuple(self)


# Автогенерація порядку і індексів
PARAM_ORDER = tuple(f.name for f in fields(Params))


def _declare_indices():
    for i, name in enumerate(Y_ORDER):
        globals()[f"I_{name}"] = i
    for i, name in enumerate(AUX_ORDER):
        globals()[f"A_{name}"] = i
    for _i, _name in enumerate(PARAM_ORDER):
        globals()[f"P_{_name}"] = _i


_declare_indices()
del _declare_indices


# "ppin", "p1", "p2", "mtp", "mp1", "m12", "m1t", "m2t"
def initial_y():
    return np.array([14e5, 17e5, 1e5, 0.1, 0.1, 0., 0.1, 0.], dtype=np.float64)


# @njit(cache=True)
@njit(cache=False)
def clamp_y_inplace(y, aux):
    limit_m = 1e6
    limit_p = 30e5
    limit_pEnv = 1e5
    # обмеження Витрати
    if y[I_m2t] < 0:  y[I_m2t] = 0
    # if aux[A_ppout] > limit_p:  aux[A_ppout] = limit_p
    # if y[I_ppin] > limit_p: y[I_ppin] = limit_p
    # if y[I_m12] > limit_m:  y[I_m12] = limit_m
    # if y[I_m12] < -limit_m: y[I_m12] = -limit_m
    # if y[I_m13] > limit_m:  y[I_m13] = limit_m
    # if y[I_m13] < -limit_m: y[I_m13] = -limit_m
    #
    # if y[I_p1] > limit_p: y[I_p1] = limit_p
    # if y[I_p1] < limit_pEnv: y[I_p1] = limit_pEnv


# @njit(cache=True)
@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y, aux)

    rho = p[P_rho]
    Cpin = p[P_Cpin]
    C1 = p[P_C1]
    C2 = p[P_C2]
    atp = p[P_atp]
    ap1 = p[P_ap1]
    aval12 = linear_law(t, p[P_aval12] * 1e10, p[P_aval12], p[P_t1_valve_1], p[P_t2_valve_1])
    a12 = aval12 + p[P_av2]
    aux[A_a12] = a12
    aval1t = linear_law(t, p[P_aval1t], p[P_aval1t] * 1e10, p[P_t1_valve_1], p[P_t2_valve_1])
    a1t = aval1t + p[P_avt]
    aux[A_a1t] = a1t
    a2t = p[P_a2t]
    jtp = p[P_jtp]
    jp1 = p[P_jp1]
    j12 = p[P_j12]
    j1t = p[P_j1t]
    j2t = p[P_j2t]
    pt = p[P_pt]
    aux[A_pt] = pt
    Hp = p[P_Hp]

    mtp = y[I_mtp]
    mp1 = y[I_mp1]
    m12 = y[I_m12]
    m1t = y[I_m1t]
    m2t = y[I_m2t]
    ppin = y[I_ppin]
    p1 = y[I_p1]
    p2 = y[I_p2]

    dy[I_ppin] = (mtp - mp1) / Cpin
    dy[I_mtp] = (pt - ppin - atp / rho * (abs(mtp) * mtp)) / jtp
    ppout = ppin + Hp
    aux[A_ppout] = ppout
    dy[I_p1] = (mp1 - m12 - m1t) / C1
    dy[I_mp1] = (ppout - p1 - ap1 / rho * (abs(mp1) * mp1)) / jp1
    dy[I_m12] = (p1 - p2 - a12 / rho * (abs(m12) * m12)) / j12
    dy[I_m1t] = (p1 - pt - a1t / rho * (abs(m1t) * m1t)) / j1t
    dy[I_p2] = (m12 - m2t) / C2
    dy[I_m2t] = (p2 - pt - a2t / rho * (abs(m2t) * m2t)) / j2t


__all__ = [
    "Y_ORDER", "AUX_ORDER", "NY", "NAUX",
    "PARAM_ORDER", "Params", "initial_y",
    "linear_law", "clamp_y_inplace", "rhs",
]
