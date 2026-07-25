# core/system.py
import math

import numpy as np
from numba import njit

from core.physics import linear_law, f_valve
from core.systems.sys_test import *
from dataclasses import dataclass, fields, astuple

Y_ORDER = (
    # mass flow rate FUEL
    "mf_p_3", "mf_b_1", "mf_1_2", "mf_2_3", "mf_3_gg",
    # mass flow rate Oxidizer
    "mo_b_gg",
    # pressure
    "pf_1", "pf_2", "pf_3", "pgg",
    # volume
    # other
)

AUX_ORDER = (
    # оберти
    # коефіцієнт заповнення об'єму
    # об'єми
    # втрати на клапанах
    "rf_v1", "rf_v2", "rf_v3",
    # втрати на ділянках
    "rf_b_1", "rf_1_2", "rf_2_v1", "rf_v1_3", "rf_p_v2", "rf_v2_3", "rf_3_v3", "rf_v3_gg",
    # інерційні втрати
    "if_b_1", "if_1_2", "if_2_3", "if_p_3", "if_3_gg",
    # витрати
    "mo_b_gg",
    # напір насосів
    "hf_pmp",
    # тиски
    "pf_tnk","pf_p_out","pf_pmp_out_nom", "pgg_var", "pf_b", "po_tnk",
    # флаг заповнення
    # флаги горіння
    "fb_gg", "fstp_gg",
    # флаг відкриття клапану
    "fvf_1", "fvf_2", "fvf_3",
    # eta pumps and turbines
    # torque pmp and trb
    # Lad turbine
    # other
    "km_gg", "R_gg", "T_gg", "k_gg",
)

NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)


@dataclass(frozen=True)
class Params:

    pf_tnk: float = 2.24e5
    po_tnk: float = 3.8e5
    pf_pmp_out_nom: float = 2.18602e7



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


def initial_y():
    return np.array([0.0, 0.0, 0.0, 1e5], dtype=np.float64)


# @njit(cache=True)
@njit(cache=False)
def clamp_y_inplace(y):
    limit_m = 1e6
    limit_p = 300e5
    limit_pEnv = 1e5
    # обмеження Витрати
    if y[I_mas_n2] > limit_m:  y[I_mas_n2] = limit_m
    if y[I_mas_n2] < -limit_m: y[I_mas_n2] = -limit_m


# @njit(cache=True)
@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)
    t_bound_1=0.5
    t_bound_2=1
    pf_p_out = linear_law(t, pf_tnk, pf_pmp_out_nom, t_bound_1, t_bound_2)


    rpm = y[I_omega] / (2 * math.pi / 60)
    power = 0.1
    aux[A_cvf_amp_2] = min((y[I_vf_amp_2] / p[P_vf_amp_2]), 1) ** power
    aux[A_cvf_2_ch] = min((y[I_vf_2_ch] / p[P_vf_2_ch]), 1) ** power
    aux[A_cvf_2_gg] = min((y[I_vf_2_gg] / p[P_vf_2_gg]), 1) ** power

    aux[A_rf_vlv_amp] = 1 / (2 * f_valve[f1, f2, t1, t2, dt1, dt2, t] ** 2)

    rhoFu = p[P_rhoFu]
    C1 = p[P_C1]
    a01 = p[P_a01]
    a12 = p[P_a12]
    a13 = p[P_a13]
    j01 = p[P_j01]
    j12 = p[P_j12]
    j13 = p[P_j13]
    p2 = p[P_p2]
    p3 = p[P_p3]

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)
    aux[A_p0] = p0

    m01 = y[I_m01]
    m12 = y[I_m12]
    m13 = y[I_m13]
    p1 = y[I_p1]

    dy[I_mas_n2] = (p0 - p1 - a01 / rhoFu * (abs(m01) * m01)) / j01
    dy[I_m12] = (p1 - p2 - a12 / rhoFu * (abs(m12) * m12)) / j12
    dy[I_m13] = (p1 - p3 - a13 / rhoFu * (abs(m13) * m13)) / j13
    dy[I_p1] = (m01 - m12 - m13) / C1


__all__ = [
    "Y_ORDER", "AUX_ORDER", "NY", "NAUX",
    "PARAM_ORDER", "Params", "initial_y",
    "linear_law", "clamp_y_inplace", "rhs",
]
