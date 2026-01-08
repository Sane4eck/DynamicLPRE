# core/system.py
import math

import numpy as np
from numba import njit

from core.systems.sys_test import *
from dataclasses import dataclass, fields, astuple


Y_ORDER = (
    # mass flow rate FUEL
    "mf_tnk_pmp1", "mf_pmp1_ch", "mf_pmp2_1", "mf_1_5", "mf_5_2", "mf_2_ch", "mf_2_3", "mf_1_3", "mf_3_gg", "mf_2_gg",
    "mf_1_2",
    # mass flow rate Oxidizer
    "mo_tnk_pmp", "mo_pmp_gg",
    # pressure
    "pf_pmp1_in", "pf_1", "pf_2", "pf_3", "pf_5", "po_pmp_in", "pgg", "pgv", "pch",
    # volume
    "vf_sv04_ch", "vf_amp_2", "vf_2_ch", "vf_2_gg", "vo_pn02_gg",
    # other
    "omega", "mas_n2"
)

AUX_ORDER = (
    # оберти
    "rpm",
    # коефіцієнт заповнення об'єму
    "cvf_amp_2", "cvf_2_ch", "cvf_2_gg",
    # об'єми
    "vf_gg_n2",
    # втрати на клапанах
    "rf_vlv_amp", "rf_vlv_pn01a", "rf_vlv_pn01p", "rf_vlv_sv07", "rf_vlv_TV01",
    # втрати на ділянках
    "rf_1_2", "rf_1_5", "rf_5_2", "rf_2_ch", "rf_2_3", "rf_3_gg", "rf_2_sv07", "rf_2_gg", "rf_1_3",
    # інерційні втрати
    "if_1_2", "if_2_ch", "if_2_gg", "if_5_2",
    # витрати
    "mf_cool_ch", "mf_pmp1_ch_fill", "mf_pmp1_ch_jet", "mn2", "mo_pmp_trb", "mo_pmp", "mo_pmp_gg_fill",
    "mo_pmp_gg_jet", "mo_gg", "mf_gg", "mo_ch", "mf_ch", "m_gg", "m_gv", "m_ch", "mf_pmp1_leak", "mf_pmp2_leak",
    "mo_pmp_leak", "mf_pmp1_pow", "mf_pmp2_pow", "mo_pmp_pow", "mn2_st_trb",
    # напір насосів
    "hf_pmp1", "hf_pmp2", "ho_pmp",
    # тиски
    "pf_pmp1_out", "pf_pmp2_out", "pgg_var", "po_pmp_out", "pg_trb_in",
    # флаг заповнення
    "fff_sv04_ch", "fff_sv04_ch", "fff_2_gg", "fff_amp_2",
    "fff_2_ch", "fff_2_gg", "ffo_pmp_gg",
    # флаги горіння
    "fb_gg", "fstp_gg",
    # флаг відкриття клапану
    "fvlvf_sv04", "fvlvf_sv04", "fvlvo_pn02",
    # eta pumps and turbines
    "etaf_pmp1", "etaf_pmp2", "etao_pmp", "eta_trb", "eta_st_trb",
    # torque pmp and trb
    "torqf_pmp1", "torqf_pmp2", "torqo_pmp", "torq_trb", "torq_st_trb",
    # Lad turbine
    "lad_trb", "lad_st_trb",
    # other
    "mode", "km_gg", "R_gg", "T_gg", "k_gg", "R_gv", "T_gv", "k_gv", "km_ch", "R_ch", "T_ch", "k_ch", "PI_trb",
    "PI_st_trb", "u_trb", "u_st_trb", "cad_trb", "cad_st_trb",
)

PARAM_ORDER = (
    "pf_tnk", "po_tnk", "rf_amp_2",
    "rhoFu", "C1",
    "a01", "a12", "a13",
    "j01", "j12", "j13",
    "p2", "p3",
)
NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)

@dataclass(frozen=True)
class Params:
    pf_tnk: float = 0.0
    po_tnk: float = 0.0
    rf_amp_2: float = 0.0

    rhoFu: float = 814.41
    C1: float = 1.27465e-9

    a01: float = 7.15292e8
    a12: float = 5.8531e12
    a13: float = 1.96184e12

    j01: float = 10115.2
    j12: float = 32228.9
    j13: float = 37852.4

    p2: float = 1e5
    p3: float = 1e5

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
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0) / (t2 - t1) * (t - t1)
    else:
        return valN


# @njit(cache=True)
@njit(cache=False)
def clamp_y_inplace(y):
    limit_m = 1e6
    limit_p = 300e5
    limit_pEnv = 1e5
    # обмеження Витрати
    if y[I_mas_n2] > limit_m:  y[I_m01] = limit_m
    if y[I_m01] < -limit_m: y[I_m01] = -limit_m
    if y[I_m12] > limit_m:  y[I_m12] = limit_m
    if y[I_m12] < -limit_m: y[I_m12] = -limit_m
    if y[I_m13] > limit_m:  y[I_m13] = limit_m
    if y[I_m13] < -limit_m: y[I_m13] = -limit_m

    if y[I_p1] > limit_p: y[I_p1] = limit_p
    if y[I_p1] < limit_pEnv: y[I_p1] = limit_pEnv


# @njit(cache=True)
@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

    rpm = y[i_omega]/(2*math.pi/60)


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
