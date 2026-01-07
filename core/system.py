# core/system.py
import numpy as np
from dataclasses import dataclass

from numba import njit

# -------------------------
# 1) Порядок змінних/aux
# -------------------------
Y_ORDER = ("m01", "m12", "m13")  # інтегровані (y)
AUX_ORDER = ("p0", "p1",)  # допоміжні (aux)

NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)

# індекси y
I_m01 = 0
I_m12 = 1
I_m13 = 2
# I_p1  = 3

# індекси aux
A_p0 = 0
A_p1 = 1

# -------------------------
# 2) Параметри (p-tuple)
# -------------------------
PARAM_ORDER = (
    "rhoFu", "C1",
    "a01", "a12", "a13",
    "j01", "j12", "j13",
    "p2", "p3",
)

for _i, _name in enumerate(PARAM_ORDER):
    globals()[f"P_{_name}"] = _i
del _i, _name


@dataclass(frozen=True)
class Params:
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
        return (
            self.rhoFu, self.C1,
            self.a01, self.a12, self.a13,
            self.j01, self.j12, self.j13,
            self.p2, self.p3,
        )


def initial_y():
    return np.array([0.0, 0.0, 0.0], dtype=np.float64)


# -------------------------
# 3) Модель: закони/обмеження/рівняння (Numba)
# -------------------------
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
    if y[I_m01] > limit_m:  y[I_m01] = limit_m
    if y[I_m01] < -limit_m: y[I_m01] = -limit_m
    if y[I_m12] > limit_m:  y[I_m12] = limit_m
    if y[I_m12] < -limit_m: y[I_m12] = -limit_m
    if y[I_m13] > limit_m:  y[I_m13] = limit_m
    if y[I_m13] < -limit_m: y[I_m13] = -limit_m


# @njit(cache=True)
@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

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

    p1 = (((p0 - a01 / rhoFu * abs(m01) * m01) * 1 / j01) + ((p2 + a12 / rhoFu * (abs(m12) * m12)) * 1 / j12) + (
                (p3 + a13 / rhoFu * (abs(m13) * m13)) * 1 / j13)) / (1 / j01 + 1 / j12 + 1 / j13)
    aux[A_p1] = p1

    dy[I_m01] = (p0 - p1 - a01 / rhoFu * (abs(m01) * m01)) / j01
    dy[I_m12] = (p1 - p2 - a12 / rhoFu * (abs(m12) * m12)) / j12
    dy[I_m13] = (p1 - p3 - a13 / rhoFu * (abs(m13) * m13)) / j13
    # dy[I_p1]  = (m01 - m12 - m13) / C1
