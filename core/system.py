# core/system.py
import numpy as np
from dataclasses import dataclass, fields

from numba import njit

# ---------- опис моделі (імена) ----------
VAR_NAMES = ("m01", "m12", "m13", "p1")   # інтегровані змінні (y)
AUX_NAMES = ("p0",)                      # допоміжні (aux)
# dy-імена зробимо автоматично: "d"+var

# ---------- параметри ----------
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
        # автоматично — додав параметр -> сам потрапить у tuple
        return tuple(getattr(self, f.name) for f in fields(self))

# ---------- початковий стан ----------
def initial_y():
    # y = [m01,m12,m13,p1] (p1 у Па)
    return np.array([0.0, 0.0, 0.0, 1e5], dtype=np.float64)

# ---------- закони/обмеження/рівняння (Numba) ----------
@njit(cache=True)
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0)/(t2 - t1)*(t - t1)
    else:
        return valN

@njit(cache=True)
def clamp_y_inplace(y):
    # тут твої обмеження (для будь-якого ny ти їх задаєш сам)
    LIMIT = 1e6

    if y[0] > LIMIT:  y[0] = LIMIT
    if y[0] < -LIMIT: y[0] = -LIMIT

    if y[1] > LIMIT:  y[1] = LIMIT
    if y[1] < -LIMIT: y[1] = -LIMIT

    if y[2] > LIMIT:  y[2] = LIMIT
    if y[2] < -LIMIT: y[2] = -LIMIT

    if y[3] > 300e5: y[3] = 300e5
    if y[3] < 1e5:   y[3] = 1e5

@njit(cache=True)
def rhs(t, y, p, dy, aux):
    """
    y: (ny,)    інтегровані змінні
    p: tuple    Params.as_tuple()
    dy: (ny,)   похідні
    aux:(naux,) допоміжні
    """
    # розпаковка параметрів (твій порядок = порядок dataclass полів)
    rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3 = p

    clamp_y_inplace(y)

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)
    aux[0] = p0

    m01, m12, m13, p1 = y[0], y[1], y[2], y[3]

    dy[0] = (p0 - p1 - a01/rhoFu * (abs(m01)*m01)) / j01
    dy[1] = (p1 - p2 - a12/rhoFu * (abs(m12)*m12)) / j12
    dy[2] = (p1 - p3 - a13/rhoFu * (abs(m13)*m13)) / j13
    dy[3] = (m01 - m12 - m13) / C1
