# core/physics.py
import math
import numpy as np
from numba import njit


@njit(cache=True)
def ggaz(P1: float, P2: float, Mu: float, F: float, T1: float, k: float, R: float) -> float:
    """
    Аналог Mathematica:
    GGAZ[P1_,P2_,Mu_,F_,T1_,k_,R_]

    Повертає GG (витрата газу) з урахуванням критичного відношення тисків.
    """
    if P1 > 1.0e4:
        g0 = 1.0
    else:
        g0 = 980.665

    if P1 <= P2:
        return 0.0

    PI = P2 / P1
    PIcrit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    if PI < PIcrit:
        PI = PIcrit

    term = (g0 * (2.0 * k) / (R * T1 * (k - 1.0))) * (
        PI ** (2.0 / k) - PI ** ((k + 1.0) / k)
    )
    if term <= 0.0:
        return 0.0

    return P1 * Mu * F * math.sqrt(term)


@njit(cache=False)
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0) / (t2 - t1) * (t - t1)
    else:
        return valN


@njit(cache=True)
def f_valve(f1: float, f2: float, t1: float, t2: float, dt1: float, dt2: float, t: float) -> float:
    """Площа клапану від циклограми."""
    if t < t1 or t >= (t2 + dt2):
        return f1
    if t >= t1 and t < (t1 + dt1):
        if dt1 <= 0.0:
            return f2
        return f1 + (f2 - f1) * (t - t1) / dt1
    if t >= (t1 + dt1) and t < t2:
        return f2
    if t >= t2 and t < (t2 + dt2):
        if dt2 <= 0.0:
            return f1
        return f2 - (f2 - f1) * (t - t2) / dt2
    return 0.0


@njit(cache=True)
def safe_power(x: float, p: float) -> float:
    """Степінь із захистом від нуля для кореляцій теплообміну."""
    return max(x, 1.0e-30) ** p


@njit(cache=True)
def interp1_clamped(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Лінійна 1D інтерполяція з притисканням до меж таблиці."""
    n = xs.shape[0]
    if n < 2:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    dx = xs[hi] - xs[lo]
    if abs(dx) < 1.0e-30:
        return ys[lo]
    f = (x - xs[lo]) / dx
    return ys[lo] + f * (ys[hi] - ys[lo])


@njit(cache=True)
def interp2_bilinear_clamped(
    x: float,
    y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
) -> float:
    """Білінійна 2D інтерполяція для таблиць властивостей CARPET."""
    nx = xs.shape[0]
    ny = ys.shape[0]
    if nx < 2 or ny < 2:
        return values[0, 0]

    xx = min(max(x, xs[0]), xs[nx - 1])
    yy = min(max(y, ys[0]), ys[ny - 1])

    ilo = 0
    ihi = nx - 1
    while ihi - ilo > 1:
        mid = (ilo + ihi) // 2
        if xs[mid] <= xx:
            ilo = mid
        else:
            ihi = mid

    jlo = 0
    jhi = ny - 1
    while jhi - jlo > 1:
        mid = (jlo + jhi) // 2
        if ys[mid] <= yy:
            jlo = mid
        else:
            jhi = mid

    dx = xs[ihi] - xs[ilo]
    dy = ys[jhi] - ys[jlo]
    fx = 0.0 if abs(dx) < 1.0e-30 else (xx - xs[ilo]) / dx
    fy = 0.0 if abs(dy) < 1.0e-30 else (yy - ys[jlo]) / dy

    a = values[ilo, jlo] * (1.0 - fx) + values[ihi, jlo] * fx
    b = values[ilo, jhi] * (1.0 - fx) + values[ihi, jhi] * fx
    return a * (1.0 - fy) + b * fy


@njit(cache=True)
def solve_tridiagonal_inplace(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    out: np.ndarray,
    n: int,
) -> None:
    """Метод Томаса. diagonal та rhs використовуються як робочі масиви."""
    for i in range(1, n):
        pivot = diagonal[i - 1]
        if abs(pivot) < 1.0e-30:
            raise ArithmeticError("Нульовий діагональний елемент")
        q = lower[i - 1] / pivot
        diagonal[i] -= q * upper[i - 1]
        rhs[i] -= q * rhs[i - 1]

    pivot = diagonal[n - 1]
    if abs(pivot) < 1.0e-30:
        raise ArithmeticError("Нульовий діагональний елемент")
    out[n - 1] = rhs[n - 1] / pivot

    for i in range(n - 2, -1, -1):
        pivot = diagonal[i]
        if abs(pivot) < 1.0e-30:
            raise ArithmeticError("Нульовий діагональний елемент")
        out[i] = (rhs[i] - upper[i] * out[i + 1]) / pivot
