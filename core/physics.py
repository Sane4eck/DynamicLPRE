# core/physics.py
from numba import njit
import math

# @njit(cache=True)
# def examples(a: float, b: float) -> float:
#     """ приклад """
#     res = a / b
#     return res

@njit(cache=True)
def ggaz(P1: float, P2: float, Mu: float, F: float, T1: float, k: float, R: float) -> float:
    """
    Аналог Mathematica:
    GGAZ[P1_,P2_,Mu_,F_,T1_,k_,R_]

    Повертає GG (витрата газу) з урахуванням критичного відношення тисків.
    """
    # g0 якщо знадобиться використання СГС
    if P1 > 1.0e4: g0 = 1.0
    else: g0 = 980.665

    # якщо немає перепаду в правильний бік — витрата 0
    if P1 <= P2: return 0.0

    PI = P2 / P1
    PIcrit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))

    # “задушення” (choking): PI не менше критичного
    if PI < PIcrit: PI = PIcrit

    term = (g0 * (2.0 * k) / (R * T1 * (k - 1.0))) * (PI ** (2.0 / k) - PI ** ((k + 1.0) / k))
    if term <= 0.0:
        return 0.0

    return P1 * Mu * F * math.sqrt(term)


# @njit(cache=True)
# def examples(a: float, b: float) -> float:
#     """ приклад """
#     res = a / b
#     return res