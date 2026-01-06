import numpy as np
from numba import njit

@njit
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0)/(t2 - t1)*(t - t1)
    else:
        return valN

@njit
def rhs(t, y, rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3):
    m01, m12, m13, p1 = y

    # clamps (як у тебе)
    LIMIT = 1e6
    if m01 > LIMIT: m01 = LIMIT
    if m01 < -LIMIT: m01 = -LIMIT
    if m12 > LIMIT: m12 = LIMIT
    if m12 < -LIMIT: m12 = -LIMIT
    if m13 > LIMIT: m13 = LIMIT
    if m13 < -LIMIT: m13 = -LIMIT
    if p1 > 300e5: p1 = 300e5
    if p1 < 1e5: p1 = 1e5

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)

    # abs без math.fabs (numba дружить)
    dm01 = (p0 - p1 - a01/rhoFu * (abs(m01)*m01)) / j01
    dm12 = (p1 - p2 - a12/rhoFu * (abs(m12)*m12)) / j12
    dm13 = (p1 - p3 - a13/rhoFu * (abs(m13)*m13)) / j13
    dp1  = (m01 - m12 - m13) / C1

    dy = np.empty(4, dtype=np.float64)
    dy[0] = dm01
    dy[1] = dm12
    dy[2] = dm13
    dy[3] = dp1
    return dy, p0

@njit
def simulate(endTime, dt, stepPrint,
             rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3,
             y0):
    n = int(endTime / dt)
    nsave = n // stepPrint + 1

    t_out  = np.empty(nsave, dtype=np.float64)
    y_out  = np.empty((nsave, 4), dtype=np.float64)
    p0_out = np.empty(nsave, dtype=np.float64)

    y = y0.copy()
    t = 0.0
    k = 0

    for i in range(n):
        if i % stepPrint == 0:
            dy_tmp, p0_tmp = rhs(t, y, rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)
            t_out[k] = t
            y_out[k, 0] = y[0]
            y_out[k, 1] = y[1]
            y_out[k, 2] = y[2]
            y_out[k, 3] = y[3]
            p0_out[k] = p0_tmp
            k += 1

        # RK4
        k1, _ = rhs(t, y, rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)
        k2, _ = rhs(t + 0.5*dt, y + 0.5*dt*k1, rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)
        k3, _ = rhs(t + 0.5*dt, y + 0.5*dt*k2, rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)
        k4, _ = rhs(t + dt,     y + dt*k3,     rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)

        y = y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t = t + dt

    return t_out[:k], y_out[:k], p0_out[:k]
