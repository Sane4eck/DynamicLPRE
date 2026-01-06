# models/hydraulic_numba.py
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
def clamp_inplace(y):
    # y = [m01, m12, m13, p1]
    LIMIT = 1e6
    if y[0] > LIMIT:  y[0] = LIMIT
    if y[0] < -LIMIT: y[0] = -LIMIT
    if y[1] > LIMIT:  y[1] = LIMIT
    if y[1] < -LIMIT: y[1] = -LIMIT
    if y[2] > LIMIT:  y[2] = LIMIT
    if y[2] < -LIMIT: y[2] = -LIMIT

    if y[3] > 300e5: y[3] = 300e5
    if y[3] < 1e5:   y[3] = 1e5

@njit
def rhs_inplace(t, y, p, dy):
    # p = (rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3)
    rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3 = p

    clamp_inplace(y)

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)

    m01, m12, m13, p1 = y[0], y[1], y[2], y[3]

    dy[0] = (p0 - p1 - a01/rhoFu * (abs(m01)*m01)) / j01
    dy[1] = (p1 - p2 - a12/rhoFu * (abs(m12)*m12)) / j12
    dy[2] = (p1 - p3 - a13/rhoFu * (abs(m13)*m13)) / j13
    dy[3] = (m01 - m12 - m13) / C1

    return p0

@njit
def simulate_rk4(endTime, dt, stepPrint, y0, p):
    n = int(endTime / dt)
    nsave = n // stepPrint + 1

    t_out  = np.empty(nsave, dtype=np.float64)
    y_out  = np.empty((nsave, 4), dtype=np.float64)
    p0_out = np.empty(nsave, dtype=np.float64)
    dy_out = np.empty((nsave, 4), dtype=np.float64)

    y = y0.copy()
    t = 0.0

    k1 = np.empty(4, dtype=np.float64)
    k2 = np.empty(4, dtype=np.float64)
    k3 = np.empty(4, dtype=np.float64)
    k4 = np.empty(4, dtype=np.float64)
    yt = np.empty(4, dtype=np.float64)

    k = 0
    for i in range(n):
        if i % stepPrint == 0:
            p0 = rhs_inplace(t, y, p, k1)  # k1 тимчасово як dy
            t_out[k] = t
            y_out[k, 0] = y[0]; y_out[k, 1] = y[1]; y_out[k, 2] = y[2]; y_out[k, 3] = y[3]
            p0_out[k] = p0
            dy_out[k, 0] = k1[0]; dy_out[k, 1] = k1[1]; dy_out[k, 2] = k1[2]; dy_out[k, 3] = k1[3]
            k += 1

        # k1
        rhs_inplace(t, y, p, k1)

        # k2
        yt[0] = y[0] + 0.5*dt*k1[0]
        yt[1] = y[1] + 0.5*dt*k1[1]
        yt[2] = y[2] + 0.5*dt*k1[2]
        yt[3] = y[3] + 0.5*dt*k1[3]
        rhs_inplace(t + 0.5*dt, yt, p, k2)

        # k3
        yt[0] = y[0] + 0.5*dt*k2[0]
        yt[1] = y[1] + 0.5*dt*k2[1]
        yt[2] = y[2] + 0.5*dt*k2[2]
        yt[3] = y[3] + 0.5*dt*k2[3]
        rhs_inplace(t + 0.5*dt, yt, p, k3)

        # k4
        yt[0] = y[0] + dt*k3[0]
        yt[1] = y[1] + dt*k3[1]
        yt[2] = y[2] + dt*k3[2]
        yt[3] = y[3] + dt*k3[3]
        rhs_inplace(t + dt, yt, p, k4)

        y[0] = y[0] + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        y[1] = y[1] + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        y[2] = y[2] + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        y[3] = y[3] + (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        t += dt

    return t_out[:k], y_out[:k], p0_out[:k], dy_out[:k]
