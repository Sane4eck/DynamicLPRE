import numpy as np


def rk4(rhs, y0, t0, t1, dt):
    ts = np.arange(t0, t1 + dt, dt)
    ys = np.zeros((len(ts), len(y0)))
    ys[0] = y0

    for i in range(len(ts) - 1):
        t = ts[i]
        y = ys[i]

        k1 = rhs(t, y)
        k2 = rhs(t + dt/2, y + dt/2 * k1)
        k3 = rhs(t + dt/2, y + dt/2 * k2)
        k4 = rhs(t + dt,   y + dt * k3)

        ys[i + 1] = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return ts, ys
