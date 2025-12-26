# solver/integrator.py
import numpy as np

def rk4_step(f, t, y, dt):
    """
    Один крок класичного RK4.
    f(t, y) -> dy/dt (np.ndarray)
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt,       y + dt * k3)

    y_next = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    t_next = t + dt
    return t_next, y_next


def rk4_solve(f, t0, y0, dt, n_steps, callback=None):
    """
    Прямий явний інтегратор.
    """
    y = np.array(y0, dtype=float)
    t = float(t0)

    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))

    ts[0] = t
    ys[0] = y

    for i in range(1, n_steps + 1):
        t, y = rk4_step(f, t, y, dt)
        ts[i] = t
        ys[i] = y

        if callback is not None:
            callback(i, t, y)

    return ts, ys
