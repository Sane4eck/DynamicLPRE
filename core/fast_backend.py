# core/fast_backend.py
import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap


def build_rk4_simulator(rhs_numba, clamp_numba, ny=4, naux=1):
    @njit(cache=True)
    def simulate(endTime, dt, stepPrint, y0, p):
        n = int(endTime / dt)
        nsave = n // stepPrint + 1

        t_out  = np.empty(nsave, dtype=np.float64)
        y_out  = np.empty((nsave, ny), dtype=np.float64)
        dy_out = np.empty((nsave, ny), dtype=np.float64)
        aux_out = np.empty((nsave, naux), dtype=np.float64)

        y  = y0.copy()
        dy = np.empty(ny, dtype=np.float64)
        aux = np.empty(naux, dtype=np.float64)

        k1 = np.empty(ny, dtype=np.float64)
        k2 = np.empty(ny, dtype=np.float64)
        k3 = np.empty(ny, dtype=np.float64)
        k4 = np.empty(ny, dtype=np.float64)
        yt = np.empty(ny, dtype=np.float64)

        t = 0.0
        k = 0

        for i in range(n):
            if i % stepPrint == 0:
                rhs_numba(t, y, p, dy, aux)
                t_out[k] = t
                for j in range(ny):
                    y_out[k, j] = y[j]
                    dy_out[k, j] = dy[j]
                for j in range(naux):
                    aux_out[k, j] = aux[j]
                k += 1

            # RK4
            rhs_numba(t, y, p, k1, aux)

            for j in range(ny):
                yt[j] = y[j] + 0.5*dt*k1[j]
            rhs_numba(t + 0.5*dt, yt, p, k2, aux)

            for j in range(ny):
                yt[j] = y[j] + 0.5*dt*k2[j]
            rhs_numba(t + 0.5*dt, yt, p, k3, aux)

            for j in range(ny):
                yt[j] = y[j] + dt*k3[j]
            rhs_numba(t + dt, yt, p, k4, aux)

            for j in range(ny):
                y[j] = y[j] + (dt/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])

            clamp_numba(y)
            t += dt

        return t_out[:k], y_out[:k], dy_out[:k], aux_out[:k]

    return simulate
