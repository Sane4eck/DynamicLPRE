# core/runner.py
import numpy as np
from core.result import Result
from core.fast_backend import build_rk4_simulator
from core.model import rhs_impl, clamp_impl, rhs_numba, clamp_numba, NUMBA_OK

def run(model, state0, params, dt, endTime, countPoint=1000, backend="numba"):
    stepPrint = max(int(abs((state0.time - endTime)) / (dt * countPoint)), 1)
    p = params.as_tuple()
    y = model.y0_from_state(state0)
    y = y.astype(np.float64)

    # --- fast backend (Numba) ---
    if backend == "numba" and NUMBA_OK:
        sim = build_rk4_simulator(rhs_numba, clamp_numba, ny=4, naux=1)
        t, y_out, dy_out, aux_out = sim(endTime, dt, stepPrint, y, p)
        return Result(t, y_out, dy_out, aux_out)

    # --- reference backend (pure python, ті самі rhs_impl/clamp_impl) ---
    n = int(endTime / dt)
    nsave = n // stepPrint + 1

    t_out = np.empty(nsave, dtype=np.float64)
    y_out = np.empty((nsave, 4), dtype=np.float64)
    dy_out = np.empty((nsave, 4), dtype=np.float64)
    aux_out = np.empty((nsave, 1), dtype=np.float64)

    dy = np.empty(4, dtype=np.float64)
    aux = np.empty(1, dtype=np.float64)

    k = 0
    t = float(state0.time)

    for i in range(n):
        if i % stepPrint == 0:
            rhs_impl(t, y, p, dy, aux)
            t_out[k] = t
            y_out[k, :] = y
            dy_out[k, :] = dy
            aux_out[k, 0] = aux[0]
            k += 1

        # RK4 (python, але без алокацій/State)
        k1 = dy.copy()
        y2 = y + 0.5*dt*k1
        rhs_impl(t + 0.5*dt, y2, p, dy, aux)
        k2 = dy.copy()

        y3 = y + 0.5*dt*k2
        rhs_impl(t + 0.5*dt, y3, p, dy, aux)
        k3 = dy.copy()

        y4 = y + dt*k3
        rhs_impl(t + dt, y4, p, dy, aux)
        k4 = dy.copy()

        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        clamp_impl(y)
        t += dt

    return Result(t_out[:k], y_out[:k], dy_out[:k], aux_out[:k])
