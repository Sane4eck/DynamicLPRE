# core/model.py
import numpy as np
from core.solver_numba import simulate_rk4
from core.system import initial_y

class HydraulicModel:
    def simulate(self, state0, params, dt, endTime, countPoint=1000, backend="numba"):
        stepPrint = max(int(abs((state0.time - endTime)) / (dt * countPoint)), 1)

        y0 = state0.y
        if y0 is None:
            y0 = initial_y()

        y0 = np.asarray(y0, dtype=np.float64)
        p = params.as_tuple()

        return simulate_rk4(endTime, dt, stepPrint, y0, p)
