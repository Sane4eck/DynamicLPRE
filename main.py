# main.py
import numpy as np

from core.fast_sim import simulate
from core.state import State
from core.params import Params
from core.model import HydraulicModel
from core.integrators import rk4_step
from core.result import Result
import time

t0 = time.perf_counter()

# твої налаштування
endTime = 1.0
dt = 1e-9
countPoint = 1000
stepPrint = max(int(abs((0.0 - endTime))/(dt*countPoint)), 1)

# параметри (як у model.py)
rhoFu = 814.41
C1 = 1.27465e-9
a01, a12, a13 = 7.15292e8, 5.8531e12, 1.96184e12
j01, j12, j13 = 10115.2, 32228.9, 37852.4
p2, p3 = 1e5, 1e5

y0 = np.array([0.0, 0.0, 0.0, 1e5], dtype=np.float64)

t_start = time.perf_counter()

t_arr, y_arr, p0_arr = simulate(
    endTime, dt, stepPrint,
    rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3,
    y0
)

elapsed = time.perf_counter() - t_start
print(f"Execution time: {elapsed:.2f} s ({int(elapsed//60)} min {elapsed%60:.1f} s)")

# привести до твого формату result (щоб plot.py не чіпати)
result = []
for i in range(len(t_arr)):
    result.append({
        "time": float(t_arr[i]),
        "m01":  float(y_arr[i, 0]),
        "m12":  float(y_arr[i, 1]),
        "m13":  float(y_arr[i, 2]),
        "p1":   float(y_arr[i, 3]) * 1e-5,  # bar
        "p0":   float(p0_arr[i]) * 1e-5     # bar
    })

t1 = time.perf_counter()
print(f"Execution time: {t1-t0:.2f} s")

from plot import plot_results
plot_results(result)
