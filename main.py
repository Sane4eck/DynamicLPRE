#main.py
import numpy as np
from rk4 import rk4
from model import prav
import time


TIME = 0.0
endTime = 1.0
dt = 1e-4

countPoint = 1000
stepPrint = max(int(abs((TIME-endTime))/(dt*countPoint)), 1)
n = int(endTime/dt)

y = np.array([0.0, 0.0, 0.0, 1e5])

result = []

t_start = time.perf_counter()


for j in range(1, n+1):
    from model import linear_law

    if (j - 1) % stepPrint == 0:
        if (j - 1) % 50000 == 0:
            print(f"t = {TIME:.3f} / {endTime} s")
        p0 = linear_law(TIME, 2.24e5, 218.602e5, 0.0, 0.1)
        result.append({
            "time": TIME,
            "m01": y[0],
            "m12": y[1],
            "m13": y[2],
            "p1": y[3] * 1e-5,  # bar
            "p0": p0 * 1e-5  # bar
        })

    y = rk4(prav, y, TIME, dt)
    TIME += dt

t_end = time.perf_counter()
elapsed = t_end - t_start
print(f"Execution time: {elapsed:.2f} s "
      f"({int(elapsed//60)} min {elapsed%60:.1f} s)")

from plot import plot_results

# plot_results(result)
