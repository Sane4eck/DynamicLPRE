import numpy as np
from rk4 import rk4
from model import prav

TIME = 0.0
endTime = 1.0
dt = 1e-4

countPoint = 1000
stepPrint = max(int(abs((TIME-endTime))/(dt*countPoint)), 1)
n = int(endTime/dt)

y = np.array([0.0, 0.0, 0.0, 1e5])

result = []

for j in range(1, n+1):
    from model import linear_law

    if (j - 1) % stepPrint == 0:
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

from plot import plot_results

plot_results(result)
