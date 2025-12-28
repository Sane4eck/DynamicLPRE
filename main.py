# main.py
import numpy as np
from rhs import Prav
from integrators import RK4

# ===== ЧАС =====
TIME = 0.0
endTime = 1
dt = 1e-6
n = int(endTime / dt)

# ===== ПАРАМЕТРИ =====
pars = {
    "a01": 7.15292e8,
    "a12": 5.8531e12,
    "a13": 1.96184e12,
    "j01": 10115.2,
    "j12": 32228.9,
    "j13": 37852.4,
    "rho": 814.41,
    "p2": 1e5,
    "p3": 1e5
}

# ===== ПОЧАТКОВІ УМОВИ =====
# y = np.array([0.0, 0.0, 0.0])   # {m01, m12, m13}
y = np.array([1e-12, 1e-12, 1e-12])


# ===== ЗБІР РЕЗУЛЬТАТІВ =====
result = []

# ===== ОСНОВНИЙ ЦИКЛ =====
for j in range(n):

    t = TIME

    if j % 1 == 0:
        _, p1, p0 = Prav(t, y, pars)
        result.append((t, y[0], y[1], y[2], p1, p0))

    # ---- інтегрування ----
    y = RK4(Prav, y, TIME, dt, pars)

    TIME += dt

print("END")


import numpy as np
from plot_results import plot_one

# data = np.array(result
data = np.array(result)

result_dict = {
    "1 time": data[:, 0],
    "m01": data[:, 1],
    "m12": data[:, 2],
    "m13": data[:, 3],
    "p1":  data[:, 4],
    "p0":  data[:, 5],
}

# t, m01, m12, m13, p1, p0
plot_one(data, 0, 4, "p1", "red",  "p1.png")
plot_one(data, 0, 5, "p0", "orange", "p0.png")

plot_one(data, 0, 1, "m01", "green", "m01.png")
plot_one(data, 0, 2, "m12", "blue",  "m12.png")
plot_one(data, 0, 3, "m13", "purple","m13.png")
