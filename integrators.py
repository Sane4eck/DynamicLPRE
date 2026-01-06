# integrators.py
import numpy as np

def RK4(rhs, y, t, h, pars):
    k1, _ = rhs(t, y, pars)
    k2, _ = rhs(t + 0.5*h, y + 0.5*h*k1, pars)
    k3, _ = rhs(t + 0.5*h, y + 0.5*h*k2, pars)
    k4, _ = rhs(t + h, y + h*k3, pars)

    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
