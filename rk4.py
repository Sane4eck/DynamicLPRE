import numpy as np

def rk4(f, y, t, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h,     y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
