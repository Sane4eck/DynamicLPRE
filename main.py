import numpy as np
from integrators.rk4 import rk4
from models.valve_system import ValveSystemModel
from core.state import StateVector
from core.params import ModelParams


params = ModelParams(
    # всі числові параметри
)

state = StateVector(
    names=[
        "mBFu", "mBOx", "pBFu", "pBOx",
        "p1Fu", "mB1Fu", "mP1Fu", "m1GgFu",
        "pGg", "mTurb", "nPump"
    ],
    y0=np.array([
        10.0, 10.0, 2e6, 1e5,
        1e5, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
)

model = ValveSystemModel(params)

ts, ys = rk4(model.rhs, state.y0, 0.0, 0.01, 1e-5)

print("y(end) =", ys[-1])
