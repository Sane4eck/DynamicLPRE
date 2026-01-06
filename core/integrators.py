# core/integrators.py
import numpy as np

def rk4_step(model, state, params, dt):
    y0 = model.pack(state)

    k1 = model.rhs_vec(state, params)

    y = y0 + 0.5*dt*k1
    model.unpack(state, y)
    k2 = model.rhs_vec(state, params)

    y = y0 + 0.5*dt*k2
    model.unpack(state, y)
    k3 = model.rhs_vec(state, params)

    y = y0 + dt*k3
    model.unpack(state, y)
    k4 = model.rhs_vec(state, params)

    y_new = y0 + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    model.unpack(state, y_new)
    state.time += dt
    model.clamp(state)
