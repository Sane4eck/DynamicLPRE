# core/integrators.py
from copy import deepcopy

def rk4_step(model, state, params, dt):
    s = state

    model.rhs(s, params)
    k1 = (s.dm01, s.dm12, s.dm13, s.dp1)

    s2 = deepcopy(s)
    s2.m01 += 0.5*dt*k1[0]
    s2.m12 += 0.5*dt*k1[1]
    s2.m13 += 0.5*dt*k1[2]
    s2.p1  += 0.5*dt*k1[3]
    s2.time += 0.5*dt
    model.rhs(s2, params)
    k2 = (s2.dm01, s2.dm12, s2.dm13, s2.dp1)

    s3 = deepcopy(s)
    s3.m01 += 0.5*dt*k2[0]
    s3.m12 += 0.5*dt*k2[1]
    s3.m13 += 0.5*dt*k2[2]
    s3.p1  += 0.5*dt*k2[3]
    s3.time += 0.5*dt
    model.rhs(s3, params)
    k3 = (s3.dm01, s3.dm12, s3.dm13, s3.dp1)

    s4 = deepcopy(s)
    s4.m01 += dt*k3[0]
    s4.m12 += dt*k3[1]
    s4.m13 += dt*k3[2]
    s4.p1  += dt*k3[3]
    s4.time += dt
    model.rhs(s4, params)
    k4 = (s4.dm01, s4.dm12, s4.dm13, s4.dp1)

    s.m01 += dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    s.m12 += dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    s.m13 += dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    s.p1  += dt*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6
    s.time += dt

    model.clamp(s)
