# core/model.py
import numpy as np

# --- Numba optional ---
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap


# =========================
# 1) ТУТ ТИ МІНЯЄШ МОДЕЛЬ
# =========================

@njit(cache=True)
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0)/(t2 - t1)*(t - t1)
    else:
        return valN


@njit(cache=True)
def clamp_locals(m01, m12, m13, p1):
    LIMIT = 1e6

    if m01 > LIMIT:  m01 = LIMIT
    if m01 < -LIMIT: m01 = -LIMIT

    if m12 > LIMIT:  m12 = LIMIT
    if m12 < -LIMIT: m12 = -LIMIT

    if m13 > LIMIT:  m13 = LIMIT
    if m13 < -LIMIT: m13 = -LIMIT

    if p1 > 300e5: p1 = 300e5
    if p1 < 1e5:   p1 = 1e5

    return m01, m12, m13, p1


@njit(cache=True)
def rhs(t, y, p, dy, aux):
    """
    y = [m01,m12,m13,p1] (p1 в Па)
    p = (rhoFu, C1, a01,a12,a13, j01,j12,j13, p2,p3)
    dy -> похідні, aux[0] -> p0 (Па)
    """
    rhoFu, C1, a01, a12, a13, j01, j12, j13, p2, p3 = p

    m01, m12, m13, p1 = y[0], y[1], y[2], y[3]
    m01, m12, m13, p1 = clamp_locals(m01, m12, m13, p1)

    p0 = linear_law(t, 2.24e5, 218.602e5, 0.0, 0.1)
    aux[0] = p0

    dy[0] = (p0 - p1 - a01/rhoFu * (abs(m01)*m01)) / j01
    dy[1] = (p1 - p2 - a12/rhoFu * (abs(m12)*m12)) / j12
    dy[2] = (p1 - p3 - a13/rhoFu * (abs(m13)*m13)) / j13
    dy[3] = (m01 - m12 - m13) / C1


@njit(cache=True)
def clamp_y_inplace(y):
    # застосовуємо ті ж обмеження до реального стану після кроку
    m01, m12, m13, p1 = clamp_locals(y[0], y[1], y[2], y[3])
    y[0] = m01; y[1] = m12; y[2] = m13; y[3] = p1


# =========================
# 2) ШВИДКИЙ СОЛВЕР RK4 (не чіпаєш при зміні моделі)
# =========================

@njit(cache=True)
def simulate_rk4(endTime, dt, stepPrint, y0, p):
    n = int(endTime / dt)
    nsave = n // stepPrint + 1

    t_out   = np.empty(nsave, dtype=np.float64)
    y_out   = np.empty((nsave, 4), dtype=np.float64)
    dy_out  = np.empty((nsave, 4), dtype=np.float64)
    aux_out = np.empty((nsave, 1), dtype=np.float64)

    y = y0.copy()
    t = 0.0

    k1 = np.empty(4, dtype=np.float64)
    k2 = np.empty(4, dtype=np.float64)
    k3 = np.empty(4, dtype=np.float64)
    k4 = np.empty(4, dtype=np.float64)
    aux = np.empty(1, dtype=np.float64)

    yt = np.empty(4, dtype=np.float64)

    k = 0
    for i in range(n):
        # k1 на поточному кроці
        rhs(t, y, p, k1, aux)

        if i % stepPrint == 0:
            t_out[k] = t
            y_out[k, 0] = y[0]; y_out[k, 1] = y[1]; y_out[k, 2] = y[2]; y_out[k, 3] = y[3]
            dy_out[k, 0] = k1[0]; dy_out[k, 1] = k1[1]; dy_out[k, 2] = k1[2]; dy_out[k, 3] = k1[3]
            aux_out[k, 0] = aux[0]
            k += 1

        # k2
        yt[0] = y[0] + 0.5*dt*k1[0]
        yt[1] = y[1] + 0.5*dt*k1[1]
        yt[2] = y[2] + 0.5*dt*k1[2]
        yt[3] = y[3] + 0.5*dt*k1[3]
        rhs(t + 0.5*dt, yt, p, k2, aux)

        # k3
        yt[0] = y[0] + 0.5*dt*k2[0]
        yt[1] = y[1] + 0.5*dt*k2[1]
        yt[2] = y[2] + 0.5*dt*k2[2]
        yt[3] = y[3] + 0.5*dt*k2[3]
        rhs(t + 0.5*dt, yt, p, k3, aux)

        # k4
        yt[0] = y[0] + dt*k3[0]
        yt[1] = y[1] + dt*k3[1]
        yt[2] = y[2] + dt*k3[2]
        yt[3] = y[3] + dt*k3[3]
        rhs(t + dt, yt, p, k4, aux)

        # update
        y[0] = y[0] + (dt/6.0)*(k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
        y[1] = y[1] + (dt/6.0)*(k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
        y[2] = y[2] + (dt/6.0)*(k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
        y[3] = y[3] + (dt/6.0)*(k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])

        clamp_y_inplace(y)
        t += dt

    return t_out[:k], y_out[:k], dy_out[:k], aux_out[:k]


# =========================
# 3) OOP-обгортка (для сумісності з State/Result)
# =========================

class HydraulicModel:
    def pack(self, s):
        return np.array([s.m01, s.m12, s.m13, s.p1], dtype=np.float64)

    def unpack(self, s, y):
        s.m01, s.m12, s.m13, s.p1 = float(y[0]), float(y[1]), float(y[2]), float(y[3])

    def simulate(self, state0, params, dt, endTime, countPoint=1000, backend="numba"):
        stepPrint = max(int(abs((state0.time - endTime)) / (dt * countPoint)), 1)
        y0 = self.pack(state0)
        p = params.as_tuple()

        if backend == "numba" and NUMBA_OK:
            return simulate_rk4(endTime, dt, stepPrint, y0, p)

        # fallback (повільно, але працює без numba)
        return simulate_rk4(endTime, dt, stepPrint, y0, p)
