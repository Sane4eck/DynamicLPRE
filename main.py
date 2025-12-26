# main.py
from solver.integrator import rk4_solve
from config.params_valve_system import make_default_params
from models.scheme_valve_system import ValveSystemScheme

def main():
    params = make_default_params()
    scheme = ValveSystemScheme(params)

    t0 = 0.0
    t_end = 0.01    # просто невеликий інтервал для тесту
    dt = 1e-5       # пізніше візьмемо з твоєї WM-програми
    n_steps = int((t_end - t0) / dt)

    y0 = scheme.initial_state()

    ts, ys = rk4_solve(
        f=scheme.rhs,
        t0=t0,
        y0=y0,
        dt=dt,
        n_steps=n_steps,
    )

    print("ts shape:", ts.shape)
    print("ys shape:", ys.shape)
    print("y(0)   =", ys[0])
    print("y(end) =", ys[-1])

if __name__ == "__main__":
    main()
