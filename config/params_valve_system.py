# config/params_valve_system.py
from models.scheme_valve_system import ValveSystemParams

def make_default_params() -> ValveSystemParams:
    """
    Тут зберемо всі числові константи з Mathematica.
    Поки – мінімальний приклад.
    """
    return ValveSystemParams(
        rho_fu=1000.0,
        p_env=1.0e5,
        v_b_fu_nom=0.01,
        p_b_fu_nom=2.0e6,
    )
