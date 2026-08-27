from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields

import numpy as np
from numba import njit

from core.carpet_io import read_input, read_properties
from core.physics import (
    interp1_clamped,
    interp2_bilinear_clamped,
    safe_power,
    solve_tridiagonal_inplace,
)


N_CELLS = max(int(os.getenv("CARPET_CELLS", "100")), 3)
N_NODES = N_CELLS + 1

# Flattened ODE state layout.
I_MASS_FLOW = 0
I_FRONT = 1
I_WALL = 2
WALL_SIZE = 4 * N_NODES
I_VAPOR_U = I_WALL + WALL_SIZE
I_VAPOR_T = I_VAPOR_U + N_CELLS
I_LIQUID_U = I_VAPOR_T + N_NODES
I_LIQUID_T = I_LIQUID_U + N_CELLS
NY = I_LIQUID_T + N_NODES


def _build_y_order() -> tuple[str, ...]:
    names = ["mass_flow", "front"]
    for k in range(4):
        names.extend(f"wall_t_r{k}_x{j:03d}" for j in range(N_NODES))
    names.extend(f"vapor_u_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(f"vapor_t_x{j:03d}" for j in range(N_NODES))
    names.extend(f"liquid_u_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(f"liquid_t_x{j:03d}" for j in range(N_NODES))
    return tuple(names)


Y_ORDER = _build_y_order()

# Algebraic/discrete output layout.
A_PRESSURE = 0
A_VOID = A_PRESSURE + N_NODES
A_QUALITY = A_VOID + N_CELLS
A_REGIME = A_QUALITY + N_CELLS
A_FRONT_TIME = A_REGIME + N_CELLS
A_FRONT_CELLS = A_FRONT_TIME + N_CELLS
A_VALVE = A_FRONT_CELLS + 1
NAUX = A_VALVE + 1


def _build_aux_order() -> tuple[str, ...]:
    names = [f"p_x{j:03d}" for j in range(N_NODES)]
    names.extend(f"void_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(f"quality_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(f"regime_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(f"front_time_x{j + 1:03d}" for j in range(N_CELLS))
    names.extend(("front_cells", "valve_factor"))
    return tuple(names)


AUX_ORDER = _build_aux_order()


@dataclass(frozen=True)
class Params:
    # Input data retained from CARPET for traceability.
    start: float
    end: float
    dt: float
    print_every: int
    heat: int
    direction: int
    test: int
    pressure: float
    inlet_celsius: float
    delta_t: float
    mass_flow0: float
    mass_flux: float
    length: float
    diameter: float
    wall_thickness: float
    metal: int
    wall_temperature: float
    environment_pressure: float
    external_htc: float
    ajo_factor: float
    ajt_factor: float
    loss_factor: float
    valve_time: float
    valve_duration: float

    # Derived geometry/hydraulic constants.
    dz: float
    dr: float
    area: float
    perimeter: float
    inlet_t: float
    inlet_rho: float
    inlet_nu: float
    inlet_u: float
    dp_theory: float
    ajo: float
    ajt: float
    ksio: float
    ksit: float
    dp: float
    p_to: float

    # Property tables, converted once to NumPy arrays for Numba.
    density_x: np.ndarray
    density_y: np.ndarray
    density_values: np.ndarray
    cp_x: np.ndarray
    cp_y: np.ndarray
    cp_values: np.ndarray
    conductivity_x: np.ndarray
    conductivity_y: np.ndarray
    conductivity_values: np.ndarray
    viscosity_x: np.ndarray
    viscosity_y: np.ndarray
    viscosity_values: np.ndarray
    surface_t: np.ndarray
    surface: np.ndarray
    saturation_p: np.ndarray
    latent_heat: np.ndarray
    saturation_t: np.ndarray
    steel: np.ndarray
    copper: np.ndarray

    def as_tuple(self):
        # Do not use dataclasses.astuple here: it deep-copies NumPy tables.
        return tuple(getattr(self, name) for name in PARAM_ORDER)

    @classmethod
    def from_files(cls, input_path, properties_path):
        c = read_input(input_path)
        tables = read_properties(properties_path)

        def a1(values):
            return np.ascontiguousarray(np.asarray(values, dtype=np.float64))

        def grid_arrays(grid):
            return a1(grid.x), a1(grid.y), np.ascontiguousarray(np.asarray(grid.values, dtype=np.float64))

        density_x, density_y, density_values = grid_arrays(tables.density)
        cp_x, cp_y, cp_values = grid_arrays(tables.heat_capacity)
        conductivity_x, conductivity_y, conductivity_values = grid_arrays(tables.conductivity)
        viscosity_x, viscosity_y, viscosity_values = grid_arrays(tables.viscosity)

        inlet_t = c.inlet_celsius + 273.15
        inlet_rho = _interp2_python(inlet_t, c.pressure * 1.0e-5, density_x, density_y, density_values)
        inlet_nu = _interp2_python(inlet_t, c.pressure * 1.0e-5, viscosity_x, viscosity_y, viscosity_values)

        area = math.pi * c.diameter ** 2 / 4.0
        perimeter = math.pi * c.diameter
        dz = c.length / N_CELLS
        dr = c.wall_thickness / 4.0
        inlet_u = c.mass_flow / (area * inlet_rho)
        re = max(inlet_u * c.diameter / inlet_nu, 1.0e-12)
        tau = 0.0395 * inlet_rho * inlet_u ** 2 * re ** -0.25
        dp_theory = tau * perimeter * c.length / area
        aj = c.length / area
        ajo = c.ajo_factor * aj
        ajt = c.ajt_factor * aj
        ksi = dp_theory * inlet_rho / c.mass_flow ** 2
        ksio = c.loss_factor * ksi
        ksit = ksi
        dp = (ksio + ksit) / inlet_rho * c.mass_flow ** 2
        p_to = c.pressure - dp + ksio / inlet_rho * c.mass_flow ** 2

        return cls(
            c.start, c.end, c.dt, c.print_every,
            c.heat, c.direction, c.test, c.pressure, c.inlet_celsius, c.delta_t,
            c.mass_flow, c.mass_flux, c.length, c.diameter, c.wall_thickness,
            c.metal, c.wall_temperature, c.environment_pressure, c.external_htc,
            c.ajo_factor, c.ajt_factor, c.loss_factor, c.valve_time, c.valve_duration,
            dz, dr, area, perimeter, inlet_t, inlet_rho, inlet_nu, inlet_u,
            dp_theory, ajo, ajt, ksio, ksit, dp, p_to,
            density_x, density_y, density_values,
            cp_x, cp_y, cp_values,
            conductivity_x, conductivity_y, conductivity_values,
            viscosity_x, viscosity_y, viscosity_values,
            a1(tables.surface_t), a1(tables.surface),
            a1(tables.saturation_p), a1(tables.latent_heat), a1(tables.saturation_t),
            np.ascontiguousarray(np.asarray(tables.steel, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(tables.copper, dtype=np.float64)),
        )


def _interp2_python(x, y, xs, ys, values):
    xx = min(max(float(x), float(xs[0])), float(xs[-1]))
    yy = min(max(float(y), float(ys[0])), float(ys[-1]))
    i = min(max(int(np.searchsorted(xs, xx)), 1), len(xs) - 1)
    j = min(max(int(np.searchsorted(ys, yy)), 1), len(ys) - 1)
    fx = (xx - xs[i - 1]) / (xs[i] - xs[i - 1])
    fy = (yy - ys[j - 1]) / (ys[j] - ys[j - 1])
    a = values[i - 1, j - 1] * (1.0 - fx) + values[i, j - 1] * fx
    b = values[i - 1, j] * (1.0 - fx) + values[i, j] * fx
    return float(a * (1.0 - fy) + b * fy)


PARAM_ORDER = tuple(f.name for f in fields(Params))
for _i, _name in enumerate(PARAM_ORDER):
    globals()[f"P_{_name}"] = _i


def initial_y_from_params(params: Params):
    y = np.zeros(NY, dtype=np.float64)
    y[I_MASS_FLOW] = params.mass_flow0
    y[I_FRONT] = 0.0
    y[I_WALL:I_WALL + WALL_SIZE] = params.wall_temperature
    y[I_VAPOR_T:I_VAPOR_T + N_NODES] = params.wall_temperature
    y[I_LIQUID_T:I_LIQUID_T + N_NODES] = params.wall_temperature
    return y


def initial_y():
    raise RuntimeError("sys_carpet: створюйте Params.from_files(...), початковий стан будується з параметрів")


@njit(cache=True)
def clamp_y_inplace(y, aux):
    # Generic integrator compatibility. sys_carpet uses simulate_system().
    return


@njit(cache=True)
def rhs(t, y, p, dy, aux):
    # Prevent accidental use of the generic ODE integrator without CARPET's
    # persistent algebraic arrays (pressure/void/quality/regimes).
    raise RuntimeError("sys_carpet requires simulate_system")


@njit(cache=True)
def _valve(t, valve_time, valve_duration):
    a = valve_time
    b = valve_time + valve_duration
    if t <= a:
        return 1.0
    if t >= b:
        return 0.0
    return (b - t) / (b - a)


@njit(cache=True)
def _material_props(temperature, metal, steel, copper):
    rows = steel if metal == 1 else copper
    return (
        interp1_clamped(temperature, rows[:, 0], rows[:, 1]),
        interp1_clamped(temperature, rows[:, 0], rows[:, 2]),
        interp1_clamped(temperature, rows[:, 0], rows[:, 3]),
    )


@njit(cache=True)
def _prop(kind, temperature, pressure_pa, p):
    pressure_bar = pressure_pa * 1.0e-5
    if kind == 0:
        return interp2_bilinear_clamped(temperature, pressure_bar,
            p[P_density_x], p[P_density_y], p[P_density_values])
    if kind == 1:
        return interp2_bilinear_clamped(temperature, pressure_bar,
            p[P_cp_x], p[P_cp_y], p[P_cp_values])
    if kind == 2:
        return interp2_bilinear_clamped(temperature, pressure_bar,
            p[P_conductivity_x], p[P_conductivity_y], p[P_conductivity_values])
    return interp2_bilinear_clamped(temperature, pressure_bar,
        p[P_viscosity_x], p[P_viscosity_y], p[P_viscosity_values])


@njit(cache=True)
def _switch_regimes(t, y, p, pressure, void, quality, regime, front_times, front_cells):
    if t > p[P_valve_time]:
        target = int(y[I_FRONT] / p[P_dz]) + 1
        if target > N_CELLS:
            target = N_CELLS
    else:
        target = 0

    while front_cells < target:
        j = front_cells
        front_cells += 1
        regime[j] = 1
        front_times[j] = t

    for j in range(front_cells):
        ts = interp1_clamped(
            pressure[j + 1] * 1.0e-5,
            p[P_saturation_p],
            p[P_saturation_t],
        )
        tl = y[I_LIQUID_T + j + 1]
        if regime[j] == 1 and tl >= ts:
            regime[j] = 2
        elif regime[j] == 2 and tl >= ts + 2.0:
            regime[j] = 3
        elif regime[j] == 3 and tl >= ts + 10.0:
            regime[j] = 4
        elif regime[j] == 4 and void[j] >= 0.35:
            regime[j] = 5
        elif regime[j] == 5 and void[j] >= 0.70:
            regime[j] = 6
        elif regime[j] == 6 and quality[j] >= 0.99:
            regime[j] = 7
    return front_cells


@njit(cache=True)
def _carpet_derivative(
    t, y, p, dy,
    pressure, void, quality, regime, front_cells,
    rho_l, rho_v, nu_l, nu_v, cp_l, cp_v, lam_l,
    sat_t, latent, sigma, ae, be, av, bv,
    lower, diagonal, upper, prhs, pressure_solution,
):
    dy[:] = 0.0
    active = front_cells

    valve = _valve(t, p[P_valve_time], p[P_valve_duration])
    mass_flow = y[I_MASS_FLOW]

    if active == 0:
        dy[I_MASS_FLOW] = (
            p[P_p_to] + p[P_dp]
            - (p[P_ksio] + p[P_ksit] * valve) / p[P_inlet_rho] * abs(mass_flow) * mass_flow
            - p[P_p_to]
        ) / (p[P_ajo] + p[P_ajt] * valve)
        return

    for j in range(active + 1):
        tl = y[I_LIQUID_T + j]
        tv = y[I_VAPOR_T + j]
        pj = pressure[j]
        rho_l[j] = _prop(0, tl, pj, p)
        rho_v[j] = _prop(0, tv, pj, p)
        nu_l[j] = _prop(3, tl, pj, p)
        nu_v[j] = _prop(3, tv, pj, p)
        cp_l[j] = _prop(1, tl, pj, p)
        cp_v[j] = _prop(1, tv, pj, p)
        lam_l[j] = _prop(2, tl, pj, p)
        sat_t[j] = interp1_clamped(pj * 1.0e-5, p[P_saturation_p], p[P_saturation_t])
        latent[j] = interp1_clamped(pj * 1.0e-5, p[P_saturation_p], p[P_latent_heat])
        sigma[j] = interp1_clamped(tl, p[P_surface_t], p[P_surface])

    area = p[P_area]
    perimeter = p[P_perimeter]
    diameter = p[P_diameter]
    dz = p[P_dz]
    direction = p[P_direction]
    g = 9.81
    flow = max(abs(mass_flow), 1.0e-4)

    for j in range(active):
        rl = 0.5 * (rho_l[j] + rho_l[j + 1])
        rv = 0.5 * (rho_v[j] + rho_v[j + 1])
        ul = y[I_LIQUID_U + j]
        uv = y[I_VAPOR_U + j]
        rj = regime[j]
        if rj < 1:
            rj = 1

        if rj == 1:
            void[j] = 0.0
            quality[j] = 0.0
        else:
            den = rl * ul - rv * uv
            if abs(den) > 1.0e-20:
                alpha = (rl * ul - flow / area) / den
            else:
                alpha = void[j]
            void[j] = min(max(alpha, 1.0e-8), 1.0 - 1.0e-8)
            quality[j] = min(max(rv * uv * area * void[j] / flow, 0.0), 1.0)

        alpha = void[j]
        nu_l_mean = max(0.5 * (nu_l[j] + nu_l[j + 1]), 1.0e-20)
        nu_v_mean = max(0.5 * (nu_v[j] + nu_v[j + 1]), 1.0e-20)
        rel = max(abs(ul) * diameter / nu_l_mean, 1.0e-12)
        rev = max(abs(uv) * diameter / nu_v_mean, 1.0e-12)
        mix = alpha * rv + (1.0 - alpha) * rl
        tau_l = 0.0395 * rl * ul * abs(ul) * rel ** -0.25
        tau_v = 0.0395 * rv * uv * abs(uv) * rev ** -0.25

        ae[j] = 1.0 / (rl * dz)
        be[j] = direction * g - tau_l * perimeter / (area * max(mix, 1.0e-20))
        av[j] = 0.0
        bv[j] = 0.0
        if rj >= 4:
            av[j] = 1.0 / (rv * dz)
            bv[j] = direction * g - tau_v * perimeter / (area * max(rv, 1.0e-20))

        pr_l = nu_l[j + 1] * rho_l[j + 1] * cp_l[j + 1] / max(lam_l[j + 1], 1.0e-20)
        htc = 0.021 * lam_l[j + 1] / diameter * rel ** 0.8 * safe_power(pr_l, 0.4)
        wall_inner = y[I_WALL + j + 1]
        superheat = max(wall_inner - sat_t[j + 1], 0.0)

        if rj == 2 or rj == 3:
            htc += (
                0.00122
                * safe_power(lam_l[j + 1], 0.79)
                * safe_power(cp_l[j + 1], 0.45)
                * safe_power(rho_l[j + 1], 0.49)
                * g ** 0.25
                / max(
                    safe_power(sigma[j + 1] * rho_l[j + 1] * nu_l[j + 1], 0.5)
                    * safe_power(latent[j + 1] * rho_v[j + 1], 0.24),
                    1.0e-30,
                )
                * safe_power(superheat, 0.24)
            )
        elif rj >= 4:
            mu_v = max(rho_v[j + 1] * nu_v[j + 1], 1.0e-20)
            htc = (
                3.566
                * safe_power(
                    lam_l[j + 1] ** 3
                    * rho_v[j + 1]
                    * max(rho_l[j + 1] - rho_v[j + 1], 1.0e-9)
                    * g
                    * latent[j + 1]
                    / max(mu_v * max(superheat, 1.0e-6), 1.0e-30),
                    0.25,
                )
                * p[P_inlet_u] ** 0.4
                * max(y[I_FRONT] - dz * (j + 0.5), 0.05) ** -0.25
            )

        q = htc * (wall_inner - y[I_LIQUID_T + j + 1])
        liquid_area = area * max(1.0 - alpha, 1.0e-8)
        dy[I_LIQUID_T + j + 1] = q * perimeter / (
            cp_l[j + 1] * liquid_area * rho_l[j + 1]
        )

        if rj >= 6:
            dy[I_VAPOR_T + j + 1] = (
                htc
                * (wall_inner - y[I_VAPOR_T + j + 1])
                * perimeter
                / (cp_v[j + 1] * area * max(alpha, 1.0e-8) * rho_v[j + 1])
            )

        for k in (1, 2):
            idx = I_WALL + k * N_NODES + j + 1
            rw, cw, kw = _material_props(
                y[idx], p[P_metal], p[P_steel], p[P_copper]
            )
            diffusivity = kw / (rw * cw)
            radius = diameter / 2.0 + p[P_dr] * k
            j_right = min(j + 2, N_CELLS)
            dy[idx] = diffusivity * (
                (
                    y[I_WALL + (k + 1) * N_NODES + j + 1]
                    - y[I_WALL + (k - 1) * N_NODES + j + 1]
                ) / (2.0 * radius * p[P_dr])
                + (
                    y[I_WALL + (k + 1) * N_NODES + j + 1]
                    - 2.0 * y[idx]
                    + y[I_WALL + (k - 1) * N_NODES + j + 1]
                ) / p[P_dr] ** 2
                + (
                    y[I_WALL + k * N_NODES + j_right]
                    - 2.0 * y[idx]
                    + y[I_WALL + k * N_NODES + j]
                ) / dz ** 2
            )

    if active == 1:
        pressure[0] = (
            p[P_p_to] + p[P_dp]
            - p[P_ksio] / p[P_inlet_rho] * abs(mass_flow) * mass_flow
        )
        pressure[1] = p[P_p_to]
    else:
        size = active + 1
        for i in range(size):
            diagonal[i] = 0.0
            prhs[i] = 0.0
        for i in range(active):
            lower[i] = 0.0
            upper[i] = 0.0

        diagonal[0] = 1.0
        upper[0] = 0.0
        prhs[0] = (
            p[P_p_to] + p[P_dp]
            - p[P_ksio] / p[P_inlet_rho] * abs(mass_flow) * mass_flow
        )

        for j in range(1, active):
            ml_left = ae[j - 1] * rho_l[j - 1] * area * (1.0 - void[j - 1])
            mv_left = av[j - 1] * rho_v[j - 1] * area * void[j - 1]
            ml_right = ae[j] * rho_l[j] * area * (1.0 - void[j])
            mv_right = av[j] * rho_v[j] * area * void[j]
            left = ml_left + mv_left
            right = ml_right + mv_right

            bl_left = be[j - 1] * rho_l[j - 1] * area * (1.0 - void[j - 1])
            bv_left = bv[j - 1] * rho_v[j - 1] * area * void[j - 1]
            bl_right = be[j] * rho_l[j] * area * (1.0 - void[j])
            bv_right = bv[j] * rho_v[j] * area * void[j]

            lower[j - 1] = left
            diagonal[j] = -left - right
            upper[j] = right
            prhs[j] = -bl_left - bv_left + bl_right + bv_right

        lower[active - 1] = 0.0
        diagonal[active] = 1.0
        prhs[active] = p[P_p_to]
        solve_tridiagonal_inplace(lower, diagonal, upper, prhs, pressure_solution, size)
        for j in range(size):
            pressure[j] = pressure_solution[j]

    mean_rho = 0.0
    for j in range(active):
        dy[I_LIQUID_U + j] = (pressure[j] - pressure[j + 1]) * ae[j] + be[j]
        dy[I_VAPOR_U + j] = (pressure[j] - pressure[j + 1]) * av[j] + bv[j]
        mean_rho += void[j] * rho_v[j] + (1.0 - void[j]) * rho_l[j]
    mean_rho /= active

    if y[I_FRONT] < p[P_length]:
        dy[I_FRONT] = mass_flow / (mean_rho * area) * (1.0 - valve)

    dy[I_MASS_FLOW] = (
        p[P_p_to] + p[P_dp]
        - (p[P_ksio] + p[P_ksit] * valve) / p[P_inlet_rho] * abs(mass_flow) * mass_flow
        - p[P_p_to]
    ) / (p[P_ajo] + p[P_ajt] * valve)


@njit(cache=True)
def _write_aux(row, t, p, pressure, void, quality, regime, front_times, front_cells):
    for j in range(N_NODES):
        row[A_PRESSURE + j] = pressure[j]
    for j in range(N_CELLS):
        row[A_VOID + j] = void[j]
        row[A_QUALITY + j] = quality[j]
        row[A_REGIME + j] = regime[j]
        row[A_FRONT_TIME + j] = front_times[j]
    row[A_FRONT_CELLS] = front_cells
    row[A_VALVE] = _valve(t, p[P_valve_time], p[P_valve_duration])


@njit(cache=True)
def simulate_system(start_time, end_time, dt, step_print, y0, p):
    duration = max(end_time - start_time, 0.0)
    nsteps = int(round(duration / dt))
    nsave = nsteps // step_print + 2

    t_out = np.empty(nsave, dtype=np.float64)
    y_out = np.empty((nsave, NY), dtype=np.float64)
    dy_out = np.empty((nsave, NY), dtype=np.float64)
    aux_out = np.empty((nsave, NAUX), dtype=np.float64)

    y = y0.copy()
    k1 = np.empty(NY, dtype=np.float64)
    k2 = np.empty(NY, dtype=np.float64)
    k3 = np.empty(NY, dtype=np.float64)
    k4 = np.empty(NY, dtype=np.float64)
    yt = np.empty(NY, dtype=np.float64)

    pressure = np.empty(N_NODES, dtype=np.float64)
    pressure[:] = p[P_p_to]
    void = np.zeros(N_CELLS, dtype=np.float64)
    quality = np.zeros(N_CELLS, dtype=np.float64)
    regime = np.zeros(N_CELLS, dtype=np.int64)
    front_times = np.empty(N_CELLS, dtype=np.float64)
    front_times[:] = np.nan
    front_cells = 0

    rho_l = np.empty(N_NODES, dtype=np.float64)
    rho_v = np.empty(N_NODES, dtype=np.float64)
    nu_l = np.empty(N_NODES, dtype=np.float64)
    nu_v = np.empty(N_NODES, dtype=np.float64)
    cp_l = np.empty(N_NODES, dtype=np.float64)
    cp_v = np.empty(N_NODES, dtype=np.float64)
    lam_l = np.empty(N_NODES, dtype=np.float64)
    sat_t = np.empty(N_NODES, dtype=np.float64)
    latent = np.empty(N_NODES, dtype=np.float64)
    sigma = np.empty(N_NODES, dtype=np.float64)
    ae = np.empty(N_CELLS, dtype=np.float64)
    be = np.empty(N_CELLS, dtype=np.float64)
    av = np.empty(N_CELLS, dtype=np.float64)
    bv = np.empty(N_CELLS, dtype=np.float64)
    lower = np.empty(N_CELLS, dtype=np.float64)
    diagonal = np.empty(N_NODES, dtype=np.float64)
    upper = np.empty(N_CELLS, dtype=np.float64)
    prhs = np.empty(N_NODES, dtype=np.float64)
    pressure_solution = np.empty(N_NODES, dtype=np.float64)

    out_i = 0
    for i in range(nsteps + 1):
        t = start_time + i * dt
        front_cells = _switch_regimes(
            t, y, p, pressure, void, quality, regime, front_times, front_cells
        )

        _carpet_derivative(
            t, y, p, k1, pressure, void, quality, regime, front_cells,
            rho_l, rho_v, nu_l, nu_v, cp_l, cp_v, lam_l,
            sat_t, latent, sigma, ae, be, av, bv,
            lower, diagonal, upper, prhs, pressure_solution,
        )

        if i % step_print == 0 or i == nsteps:
            t_out[out_i] = t
            y_out[out_i, :] = y
            dy_out[out_i, :] = k1
            _write_aux(
                aux_out[out_i], t, p, pressure, void, quality, regime,
                front_times, front_cells,
            )
            out_i += 1

        if i == nsteps:
            break

        for j in range(NY):
            yt[j] = y[j] + 0.5 * dt * k1[j]
        _carpet_derivative(
            t + 0.5 * dt, yt, p, k2, pressure, void, quality, regime, front_cells,
            rho_l, rho_v, nu_l, nu_v, cp_l, cp_v, lam_l,
            sat_t, latent, sigma, ae, be, av, bv,
            lower, diagonal, upper, prhs, pressure_solution,
        )

        for j in range(NY):
            yt[j] = y[j] + 0.5 * dt * k2[j]
        _carpet_derivative(
            t + 0.5 * dt, yt, p, k3, pressure, void, quality, regime, front_cells,
            rho_l, rho_v, nu_l, nu_v, cp_l, cp_v, lam_l,
            sat_t, latent, sigma, ae, be, av, bv,
            lower, diagonal, upper, prhs, pressure_solution,
        )

        for j in range(NY):
            yt[j] = y[j] + dt * k3[j]
        _carpet_derivative(
            t + dt, yt, p, k4, pressure, void, quality, regime, front_cells,
            rho_l, rho_v, nu_l, nu_v, cp_l, cp_v, lam_l,
            sat_t, latent, sigma, ae, be, av, bv,
            lower, diagonal, upper, prhs, pressure_solution,
        )

        for j in range(NY):
            y[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
        y[I_FRONT] = min(max(y[I_FRONT], 0.0), p[P_length])

    return t_out[:out_i], y_out[:out_i], dy_out[:out_i], aux_out[:out_i]


_sample_cells = tuple(j for j in (10, 30, 50, 70, 90) if j <= N_CELLS)
PLOT_ORDER = ("mass_flow", "front") + tuple(
    name
    for cell in _sample_cells
    for name in (
        f"liquid_t_x{cell:03d}",
        f"vapor_t_x{cell:03d}",
        f"p_x{cell:03d}",
        f"void_x{cell:03d}",
        f"quality_x{cell:03d}",
        f"regime_x{cell:03d}",
    )
)


__all__ = [
    "Y_ORDER", "AUX_ORDER", "NY", "NAUX", "PARAM_ORDER", "Params",
    "N_CELLS", "initial_y", "initial_y_from_params", "clamp_y_inplace",
    "rhs", "simulate_system", "PLOT_ORDER",
]
