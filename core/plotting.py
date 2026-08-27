# core/plotting.py
import os
import matplotlib.pyplot as plt

import core.system as system
from core.system import Y_ORDER, AUX_ORDER


def _is_pressure(name: str) -> bool:
    return name.startswith("p") or name.startswith("dp") or name.startswith("dP")


def _selected(name: str) -> bool:
    plot_order = getattr(system, "PLOT_ORDER", None)
    return plot_order is None or name in plot_order


def _save_series(t, values, name, out_dir, show, derivative=False):
    pressure = _is_pressure(name)
    yplot = values * 1e-5 if pressure else values
    ylabel = name
    if pressure:
        ylabel += " [bar/s]" if derivative else " [bar]"

    fig = plt.figure()
    plt.plot(t, yplot)
    plt.grid()
    plt.xlabel("t [s]")
    plt.ylabel(ylabel)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_all_plots(res, out_dir="out/plots", show=False):
    os.makedirs(out_dir, exist_ok=True)
    t = res.t

    for j, name in enumerate(Y_ORDER):
        if _selected(name):
            _save_series(t, res.y[:, j], name, out_dir, show)

    for j, name in enumerate(AUX_ORDER):
        if _selected(name):
            _save_series(t, res.aux[:, j], name, out_dir, show)

    for j, base_name in enumerate(Y_ORDER):
        name = "d" + base_name
        if _selected(name):
            _save_series(t, res.dy[:, j], name, out_dir, show, derivative=True)
