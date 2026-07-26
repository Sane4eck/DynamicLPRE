# plot.py
import os
import matplotlib.pyplot as plt

# def plot_results(result, save_dir=None, prefix="run", show=True):
#     t  = [r["time"] for r in result]
#     p0 = [r["p0"]   for r in result]
#     p1 = [r["p1"]   for r in result]
#     m01= [r["m01"]  for r in result]
#     m12= [r["m12"]  for r in result]
#     m13= [r["m13"]  for r in result]
#
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#
#     # Figure 1: pressures
#     fig1 = plt.figure()
#     plt.plot(t, p0, label="p0")
#     plt.plot(t, p1, label="p1")
#     plt.xlabel("t, s")
#     plt.ylabel("p, bar")
#     plt.grid()
#     plt.legend()
#     if save_dir:
#         fig1.savefig(os.path.join(save_dir, f"{prefix}_p.png"), dpi=200, bbox_inches="tight")
#
#     # Figure 2: flows
#     fig2 = plt.figure()
#     plt.plot(t, m01, label="m01")
#     plt.plot(t, m12, label="m12")
#     plt.plot(t, m13, label="m13")
#     plt.xlabel("t, s")
#     plt.ylabel("m")
#     plt.grid()
#     plt.legend()
#     if save_dir:
#         fig2.savefig(os.path.join(save_dir, f"{prefix}_m.png"), dpi=200, bbox_inches="tight")
#
#     if show:
#         plt.show()
#     else:
#         plt.close(fig1)
#         plt.close(fig2)

def plot_results(result, save_dir=None, prefix="run", show=True):
    """sys_two_phase"""
    t  = [r["time"] for r in result]
    pt = [r["pt"]   for r in result]
    ppin = [r["ppin"]   for r in result]
    ppout = [r["ppout"]   for r in result]
    p1 = [r["p1"]   for r in result]
    p2 = [r["p2"]   for r in result]
    mtp= [r["mtp"]  for r in result]
    mp1= [r["mp1"]  for r in result]
    m12= [r["m12"]  for r in result]
    m2t= [r["m2t"]  for r in result]
    m1t= [r["m1t"]  for r in result]

    a1t = [r["a1t"]  for r in result]
    a12 = [r["a12"]  for r in result]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Figure 1: pressures
    fig1 = plt.figure()
    plt.plot(t, pt, label="pt")
    plt.plot(t, ppin, label="ppin")
    plt.plot(t, ppout, label="ppout")
    plt.plot(t, p1, label="p1")
    plt.plot(t, p2, label="p2")
    plt.xlabel("t, s")
    plt.ylabel("p, bar")
    plt.grid()
    plt.legend()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, f"{prefix}_p.png"), dpi=200, bbox_inches="tight")

    # Figure 2: flows
    fig2 = plt.figure()
    plt.plot(t, mtp, label="mtp")
    plt.plot(t, mp1, label="mp1")
    plt.plot(t, m12, label="m12")
    plt.plot(t, m1t, label="m1t")
    plt.plot(t, m2t, label="m2t")
    plt.xlabel("t, s")
    plt.ylabel("m")
    plt.grid()
    plt.legend()
    if save_dir:
        fig2.savefig(os.path.join(save_dir, f"{prefix}_m.png"), dpi=200, bbox_inches="tight")

    # Figure 3: resist
    fig3 = plt.figure()
    plt.plot(t, a12, label="a12")
    plt.plot(t, a1t, label="a1t")
    plt.xlabel("t, s")
    plt.ylabel("a, resistance")
    plt.grid()
    plt.legend()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, f"{prefix}_a.png"), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
