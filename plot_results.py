# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

def plot_one(data, t_col, y_col, label, color, fname=None):
    t = data[:, t_col]
    y = data[:, y_col]

    plt.figure(figsize=(8, 4))
    plt.plot(t, y, color=color, linewidth=2)
    plt.xlabel("t")
    plt.ylabel(label)
    plt.grid(True)
    plt.title(label)

    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()
