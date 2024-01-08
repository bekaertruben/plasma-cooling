from simulation import Simulation
from fields import Fields
from utils import lorentz_factor
import numpy as np
from matplotlib import axes, patches, lines
import matplotlib.pyplot as plt
from typing import Optional
from warnings import warn
import matplotlib.pyplot as plt
from simulation_parameters import SimulationParameters
import os

plt.rcdefaults()
plt.style.use("ggplot")

gd = [3, 30, 300]
cs = [{"syn": g, "ic": g} for g in gd]
particles = 1
temp = 0.3
prefix = "ex1"


def simulate_pic():
    names = []
    for gamma_drag in cs:
        sim = Simulation(
            N=particles,
            T=temp,
            fields=Fields.from_file(),
            parameters=SimulationParameters(
                gamma_syn=gamma_drag['syn'],
                gamma_ic=gamma_drag['ic']
            )
        )
        iterations = 1000
        x_hist = np.zeros((iterations, particles, 3))
        u_hist = np.zeros((iterations, particles, 3))
        for i, positions, velocities in sim.run(iterations, iterations):
            x_hist[i] = positions
            u_hist[i] = velocities

        name = f"P1-T0.3-Sall-syn{int(gamma_drag['syn'])}-ic{int(gamma_drag['ic'])}"
        path = f"{prefix}/{name}"
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(f"{path}/x_hist.npy", x_hist)
        np.save(f"{path}/u_hist.npy", u_hist)
        names.append(name)
    return names


def axplot_trajectory(ax: axes.Axes, X: np.ndarray, time: Optional[np.ndarray] = None, *args, **kwargs):
    if time is None:
        time = np.arange(X.shape[0])
    colors = ["k", "c", "m"]
    lines = []
    for i in range(3):
        lines.append(ax.plot(time, X[:, 0, i],
                     color=colors[i], *args, **kwargs))
    ax.set_xlabel("Iteration")
    return ax, lines


def exercise1(names: list[str], cooling_strenghts: list[dict] = cs, maxit: int = 100):
    if len(names) > 4:
        warn("len(names) > 4, all items after the 4th will not be considered")
        names = names[:4]
    fig = plt.figure(figsize=(6, 3), dpi=300)
    gs = fig.add_gridspec(1, 2, wspace=0.35, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    linestyles = ["-", "--", "-.", ":"]

    for name, style in zip(names, linestyles):
        x_hist = np.load(f"{prefix}/{name}/x_hist.npy")[:maxit]
        u_hist = np.load(f"{prefix}/{name}/u_hist.npy")[:maxit]
        print(f"gamma {name} : {lorentz_factor(u_hist[:,0])}")
        ax1, _ = axplot_trajectory(ax1, x_hist, lw=0.8, ls=style)
        ax2, _ = axplot_trajectory(ax2, u_hist, lw=0.8, ls=style)

    ax1.set_ylim(0, 160)
    ax2.set_xlim(0, maxit)

    kpatch = patches.Patch(color="k", label=r"$\cdot \hat{x}$")
    cpatch = patches.Patch(color="c", label=r"$\cdot \hat{y}$")
    mpatch = patches.Patch(color="m", label=r"$\cdot \hat{z}$")
    legendlines = [lines.Line2D([], [], color="grey", linestyle=style, label=f"$\gamma_\mathrm{{IC}} = {drag['ic']}, \gamma_\mathrm{{syn}} = {drag['syn']}$")
                   for style, drag in zip(linestyles, cooling_strenghts)]

    fig.legend(handles=[kpatch, cpatch, mpatch] +
               legendlines, fontsize="small")

    ax1.set_ylabel("Position")
    ax2.set_ylabel("Proper Velocity")

    return fig


def main():
    names = simulate_pic()
    f1 = exercise1(names)
    f1.subplots_adjust(bottom=0.15)
    plt.show()
    # f1.savefig(f"{prefix}/images/ex1-trajecetories.png", facecolor="white")


if __name__ == '__main__':
    main()
