from wrapper_sim import Simulation
import pusher
import numpy as np
from matplotlib import figure, axes, patches, lines, colors
from typing import Optional
from warnings import warn


def diagnose_trajectories(time: np.ndarray, pos: np.ndarray, vel: np.ndarray, fields: np.ndarray) -> figure.Figure:
    linestyles = ["-", "--", "-.", ":"]
    colors = ["k", "c", "m", "y", "r", "g", "b"]

    fig = figure.Figure(figsize=(7, 5))
    gs = fig.add_gridspec(2, 2)

    axp = fig.add_subplot(gs[0, 0])
    axu = fig.add_subplot(gs[1, 0], sharex=axp)
    axE = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1], sharex=axE)

    for i, x in enumerate(np.swapaxes(pos, 0, -1)):
        for j in range(3):
            axp.plot(time, x[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    for i, u in enumerate(np.swapaxes(vel, 0, -1)):
        for j in range(3):
            axu.plot(time, u[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    fields_ci = {key: pusher.interpolate_field(pos, value)
                 for key, value in fields.items()}
    Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])
    Eci = np.swapaxes(Eci, 0, 1)

    Bci = np.swapaxes(Bci, 0, 1)

    for i, E in enumerate(np.swapaxes(Eci, 0, -1)):
        for j in range(3):
            axE.plot(time, E[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    for i, B in enumerate(np.swapaxes(Bci, 0, -1)):
        for j in range(3):
            axB.plot(time, B[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    axp.set_ylabel("Position")
    axu.set_ylabel("Velocity")
    axE.set_ylabel("Electric Field")
    axB.set_ylabel("Magnetic Field")
    axB.set_xlabel("Time")
    axE.set_xlabel("Time")
    axp.set_xlabel("Time")
    axu.set_xlabel("Time")
    axp.set_title("Particle Trajectories")
    axu.set_title("Particle Velocities")
    axE.set_title("Electric Field")
    axB.set_title("Magnetic Field")

    fig.suptitle("Time evolution (Black: x, Cyan: y, Magenta: z)")

    return fig


def axplot_trajectory(ax: axes.Axes, X: np.ndarray, time: Optional[np.ndarray] = None, *args, **kwargs):
    if time is None:
        time = np.arange(X.shape[0])
    colors = ["k", "c", "m"]
    lines = []
    for i in range(3):
        lines.append(ax.plot(time, X[0, i], color=colors[i], *args, **kwargs))
    ax.set_xlabel("Time")
    return ax, lines


def exercise1(cooling_strenghts: list[dict], iterations: int = 1000, temp: float = 0.3, no_of_saves: int = 100):
    if len(cooling_strenghts) > 4:
        warn("len(cooling_strengths) > 4, all items after the 4th will not be considered")
        cooling_strenghts = cooling_strenghts[:4]
    sims = [Simulation(iterations=iterations)
            for _ in range(len(cooling_strenghts))]
    fig = figure.Figure(figsize=(5, 5))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    linestyles = ["-", "--", "-.", ":"]

    for sim, drag, style in zip(sims, cooling_strenghts, linestyles):
        sim.begin(1, temp, gamma_drag=drag, number_of_saves=no_of_saves)
        sim.run()
        sim.end()

        ax1, lines1 = axplot_trajectory(ax1, sim.pos_history, lw=0.5, ls=style)
        ax2, lines2 = axplot_trajectory(ax2, sim.vel_history, lw=0.5, ls=style)

    kpatch = patches.Patch(color="k", label=r"$\cdot \hat{x}$")
    cpatch = patches.Patch(color="c", label=r"$\cdot \hat{y}$")
    mpatch = patches.Patch(color="m", label=r"$\cdot \hat{z}$")
    legendlines = [lines.Line2D([], [], color="grey", linestyle=style, label=fr"$\gamma_\mathrm{{IC}} = {drag['ic']}, \gamma_\mathrm{{SYN}} = {drag['syn']}$")
                   for style, drag in zip(linestyles, cooling_strenghts)]

    fig.legend(handles=kpatch + cpatch + mpatch + legendlines)

    ax1.set_ylabel("Position")
    ax2.set_ylabel("Proper Velocity")

    return fig


def axhist_energy_spectrum(ax: axes.Axes, U: np.ndarray, color: str, *args, **kwargs):
    gm1 = pusher.lorentz_factor(U) - 1
    opacity = np.linspace(0, 1, gm1.shape[0])

    for Ek, alpha in zip(gm1, opacity):
        l = ax.hist(np.log10(Ek), histtype="step", bins="fd",
                    alpha=alpha, color=color, *args, **kwargs)

    return ax, l


def exercise2(drags: list[dict], iterations: int = 1000, no_of_saves: int = 10, temp: float = 0.3):
    sims = [Simulation(iterations=iterations) for _ in range(len(drags))]

    fig = figure.Figure()
    ax = fig.add_subplot()

    for sim, drag, color in zip(sims, drags, colors.TABLEAU_COLORS.values()):
        sim.begin(int(1e5), temp, gamma_drag=drag, number_of_saves=no_of_saves)
        sim.run()
        sim.end()

        ax, l = axhist_energy_spectrum(ax, sim.vel_history, color)
        ax.set_xlabel(r"$\log_{10}\left(\gamma - 1\right)$")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")

    return fig, ax
