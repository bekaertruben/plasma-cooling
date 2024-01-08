from utils import lorentz_factor
import numpy as np
from matplotlib import figure, axes, patches, lines, colors
# from typing import Optional
# from warnings import warn
import matplotlib.pyplot as plt
import os
import re

plt.rcdefaults()
plt.style.use("ggplot")

prefix = "ex2"
names = [name for name in os.listdir(prefix) if name[0] == "N"]
names.sort()


def axhist_energy_spectrum(ax: axes.Axes, U: np.ndarray, color: str, *args, **kwargs):
    gm1 = lorentz_factor(U) - 1
    opacity = np.linspace(0, 1, gm1.shape[0])

    for Ek, alpha in zip(gm1, opacity):
        # hist, bins = np.histogram(np.log10(Ek), bins="fd", density=True)
        # bincenters = 0.5 * (bins[1:] + bins[:-1])
        # ax.plot(bincenters, hist * (10**(bincenters))**2, alpha=alpha,
        #         color=color, *args, **kwargs)
        l = ax.hist(np.log10(Ek), histtype="step", bins="auto",
                    alpha=alpha, color=color, *args, **kwargs)

    return ax


def exercise2(names: list[str], prefix: str = prefix, lc: int = 5):
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot()

    handles = []
    labels = []
    pattern = r"N(.+)-S(.+)-T(.+)-alph(.+)-syn(.+)-ic(.+)"

    for name, color in zip(names, colors.TABLEAU_COLORS.values()):
        match = re.search(pattern, name)
        N, S, T, alph, syn, ic = (match.group(i+1) for i in range(6))

        u_hist = np.load(f"{prefix}/{name}/u_hist.npy")
        ax = axhist_energy_spectrum(ax, u_hist, color)
        # ax = axhist_energy_spectrum(ax, u_hist[::(len(u_hist) // lc)], color)
        handles.append(patches.Rectangle(
            (0, 0), 1, 1, ec=color, fill=False))
        labels.append(
            fr"$\gamma_\mathrm{{syn}} = {syn}, \gamma_\mathrm{{IC}} = {ic}$")

    ax.set_xlabel(r"$\log_{10}\left(\gamma - 1\right)$")
    ax.set_ylabel(
        r"$N\left(\,E_\mathrm{kin}\, / \,m_e c^2 = \gamma - 1\right)$")
    ax.set_yscale("log")

    ax.set_xlim(left=-3.4)

    ax.legend(handles, labels, loc="upper left", fontsize="small")

    return fig


def main():
    # figs = []
    figcount = int(np.sqrt(len(names)))
    for i in range(figcount):
        fig = exercise2(names[figcount*i: figcount*(i+1)], lc=10)
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(f"ex2/images/ex2-{i+1}.png", facecolor="white")
        # plt.show()
        # break


if __name__ == '__main__':
    main()
