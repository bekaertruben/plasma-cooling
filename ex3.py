from utils import lorentz_factor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, colors, patches
import os
import re
from typing import Optional
from maxwell_juttner import MaxwellJuttnerDistribution as MJ

plt.style.use("ggplot")

prefix = "ex2"
names = [name for name in os.listdir(prefix) if name[0] == "N"]
names.sort()


def last_axhist_energy_spectrum(ax: axes.Axes, lor: np.ndarray, color: str, *args, **kwargs):
    # hist, bins = np.histogram(np.log10(Ek), bins="fd", density=True)
    # bincenters = 0.5 * (bins[1:] + bins[:-1])
    # ax.plot(bincenters, hist * (10**(bincenters))**2, alpha=alpha,
    #         color=color, *args, **kwargs)
    n, bins, patches = ax.hist(np.log10(lor-1), histtype="step",
                               bins="auto", color=color, density=False, *args, **kwargs)

    return ax, bins


def plot_fit(ax: axes.Axes, lordata: np.ndarray, color: str, norm: int = 1, N: Optional[int] = None,  *args, **kwargs):
    t, tstd = MJ.fit(lordata, N=N)

    fitspace = np.geomspace(min(lordata), max(lordata))
    fitcurve = MJ(t).pdf(fitspace)
    ufitcurve = MJ(t+tstd).pdf(fitspace)
    lfitcurve = MJ(t-tstd).pdf(fitspace)

    newnorm = norm * (fitspace-1) * np.log(10)

    l = ax.plot(np.log10(fitspace - 1), newnorm *
                fitcurve, color="k", *args, **kwargs)
    ax.fill_between(np.log10(fitspace-1), newnorm * lfitcurve,
                    newnorm * ufitcurve, alpha=0.2, color=color)

    return ax, l


def exercise3(names: list[str], prefix: str = prefix):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()

    handles = []
    labels = []
    pattern = r"N(.+)-S(.+)-T(.+)-alph(.+)-syn(.+)-ic(.+)"
    for name, color in zip(names, colors.TABLEAU_COLORS.values()):
        match = re.search(pattern, name)
        N, S, T, alph, syn, ic = (match.group(i+1) for i in range(6))

        u_hist = np.load(f"{prefix}/{name}/u_hist.npy")
        gamma = lorentz_factor(u_hist[-1])

        hist, bin_edges = np.histogram(np.log10(gamma - 1), bins=500)
        bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
        argpeak = bincenters[np.argmax(hist)] - 0.3

        ax, bins = last_axhist_energy_spectrum(ax, gamma, color)

        binw = bins[1]-bins[0]

        ax, lfit = plot_fit(
            ax, gamma[gamma < (10**argpeak + 1)], color, lw=0.8, ls="--", zorder=0, norm=len(gamma) * binw, N=len(gamma))  # lowenergy
        ax.vlines(argpeak, *ax.get_ylim())

        handles.append(patches.Patch(color=color))
        labels.append(
            fr"$\gamma_\mathrm{{syn}} = {syn}, \gamma_\mathrm{{IC}} = {ic}$")

    ax.set_xlabel(r"$\log_{10}\left(\gamma - 1\right)$")
    ax.set_ylabel(r"$dN_\mathrm{e}/d(\gamma - 1)$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=2)

    ax.legend(handles, labels, loc="upper left")

    return fig


def main():
    figs = []
    figcount = int(np.sqrt(len(names)))
    for i in range(figcount):
        figs.append(exercise3(names[figcount*i: figcount*(i+1)]))
        plt.show()


if __name__ == '__main__':
    main()
