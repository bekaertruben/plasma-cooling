from utils import lorentz_factor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, colors, patches, lines
import os
import re
from typing import Optional
from maxwell_juttner import MaxwellJuttnerDistribution as MJ

plt.rcdefaults()
plt.style.use("ggplot")

prefix = "ex2"
names = [name for name in os.listdir(prefix) if name[0] == "N"]
names.sort()


def last_axhist_energy_spectrum(ax: axes.Axes, lor: np.ndarray, color: str, *args, **kwargs):
    # hist, bins = np.histogram(np.log10(Ek), bins="fd", density=True)
    # bincenters = 0.5 * (bins[1:] + bins[:-1])
    # ax.plot(bincenters, hist * (10**(bincenters))**2, alpha=alpha,
    #         color=color, *args, **kwargs)
    n, bin_edges = np.histogram(np.log10(lor-1), bins="auto", density=False)
    binc = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.stairs(n * (10**(binc))**3, bin_edges, color=color, *args, **kwargs)

    return ax, bin_edges


def plot_fit(ax: axes.Axes, lordata: np.ndarray, fitwhere: np.ndarray, color: str, norm: int = 1, N: Optional[int] = None,  *args, **kwargs):
    t, tstd = MJ.fit(lordata[fitwhere], N=N)

    fitspace = np.geomspace(min(lordata[fitwhere]), max(lordata[fitwhere]))
    fitspaceext = np.geomspace(max(lordata[fitwhere]), 10*max(lordata))
    fitcurve = MJ(t).pdf(fitspace)
    fitcurveext = MJ(t).pdf(fitspaceext)
    ufitcurve = MJ(t + 3 * tstd).pdf(fitspaceext)
    lfitcurve = MJ(t - 3 * tstd).pdf(fitspaceext)

    newnorm = norm * (fitspace-1) * np.log(10) * (fitspace-1)**3
    newnormext = norm * (fitspaceext-1) * np.log(10) * (fitspaceext-1)**3

    ax.plot(np.log10(fitspace - 1), newnorm *
            fitcurve, color="k", *args, **kwargs)
    ax.plot(np.log10(fitspaceext - 1), newnormext *
            fitcurveext, color="grey", *args, **kwargs)
    ax.fill_between(np.log10(fitspaceext-1), newnormext * lfitcurve,
                    newnormext * ufitcurve, alpha=0.2, color=color)

    return ax, t, tstd


@np.vectorize
def powerlaw(x: np.ndarray, alpha: float, xmin: float):
    assert xmin > 0
    assert np.sum(x < 0) == 0
    assert alpha > 1

    return (alpha - 1) / xmin * (x / xmin) ** (-alpha)


def plot_powerlaw(ax: axes.Axes, index: float, xmin: float, xmax: float, N: int = 1, *args, **kwargs):
    E = np.geomspace(xmin, xmax)  # gamma - 1
    y = N * powerlaw(E, index, xmin)

    ax.plot(np.log10(E), E**2 * y, ls=":", *args, **kwargs)

    return ax


def exercise3(names: list[str], prefix: str = prefix):
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot()

    handles = []
    labels = []
    pattern = r"N(.+)-S(.+)-T(.+)-alph(.+)-syn(.+)-ic(.+)"

    tfit = []
    tstdfit = []
    for name, color in zip(names, colors.TABLEAU_COLORS.values()):
        match = re.search(pattern, name)
        N, S, T, alph, syn, ic = (match.group(i+1) for i in range(6))

        u_hist = np.load(f"{prefix}/{name}/u_hist.npy")
        gamma = lorentz_factor(u_hist[-1])

        hist, bin_edges = np.histogram(np.log10(gamma - 1), bins=500)
        bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
        arglogpeak = bincenters[np.argmax(hist)]
        argpeak = 10**arglogpeak

        ax, bins = last_axhist_energy_spectrum(
            ax, gamma, color, zorder=0, lw=0.5)

        binw = bins[1]-bins[0]

        ax, t, tstd = plot_fit(
            ax, gamma, gamma < (10**(arglogpeak-0.3) + 1), color, ls="--", zorder=1, norm=len(gamma) * binw, N=len(gamma))  # lowenergy
        # ax.vlines(argpeak, *ax.get_ylim())

        # ax = plot_powerlaw(ax, 12, argpeak, 6*argpeak,
        #                    len(gamma[gamma - 1 >= argpeak]), color=color)

        tfit.append(t)
        tstdfit.append(tstd)

        # print(
        #     f"${syn}$,\t ${ic}$,\t ${round(np.mean(gamma),1)}$")
        # f"${syn}$,\t ${ic}$,\t ${round(t,4)}$, \t ${round(tstd,4)}$,")

        handles.append(patches.Patch(color=color))
        labels.append(
            fr"$\gamma_\mathrm{{syn}} = {syn}, \gamma_\mathrm{{IC}} = {ic}$")

    fitpatch = lines.Line2D([], [], color="k", ls="--")
    extpatch = lines.Line2D([], [], color="grey", ls="--")

    handles += [fitpatch, extpatch]
    labels += ["MJ Fit", "Fit Projection"]

    ax.set_xlabel(r"$\log_{10}\left(\gamma - 1\right)$")
    ax.set_ylabel(r"$E^3 \, dN_\mathrm{e}/d(\log\left(\gamma - 1)\right)$")
    ax.set_yscale("log")
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=2)

    # logE = np.linspace(-0.5, 3)
    # E = 10**logE
    # index = 0.5

    # def power(x, a):
    #     return x ** a

    # ax.plot(logE, E**2 * len(gamma)*power(E, index) /
    #         15., ls=":", color="purple")

    ax.legend(handles, labels, loc="upper left")

    return fig  # , tfit, tstdfit


def main():
    figcount = int(np.sqrt(len(names)))
    for i in range(figcount):
        fig = exercise3(names[figcount*i: figcount*(i+1)])
        fig.subplots_adjust(bottom=0.15)
        # fig.savefig(f"ex2/ex3-{i}.png", facecolor="white")
        plt.show()
        # break


if __name__ == '__main__':
    main()
