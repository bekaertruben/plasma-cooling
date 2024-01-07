from utils import lorentz_factor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, colors
import os
import re
from typing import Optional
from maxwell_juttner import MaxwellJuttnerDistribution

prefix = "ex2"
names = [name for name in os.listdir(prefix) if name[0] == "N"]
names.sort()


def last_axhist_energy_spectrum(ax: axes.Axes, U: np.ndarray, color: str, *args, **kwargs):
    gm1 = lorentz_factor(U) - 1
    Ek = gm1[-1]
    # hist, bins = np.histogram(np.log10(Ek), bins="fd", density=True)
    # bincenters = 0.5 * (bins[1:] + bins[:-1])
    # ax.plot(bincenters, hist * (10**(bincenters))**2, alpha=alpha,
    #         color=color, *args, **kwargs)
    l = ax.hist(np.log10(Ek), histtype="step",
                bins="auto", color=color, *args, **kwargs)

    return ax


def power_law(U: np.ndarray, lorentz_range: Optional[tuple] = None):
    lor = lorentz_factor(U[-1])
    if lorentz_range is None:
        lorentz_range = (1, np.inf)
    lo, hi = lorentz_range
    temp = MaxwellJuttnerDistribution.fit(lor[(lo < lor) * (lor < hi)])
    print(f'fitted temperature: {temp}')
    return temp


def axplot_powerlaw(ax: axes.Axes, temp: float, lorentz_range: tuple, color: str, *args, **kwargs):
    gamma_space = np.geomspace(*lorentz_range, 200)
    mj = MaxwellJuttnerDistribution()
    mj_space = mj.pdf(gamma_space, temp)
    log_gminus1 = np.log10(gamma_space-1)
    l = ax.plot(log_gminus1, mj_space, color=color, *args, **kwargs)
    return ax, l, np.mean(log_gminus1)


def exercise3(names: list[str]):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()

    handles = []
    labels = []
    pattern = r"N(.+)-S(.+)-T(.+)-alph(.+)-syn(.+)-ic(.+)"
    for name, color in zip(names, colors.TABLEAU_COLORS.values()):
        match = re.search(pattern, name)
        N, S, T, alph, syn, ic = (match.group(i+1) for i in range(6))

        u_hist = np.load(f"{prefix}/{name}/u_hist.npy")
        ax, l, mean = last_axhist_energy_spectrum(ax, u_hist, color)

        handles.append(l)
        labels.append(
            fr"$\gamma_\mathrm{{syn}} = {syn}, \gamma_\mathrm{{IC}} = {ic}, \overline{{\log_{{10}}(\gamma - 1)}} = {mean}$")

    ax.set_xlabel(r"$\log_{10}\left(\gamma - 1\right)$")
    ax.set_ylabel(r"$dN_\mathrm{e}/d(\gamma - 1)$")
    ax.set_yscale("log")

    fig.legend(handles, labels)

    return fig, ax
