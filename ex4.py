from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters
import utils

import numpy as np


def transferred_power(x, u, E, B):
    """ See exercise 4 problem statement """
    E_parallel = utils.project(E, B)
    E_perpendicular = E - E_parallel

    v = u / utils.lorentz_factor(u)[..., np.newaxis]

    # using q = -1
    P_parallel = - np.sum(v * E_parallel, axis=-1)
    P_perpendicular = - np.sum(v * E_perpendicular, axis=-1)

    return P_parallel, P_perpendicular


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.style.use("ggplot")

    gamma_syn = None
    gamma_ic = 3
    plot_identifier = f"$\\gamma_\\text{{syn}} = {gamma_syn}$, $\\gamma_\\text{{IC}} = {gamma_ic}$"

    # load simulation data
    print("Loading simulation data...")
    prefix = f"./simulations/M1e5-S100-T0.3-syn{gamma_syn}-ic{gamma_ic}"
    u_hist = np.load(f"{prefix}/u_hist.npy")
    x_hist = np.load(f"{prefix}/x_hist.npy")
    print("Done.")

    # print("Loading simulation data...")
    # prefix = f"sim6"
    # u_hist = np.load(f"{prefix}/u_hist.npy")
    # x_hist = np.load(f"{prefix}/x_hist.npy")
    # print("Done.")
    # plot_identifier = "particle escape $\\tau=5$"


    # 0) plot the particle positions
    timestep = -1
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_hist[timestep, :, 0], x_hist[timestep, :, 1], x_hist[timestep, :, 2], s=1, alpha=0.05, c='blue')
    ax.set_title(f"Particle distribution ({plot_identifier})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    # plt.show()


    # a) plot the ratio of parallel to perpendicular power
    timestep = -1
    Ee = utils.lorentz_factor(u_hist[timestep]) - 1
    Ei, Bi = Fields.from_file().interpolate(x_hist[timestep])
    Ppar, Pperp = transferred_power(x_hist[timestep], u_hist[timestep], Ei, Bi)
    Pratio = Ppar / Pperp

    ratio_order_mean = np.log10(np.abs(Pratio)).mean()
    ratio_order_spread = np.log10(np.abs(Pratio)).std()
    print(f"Mean order of magnitude of ratio: {ratio_order_mean} \\pm {ratio_order_spread}")

    print(f"Minimum energy (log): {np.log10(Ee.min())}, Maximum energy (log): {np.log10(Ee.max())}")
    # bins_edges = np.logspace(np.log10(Ee.min()), np.log10(Ee.max()), 20)
    bins_edges = np.logspace(np.log10(0.2), np.log10(2), 20) 
    bins = np.moveaxis(np.array([bins_edges[:-1], bins_edges[1:]]), 0, -1)
    bin_centers = 0.5 * (bins[:, 1] + bins[:, 0])
    averages = np.zeros(len(bins))
    averages_err = np.zeros(len(bins))
    for i, (a, b) in enumerate(bins):
        mask = (Ee >= a) & (Ee < b)
        if mask.sum() <= 50:
            averages[i] = np.nan
            averages_err[i] = np.nan
        else:
            averages[i] = 10**np.log10(np.abs(Pratio[mask])).mean()
            averages_err[i] = 10**np.log10(np.abs(Pratio[mask])).std() / np.sqrt(mask.sum() - 1)
    
    logbincenters = np.log10(bin_centers)
    logaverages = np.log10(averages)
    (slope, intercept), cov = np.polyfit(
        logbincenters[~np.isnan(logaverages)],
        logaverages[~np.isnan(logaverages)],
        1,
        w=1/averages_err[~np.isnan(logaverages)],
        cov=True
    )
    print(f"Slope of average: {slope} \\pm {np.sqrt(cov[0,0])}")
    print(f"Intercept of average: {intercept} \\pm {np.sqrt(cov[1,1])}")

    fig = plt.figure(figsize=(10, 5))

    plt.scatter(Ee, -Pratio, s=1, alpha=0.05, c='red')
    plt.scatter(Ee, Pratio, s=1, alpha=0.05, c='blue')
    plt.plot(bin_centers, averages, ":", color="black", label="Average")

    plt.xlim(1e-2, 1e1)
    # plt.ylim(1e-7, 1e3)

    plt.title(f"Ratio of parallel to perpendicular power ({plot_identifier})")
    plt.xlabel("$E_e$ [$m_e c^2$]")
    plt.ylabel("$| P_{\\parallel} / P_{\\perp} |$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper left', handles=[
        Patch(color='blue', alpha=0.5, label='Positive'),
        Patch(color='red', alpha=0.5, label='Negative'),
        Line2D([], [], linestyle=":", color='black', label='Average')
    ])
    plt.show()

    # b) plot the integrated power over time
    Ei, Bi = Fields.from_file().interpolate(x_hist)
    Ppar, Pperp = transferred_power(x_hist, u_hist, Ei, Bi)

    Ppar_int = 6000/100 * np.cumsum(Ppar, axis=0)
    Pperp_int = 6000/100 * np.cumsum(Pperp, axis=0)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    t = np.linspace(0, 6000, 100)

    color1 = 'tab:red'
    ax1.plot(t, Ppar_int.mean(axis=-1), color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylabel("Mean $\\int P_{\\parallel} \\text{d}t$", color=color1)
    ax1.set_xlabel("Time")

    color2 = 'tab:blue'
    ax2.plot(t, Pperp_int.mean(axis=-1), color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylabel("Mean $\\int P_{\\perp} \\text{d}t$", color=color2)

    plt.title(f"Integrated power ({plot_identifier})")
    # plt.xlabel("Time")
    # plt.legend()
    plt.grid(None)
    plt.show()