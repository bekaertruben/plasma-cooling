from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters
from maxwell_juttner import MaxwellJuttnerDistribution
from utils import lorentz_factor

import numpy as np
from scipy import optimize

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import figure, axes, patches, lines, colors
plt.style.use("ggplot")


@np.vectorize
def powerlaw(x: np.ndarray, alpha: float, xmin: float):
    assert xmin > 0
    assert np.sum(x < 0) == 0
    assert alpha > 1
    return (alpha - 1) / xmin * (x / xmin) ** (-alpha)


def run_simulation(N=10_000, iterations=10_000, saves=100):
    fields = Fields.from_file()
    sim = Simulation(
        N = N,
        T = 0.3,
        fields = fields,
        parameters = SimulationParameters(
            gamma_syn = None,
            gamma_ic = None,
            particle_lifetime = 5,
            cc = 0.45,
            n_cells = fields.edges_cells[0] * fields.downsampling,
        )
    )

    u_hist = np.zeros((saves, sim.N, 3))
    x_hist = np.zeros((saves, sim.N, 3))
    for i, x, u in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
        u_hist[i] = u
        x_hist[i] = x
    
    np.save("sim6/u_hist.npy", u_hist)
    np.save("sim6/x_hist.npy", x_hist)



if __name__ == "__main__":
    # run_simulation()

    print("Loading simulation data...")
    prefix = f"sim6"
    u_hist = np.load(f"{prefix}/u_hist.npy")
    x_hist = np.load(f"{prefix}/x_hist.npy")
    print("Done.")

    # 1) plot the energy distribution in log-log scale
    gamma = lorentz_factor(u_hist[-1])
    N = gamma.shape[0]
    Ek = gamma - 1

    fig = plt.figure(figsize=(10, 5))

    hist, bins, _ = plt.hist(np.log10(Ek), histtype="step", bins="auto", density=True, color="green", label="Simulation Energy Distribution")
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    binwidths = bins[1:] - bins[:-1]
    binwidths_gamma = 10**bins[1:] - 10**bins[:-1]
    plt.yscale("log")

    # fit powerlaws to the data
    mask = (bincenters > 0) & (bincenters < 2)
    p, cov = np.polyfit(bincenters[mask], np.log10(hist[mask]), 1, cov=True)
    x = np.linspace(0, 4, 500)
    plt.plot(x, 10**p[1] * 10**(p[0] * x), "r:", label=f"Fitted Powerlaw $\\alpha = {p[0].round(2)}$")

    mask = (bincenters > 2.8)
    p, cov = np.polyfit(bincenters[mask], np.log10(hist[mask]), 1, cov=True)
    x = np.linspace(0, 4, 500)
    plt.plot(x, 10**p[1] * 10**(p[0] * x), "y:", label=f"Fitted Powerlaw $\\alpha = {p[0].round(2)}$")

    plt.ylim(1e-4, 1e1)
    plt.xlim(0, 4)

    plt.xlabel("$\log_{10} (E_k)$")
    plt.ylabel("$N(E_k)$")
    plt.title("Energy distribution of final state")

    plt.legend()
    plt.show()


    T, T_std = MaxwellJuttnerDistribution.fit(gamma, N=N)
    print(f"Temperature: {T} +/- {T_std}")
    mj = MaxwellJuttnerDistribution(T)

    plt.figure(figsize=(10, 5))
    hist, bins, _ = plt.hist(gamma, 100, density=True, histtype="step", color="green", label="Simulation Energy Distribution")
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bincenters, mj.pdf(bins[1:]+1), ":", color="red", label="Fitted Maxwell-Juttner Distribution")

    # fit an exponential to the data
    func = lambda x, a, tau: a * np.exp(-tau * (x-1))
    popt, pcov = optimize.curve_fit(func, bincenters, hist, p0=[1, 1])
    plt.plot(bincenters, func(bincenters, *popt), ":", color="blue", label=f"Fitted Exponential $\\tau = {popt[1].round(3)}$")

    # # fit a powerlaw to the data
    # func = lambda x, x_min, a: (x/x_min)**aTrajectories are initialized uniformly in the box. Lorentz factors from
    # mask = (bincenters > 0.5) & (bincenters < np.inf)
    # popt, pcov = optimize.curve_fit(func, bincenters[mask], hist, p0=[1, -2])
    # plt.plot(bincenters, func(bincenters, *popt), ":", color="orange", label=f"Fitted Powerlaw")

    plt.xlabel("$\gamma$")
    plt.ylabel("$N(\gamma)$")

    plt.legend()
    plt.show()