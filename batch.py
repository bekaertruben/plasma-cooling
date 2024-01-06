from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters
from pusher import lorentz_factor


import numpy as np
from scipy.stats import kstest
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import os

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def run(gamma_syn: float,
        gamma_ic: float,
        N: int = 100_000,
        temperature: float = 0.3,
        saves=10,
        iterations: int = 1000,
        alpha: float = 0.05,
        name: str = "test",
        prefix: str = "refactored-simresults"
        ):
    np.random.seed()

    sim = Simulation(
        N=N,
        T=temperature,
        fields=Fields.from_file(),
        parameters=SimulationParameters(
            gamma_syn=gamma_syn, gamma_ic=gamma_ic, cc=0.45
        )
    )

    x_hist = np.zeros((0, N, 3))
    u_hist = np.zeros((0, N, 3))

    pvals = []

    while len(pvals) == 0 or pvals[-1] < (1 - alpha):
        x_hist = np.pad(x_hist, [(0, saves), (0, 0),
                        (0, 0)], constant_values=(0,))
        u_hist = np.pad(u_hist, [(0, saves), (0, 0),
                        (0, 0)], constant_values=(0,))
        for i, positions, velocities in sim.run(iterations, saves):
            x_hist[i[0] + saves * len(pvals)] = positions
            u_hist[i[0] + saves * len(pvals)] = velocities

        pvals.append(kstest(lorentz_factor(
            u_hist[-1]), lorentz_factor(u_hist[-2])).pvalue)

        print(f"{name} pvals:\t{pvals}")

        if len(pvals) >= 10:
            break

    if not os.path.exists(f"{prefix}/{name}"):
        os.mkdir(f"{prefix}/{name}")
    np.save(f"{prefix}/{name}/x_hist.npy", x_hist)
    np.save(f"{prefix}/{name}/u_hist.npy", u_hist)


def main():
    gfactors = [3, 30, 300]

    gammas = [(g1, g2) for g1 in gfactors for g2 in gfactors]

    nstr = "1e5"
    saves = 3
    temp = 0.3
    alpha = 0.05
    name = f"N{nstr}-S{saves}-T{temp}-alph{alpha}"

    N = int(float(nstr))
    Parallel(n_jobs=min(os.cpu_count()-2, len(gammas)))(delayed(run)(
        gamma_syn=g[0], gamma_ic=g[1], N=N, iterations=600, temperature=temp, alpha=alpha, saves=saves, name=name+f"-syn{g[0]}-ic{g[1]}") for g in gammas)


if __name__ == '__main__':
    main()
