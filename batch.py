from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters
from utils import lorentz_factor


import numpy as np
from scipy.stats import kstest
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import os
from typing import Optional

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def run(gamma_syn: float,
        gamma_ic: float,
        N: int = 100_000,
        temperature: float = 0.3,
        saves=100,
        iterations: int = 6000,
        name: str = "test",
        prefix: Optional[str] = None
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

    x_hist = np.zeros((saves, N, 3))
    u_hist = np.zeros((saves, N, 3))

    # pvals = []

    for i, positions, velocities in sim.run(iterations, saves):
        x_hist[i] = positions
        u_hist[i] = velocities

    if not os.path.exists(f"{prefix}/{name}"):
        os.mkdir(f"{prefix}/{name}")
    np.save(f"{prefix}/{name}/x_hist.npy", x_hist)
    np.save(f"{prefix}/{name}/u_hist.npy", u_hist)


def main():
    from tqdm import tqdm
    gfactors = [3, 20, 100]

    gammas = [(g1, g2) for g1 in gfactors for g2 in gfactors]

    nstr = "1e5"
    saves = 100
    temp = 0.3
    name = f"M{nstr}-S{saves}-T{temp}"

    N = int(float(nstr))
    N_THREADS = min(os.cpu_count()-2, len(gammas))
    Parallel(n_jobs=N_THREADS)(
        delayed(run)(
            gamma_syn=g[0],
            gamma_ic=g[1],
            N=N,
            temperature=temp,
            saves=saves,
            name=name+f"-syn{g[0]}-ic{g[1]}"
        ) for g in tqdm(gammas) # progress bar will be instantly full if N_THREADS > len(gammas)
    )

if __name__ == '__main__':
    main()
