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
        lifetime: float = None,
        N: int = 100_000,
        temperature: float = 0.3,
        saves=100,
        iterations: int = 6000,
        name: str = "test",
        prefix: Optional[str] = None
        ):
    np.random.seed()

    fields = Fields.from_file()

    sim = Simulation(
        N=N,
        T=temperature,
        fields=fields,
        parameters=SimulationParameters(
            gamma_syn=gamma_syn, gamma_ic=gamma_ic, particle_lifetime=lifetime, cc=0.45, n_cells=fields.downsampling*fields.edges_cells[0]
        )
    )

    x_hist = np.zeros((saves, N, 3))
    u_hist = np.zeros((saves, N, 3))
    for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves):
        x_hist[i] = positions
        u_hist[i] = velocities

    if not os.path.exists(f"{prefix}/{name}"):
        os.mkdir(f"{prefix}/{name}")
    np.save(f"{prefix}/{name}/x_hist.npy", x_hist)
    np.save(f"{prefix}/{name}/u_hist.npy", u_hist)


def main():
    gfactors = [3, 20, 100, None]

    gammas = [(g1, g2) for g1 in gfactors for g2 in gfactors]
    gammas = [_ for _ in gammas if _[0] is None or _[1] is None]

    nstr = "1e5"
    iterations = 1000
    saves = iterations
    temp = 0.3

    N = int(float(nstr))
    N_THREADS = min(os.cpu_count()-2, len(gammas))
    # Parallel(n_jobs=N_THREADS)(
    #     delayed(run)(
    #         gamma_syn=g[0],
    #         gamma_ic=g[1],
    #         lifetime=None,
    #         N=N,
    #         temperature=temp,
    #         saves=saves,
    #         prefix="simulations",
    #         name=f"M{nstr}-S{saves}-T{temp}-syn{g[0]}-ic{g[1]}",
    #     ) for g in tqdm(gammas) # progress bar will be instantly full if N_THREADS >= len(gammas)
    # )

    lifetimes = [1, 5, 10]

    Parallel(n_jobs=N_THREADS)(
        delayed(run)(
            gamma_syn=None,
            gamma_ic=None,
            lifetime=tau,
            N=N,
            temperature=temp,
            iterations=iterations,
            saves=saves,
            prefix="simulations",
            name=f"M{nstr}-S{saves}-T{temp}-tau{tau}",
        ) for tau in tqdm(lifetimes) # progress bar will be instantly full if N_THREADS >= len(gammas)
    )

if __name__ == '__main__':
    main()
