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

    # using q = 1
    P_parallel = np.sum(v * E_parallel, axis=-1)
    P_perpendicular = np.sum(v * E_perpendicular, axis=-1)

    return P_parallel, P_perpendicular


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

    N = 100_000
    iterations = 500
    saves = 100

    fields = Fields.from_file()
    sim = Simulation(
        N=N,
        T=1,
        fields=fields,
        parameters=SimulationParameters(
            gamma_syn=2.0, gamma_ic=2.0, cc=0.45
        )
    )

    p_hist = np.zeros((saves, 2, N))
    for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
        Ei, Bi = fields.interpolate(positions)
        p_hist[i] = transferred_power(positions, velocities, Ei, Bi)
    
    p_parallel_total = p_hist[:, 0, :].sum(axis=-1)
    p_perpendicular_total = p_hist[:, 1, :].sum(axis=-1)

    fig = plt.figure(figsize=(10, 5))
    t = np.linspace(0, iterations, saves)
    plt.scatter(t, p_parallel_total / p_perpendicular_total)
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("$P_{\\parallel} / P_{\\perp}$")
    plt.show()