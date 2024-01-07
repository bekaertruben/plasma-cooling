from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters
import utils

import numpy as np


def pitch_angle(u: np.ndarray, B: np.ndarray):
    """ See exercise 5 problem statement """
    assert u.shape == B.shape, f"Shapes of u and B must match, got {u.shape} and {B.shape}"
    u_norm = u / np.linalg.norm(u, axis=-1)[..., np.newaxis]
    B_norm = B / np.linalg.norm(B, axis=-1)[..., np.newaxis]
    return np.arccos(np.sum(u_norm * B_norm, axis=-1))


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    N = 10_000
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

    alpha_hist = np.zeros((saves, N))
    for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
        Ei, Bi = fields.interpolate(positions)
        alpha_hist[i] = pitch_angle(velocities, Bi)


    fig = plt.figure(figsize=(10, 5))
    t = np.linspace(0, iterations, saves)
    plt.scatter(t, ...)
    plt.xlabel("Time")
    plt.ylabel("$P_{\\parallel} / P_{\\perp}$")
    plt.show()