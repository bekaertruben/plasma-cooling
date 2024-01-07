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
    import seaborn as sns
    import matplotlib.patches as mpatches
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
            gamma_syn=3.0, gamma_ic=3.0, cc=0.45
        )
    )

    E_hist = np.zeros((saves, N)) # kinetic energy
    alpha_hist = np.zeros((saves, N))
    for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
        Ei, Bi = fields.interpolate(positions)

        E_hist[i] = utils.lorentz_factor(velocities) - 1
        alpha_hist[i] = pitch_angle(velocities, Bi)

    logE = np.log10(E_hist)

    fig = plt.figure(figsize=(10, 5))

    sns.kdeplot(x=logE[0], y=alpha_hist[0], fill=True, cmap="Blues", alpha=1)
    sns.kdeplot(x=logE[-1], y=alpha_hist[-1], fill=True, cmap="Reds", alpha=0.5)

    plt.xlabel("$\\log_{10}(E_e)$")
    plt.ylabel("$\\alpha$")

    plt.xlim(-1.5, 1)

    plt.legend(loc='upper left', handles=[
        mpatches.Patch(color='blue', alpha=0.5, label='Initial distribution'),
        mpatches.Patch(color='red', alpha=0.5, label='Steady state distribution')
    ])
    plt.show()