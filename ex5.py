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
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.style.use("ggplot")

    gamma_syn = 20
    gamma_ic = None
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
    # timestep = -1
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x_hist[timestep, :, 0], x_hist[timestep, :, 1], x_hist[timestep, :, 2], s=1, alpha=0.05, c='blue')
    # ax.set_title(f"Particle distribution ({plot_identifier})")
    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$y$")
    # ax.set_zlabel("$z$")
    # plt.show()


    # a) plot the pitch angle distribution
    timestep = 0
    Ee0 = utils.lorentz_factor(u_hist[timestep]) - 1
    Ei, Bi = Fields.from_file().interpolate(x_hist[timestep])
    alpha0 = pitch_angle(u_hist[timestep], Bi)

    timestep = -1
    Eef = utils.lorentz_factor(u_hist[timestep]) - 1
    Ei, Bi = Fields.from_file().interpolate(x_hist[timestep])
    alphaf = pitch_angle(u_hist[timestep], Bi)

    logE0 = np.log10(Ee0)
    logEf = np.log10(Eef)

    fig = plt.figure(figsize=(10, 5))

    # sns.kdeplot(x=logE0, y=alpha0, fill=False, cmap="Blues", alpha=1)
    # sns.kdeplot(x=logEf, y=alphaf, fill=False, cmap="Reds", alpha=1)
    plt.scatter(logE0, alpha0, s=3, alpha=0.03, c='blue', label='Initial distribution')
    plt.scatter(logEf, alphaf, s=3, alpha=0.03, c='red', label='Steady state distribution')

    plt.title(f"Pitch angle distribution ({plot_identifier})")
    plt.xlabel("$\\log_{10}(E_e)$")
    plt.ylabel("$\\alpha$")

    plt.xlim(-2, 3)
    plt.ylim(0, np.pi)

    plt.legend(loc='upper left', handles=[
        Patch(color='blue', alpha=0.5, label='Initial distribution'),
        Patch(color='red', alpha=0.5, label='Steady state distribution'),
    ])
    plt.show()