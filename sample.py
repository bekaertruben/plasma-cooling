from constants import *
import numpy as np
from scipy.stats import maxwell
from warnings import warn


def position_uniform(N: int):
    """
    Sample uniformly positions in the simulation space.

    Arguments
    ---------

    N: int
        number of particles

    Returns
    -------

    numpy.ndarray (shape: (3, N))
        sampled positions

    """
    return np.random.rand(3, N) * EDGES_METER[:, np.newaxis]


def velocity_thermal(N: int, temp: float):
    """
    Sample N velocity vectors from a thermal Maxwell distribution with temperature `temp`.
    The sampling is not safe against superluminous particles, do not put temperatures above ~1e8 K

    Arguments
    ---------

    N: int
        number of particles

    temp: float
        distribution temperature in Kelvin

    Returns
    -------

    u: numpy.ndarray (shape: (3, N))
        velocities
    """
    u = np.random.rand(3, N)
    u /= np.linalg.norm(u, axis=0)
    scale = np.sqrt(KB * temp / ELECTRON_MASS)
    norm = maxwell.rvs(loc=0, scale=scale, size=N)
    where_superluminous = norm >= C
    if np.sum(where_superluminous) != 0:
        warn(
            f"Sampled {round(100* np.sum(where_superluminous) / N,2)}% superluminous particles (v >= c), setting their velocities to (1-1e-8)c.")
        norm[where_superluminous] = (1-1e-8) * C
    return u * norm[np.newaxis, :]


def main():
    from matplotlib import figure
    N = int(1e5)
    pos = position_uniform(N)

    fig = figure.Figure()
    ax = fig.add_subplot()

    for i in range(3):
        ax.hist(pos[i], histtype="step", bins="fd", label=f"$x_{i}$")
    ax.set_xlabel("position")
    ax.set_ylabel("counts")
    ax.legend()
    fig.savefig("images/positions.png", facecolor="white")


if __name__ == '__main__':
    main()
