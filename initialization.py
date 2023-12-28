from constants import *
import numpy as np
from scipy.stats import maxwell
from warnings import warn
from typing import Optional


def sample_pos_uniform(N: int, edges_meter: np.ndarray = EDGES_METER):
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
    return np.random.rand(3, N) * edges_meter[:, np.newaxis]


def sample_velocity_thermal(N: int, temp: float):
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
    u = np.random.randn(3, N)
    u /= np.linalg.norm(u, axis=0)
    scale = np.sqrt(KB * temp / ELECTRON_MASS)
    norm = maxwell.rvs(loc=0, scale=scale, size=N)
    where_superluminous = norm >= C
    if np.sum(where_superluminous) != 0:
        warn(
            f"Sampled {round(100* np.sum(where_superluminous) / N,2)}% superluminous particles (v >= c), setting their velocities to (1-1e-8)c.")
        norm[where_superluminous] = (1-1e-8) * C
    return u * norm[np.newaxis, :]


def load_fields(path: str = "data/flds.tot.00410"):
    "from Daniel"
    import h5py
    prec = "float32"
    f = h5py.File(path, 'r')
    ex = np.array(f["/ex"], dtype=prec).T
    ey = np.array(f["/ey"], dtype=prec).T
    ez = np.array(f["/ez"], dtype=prec).T
    bx = np.array(f["/bx"], dtype=prec).T
    by = np.array(f["/by"], dtype=prec).T
    bz = np.array(f["/bz"], dtype=prec).T
    f.close()

    fields = {
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "bx": bx,
        "by": by,
        "bz": bz
    }

    Bnorm = np.mean(bz)

    return fields, Bnorm


def uniform_B(bdir: str = "z", val: Optional[float] = None):
    directions = ["x", "y", "z"]
    if bdir not in directions:
        raise ValueError(f"Direction {bdir} not in {directions}")
    fields = {key: np.zeros((N_CELLS, N_CELLS, N_CELLS)) for key in FIELDNAMES}
    if val is not None:
        fields[f"b{bdir}"] += val
    else:
        fields[f"b{bdir}"] += 50
    return fields
