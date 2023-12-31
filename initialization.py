from constants import *
from maxwell_juttner import MaxwellJuttner
import numpy as np
import pandas as pd
from warnings import warn
from typing import Optional


MJ_gammas = pd.read_csv('data/MJ_gammas.csv', index_col=0)


def sample_pos_uniform(N: int, edges_cells: np.ndarray = EDGES_CELLS):
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
    return np.random.rand(3, N) * edges_cells[:, np.newaxis]


def sample_velocity_thermal(N: int, temp: float):
    """
    Sample N velocity vectors from a thermal Maxwell-Jüttner distribution with temperature `temp`.

    Arguments
    ---------

    N: int
        number of particles

    temp: float
        the temperature of the distribution in units of mc^2 / k_B

    Returns
    -------

    u: numpy.ndarray (shape: (3, N))
        velocities
    """
    dirs = np.random.randn(3, N)
    dirs /= np.linalg.norm(dirs, axis=0)

    if f"T={temp}" in df.index and N <= df.shape[1]:
        gammas = df.loc[f"T={temp}"].sample(N).values
    else:
        mj = MaxwellJuttner(name='maxwell_juttner')
        gammas = mj.rvs(T=temp, size=N)

    us = np.sqrt(gammas**2 - 1)
    return us * dirs


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

    Bnorm = np.mean(bz)

    fields = {
        "ex": ex/Bnorm,
        "ey": ey/Bnorm,
        "ez": ez/Bnorm,
        "bx": bx/Bnorm,
        "by": by/Bnorm,
        "bz": bz/Bnorm
    }

    return fields, Bnorm
