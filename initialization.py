from constants import *
import numpy as np
from scipy.stats import rv_continuous
from scipy.special import kn
from scipy.integrate import quad
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


class MaxwellJuttner(rv_continuous):
    """
    Describes the Maxwell-Jüttner distribution for the Lorentz factor
    See https://en.wikipedia.org/wiki/Maxwell-Jüttner_distribution

    Arguments
    ---------
    
    temp: float
        the temperature of the distribution in units of mc^2 / k_B
    """

    def _pdf(self, gamma, temp):
        if gamma < 1:
            return 0

        beta = np.sqrt(1 - 1/gamma**2)
        N =  gamma**2 * beta / (temp * kn(2, 1/temp))
        return N * gamma**2 * beta * np.exp(-gamma/temp)
    
    def _cdf(self, gamma, temp):
        if gamma < 1:
            return 0

        beta = np.sqrt(1 - 1/gamma**2)
        N =  1 / (temp * kn(2, 1/temp))

        def integrand(g):
            b = np.sqrt(1 - 1/g**2)
            return g**2 * b * np.exp(-g/temp)

        return N * quad(integrand, 1, gamma, epsabs=1e-5)[0]


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

    mj = MaxwellJuttner(name='maxwell_juttner')
    gammas = mj.rvs(temp=temp, size=N)
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
