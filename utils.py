import numpy as np


def apply_periodicity(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """ Wraps positions to the simulation box defined by edges """
    return np.mod(x, edges)


def lorentz_factor(u: np.ndarray) -> np.ndarray:
    """ Compute the lorentz factor of particles with velocity u. """
    return np.sqrt(1 + np.sum(np.square(u), axis=-1))


def proper_velocity(gammas: np.ndarray) -> np.ndarray:
    """ Compute the proper velocity of particles with lorentz factor gamma. """
    return np.sqrt(gammas**2 - 1)


def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Project vector a onto vector b (last axis) """
    assert a.shape == b.shape, f"Shapes of a and b must match, got {a.shape} and {b.shape}"
    b_sqr = np.sum(b**2, axis=-1)
    dot_poducts = np.sum(a * b, axis=-1)
    return (dot_poducts / b_sqr)[..., np.newaxis] * b