import numpy as np
from scipy.ndimage import map_coordinates


def apply_periodicity(X0: np.ndarray, edges: np.ndarray):
    """
    X0: (N,3)

    edges: (3,)
        assumes boundaries of [(0,edge) for edge in edges]
    """

    return np.mod(X0, edges)


def interpolate_field(x0: np.ndarray, F: np.ndarray):
    """
    x0: (3,N)
    F: (nx,ny,nz)
    """
    return map_coordinates(F, x0, order=1, mode='grid-wrap')


def lorentz_factor(u: np.ndarray):
    """
    u: (3,N)
    """

    return np.sqrt(1 + np.sum(np.square(u), axis=0))


def boris_push(x0: np.ndarray, u0: np.ndarray, fields: dict, dt: float, q_over_m: float):
    """
    Borish Pusher
    -----

    x0: (3,N)
    u0: (3,N)

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'
    """

    xci = x0 + u0*dt / (2 * lorentz_factor(u0))

    # interpolate fields if bottleneck could make it multiprocessed
    fields_ci = {key: interpolate_field(xci, value)
                 for key, value in fields.items()}

    # fields_ci[key] is a (N,) shaped array

    # Fci shape: (3,N)
    Eci = np.asarray([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.asarray([fields_ci[key] for key in ["bx", "by", "bz"]])

    umin = u0 + q_over_m * dt * Eci / 2

    # == lorentz_factor(uplus) == lorentz_factor(uci) see paper
    g = lorentz_factor(umin)

    t = Bci * q_over_m * dt / (2 * g)

    s = 2*t / (1 + np.linalg.norm(t)**2)

    uplus = umin + \
        np.cross((umin + np.cross(umin, t, axisa=0, axisb=0)),
                 s, axisa=0, axisb=0)

    unext = uplus + q_over_m * dt * Eci / 2
    xnext = xci + unext * dt / (2 * g)

    return xnext, unext
