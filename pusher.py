import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib import figure, gridspec
import os
from typing import Optional
from initialization import *
from tqdm import tqdm

from constants import *


def apply_periodicity(X0: np.ndarray, edges: np.ndarray):
    """
    X0: (N,3)

    edges: (3,)
        assumes boundaries of [(0,edge) for edge in edges]
    """

    return np.mod(X0, edges[:, np.newaxis])


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

    return np.sqrt(1 + np.sum(np.square(u)/C**2, axis=0))


def boris_push(x0: np.ndarray, u0: np.ndarray, fields: dict):
    """
    Borish Pusher
    -----

    x0: (3,N)
    u0: (3,N)

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'
    """
    if set(fields.keys()) != set(FIELDNAMES):
        raise KeyError(
            f"`fields` keys ({fields.keys()}) does not corresponds with {FIELDNAMES}")

    xci = x0 + u0*DT / (2 * lorentz_factor(u0))

    # interpolate fields. if bottleneck could make it multiprocessed
    fields_ci = {key: interpolate_field(xci, value)
                 for key, value in fields.items()}

    # fields_ci[key] is a (N,) shaped array

    # Fci shape: (3,N)
    Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])

    umin = u0 + Q_OVER_M * DT * Eci / 2

    # == lorentz_factor(uplus) == lorentz_factor(uci) see paper
    g = lorentz_factor(umin)

    t = Bci * Q_OVER_M * DT / (2 * g)

    s = 2*t / (1 + np.linalg.norm(t)**2)

    uplus = umin + \
        np.cross((umin + np.cross(umin, t, axis=0)),
                 s, axis=0)

    unext = uplus + Q_OVER_M * DT * Eci / 2
    xnext = xci + unext * DT / (2 * g)

    xnext = apply_periodicity(xnext, EDGES_METER)

    return xnext, unext, Eci, Bci

# TODO: verify difference between relativistic velocity and regular (u = \gamma v or \beta = \gamma u or whatever). We are very much using velocity I believe


def radiate_synchrotron(u0: np.ndarray, u1: np.ndarray, Eci: np.ndarray, Bci: np.ndarray, Bnorm: float):
    """
    Compute the radiative drag on a particle due to synchrotron radiation.

    Arguments
    ---------

    u0: np.ndarray (shape: (3,N))
        particle velocities at step n - 1/2

    u1: np.ndarray (shape: (3,N))
        particle velocities at step n + 1/2 from the unmodified Boris pusher

    Eci: np.ndarray (shape: (3,N))
        interpolated electric field at step n

    Bci: np.ndarray (shape: (3,N))
        interpolated magnetic field at step n

    Bnorm: float
        average z magnetic field

    Returns
    -------

    unext: np.ndarray (shape: (3,N))
        synchrotron dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)
    betaci = uci/gci

    Ebar = Eci + np.cross(betaci, Bci, axis=0)

    beta_dot_e = betaci @ Eci

    kappa_R = np.cross(Ebar, Bci, axis=0) + beta_dot_e * Eci
    chi_R_sq = np.square(np.norm(Ebar, axis=0)) - np.square(beta_dot_e)

    prefactor = abs(Q_OVER_M) * Bnorm * BETA_REC / (C * GAMMA_SYN**2)

    unext = u0 + DT * prefactor * (kappa_R - gci**2 * chi_R_sq * betaci)
    return unext


def radiate_inversecompton(u0: np.ndarray, u1: np.ndarray, Bnorm: float):
    """
    Compute the radiative drag on a particle due to inverse compton radiation.

    Arguments
    ---------

    u0: np.ndarray (shape: (3,N))
        particle velocities at step n - 1/2

    u1: np.ndarray (shape: (3,N))
        particle velocities at step n + 1/2 from the unmodified Boris pusher

    Bnorm: float
        average z magnetic field

    Returns
    -------

    unext: np.ndarray (shape: (3,N))
        inverse compton dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)
    betaci = uci/gci

    prefactor = abs(Q_OVER_M) * Bnorm * BETA_REC / (C * GAMMA_IC**2)

    unext = u0 - DT * prefactor * betaci * gci ** 2
    return unext


def push(x0: np.ndarray, u0: np.ndarray, fields: dict, syn_drag: bool = True, ic_drag: bool = True, Bnorm: Optional[float] = None):
    """
    Combine the unmodified Boris pusher with radiative drag (inverse compton and synchrotron).
    This pusher assumes a dominant contribution from the Lorentz force, such that no modified pusher is needed.

    Arguments
    ---------

    x0: np.ndarray (shape: (3,N))
        particle positions at step n

    u0: np.ndarray (shape: (3,N))
        particle velocities at step n - 1/2

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS)

    syn_drag: bool
        whether to include synchrotron drag

    ic_drag: bool
        whether to include inverse compton drag

    Bnorm: float
        normalization of the magnetic field. If None, the mean of the magnetic field in the z direction is used.

    Returns
    -------

    xnext: np.ndarray (shape: (3,N))
        particle positions at step n + 1

    unext: np.ndarray (shape: (3,N))
        particle velocities at step n + 1/2

    """
    if Bnorm is None:
        Bnorm = np.mean(fields["bz"])

    xnext, u_lorentz, Eci, Bci = boris_push(x0, u0, fields)

    if not syn_drag and not ic_drag:
        return xnext, u_lorentz

    if syn_drag:
        u_syn = radiate_synchrotron(u0, u_lorentz, Eci, Bci, Bnorm)

    if ic_drag:
        u_ic = radiate_inversecompton(u0, u_lorentz, Bnorm)

    if syn_drag and not ic_drag:
        unext = u_lorentz + u_syn - u0
        return xnext, unext

    elif ic_drag and not syn_drag:
        unext = u_lorentz + u_ic - u0
        return xnext, unext

    unext = u_lorentz + u_syn + u_ic - 2 * u0
    return xnext, unext


def kinetic_energy(u_history: list[np.array], mass: float = 1.0):
    Ek = []
    for u in tqdm(u_history):
        Ek.append(mass * lorentz_factor(u)*C**2)
    return Ek


def main():

    # fields = load_fields()
    fields = uniform_B()

    N_PARTICLES = 1
    # x = init_random_x(N_PARTICLES)
    # u = init_random_u(N_PARTICLES)
    # manual
    x = np.asarray([[5], [5], [5]])*1e7

    u = sample_velocity_thermal(N_PARTICLES, 5e5)
    print(u)

    x_history = []
    y_history = []
    z_history = []

    u_history = []

    for _ in tqdm(range(ITERATIONS)):
        x, u = boris_push(x, u, fields)

        x_history.append(x[0])
        y_history.append(x[1])
        z_history.append(x[2])
        u_history.append(u)

    # fig = plt.figure(figsize=(4, 3.5))
    fig = figure.Figure(figsize=(4, 3.5))
    ax = fig.add_subplot(projection="3d")

    for i in range(N_PARTICLES):
        ax.scatter(np.asarray(x_history)[:, i], np.asarray(y_history)[:, i], np.asarray(z_history)[:, i],
                   c=np.arange(ITERATIONS), cmap="rainbow", s=.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_xlim([0, BOXSIZE])
    ax.set_ylim([0, BOXSIZE])
    ax.set_zlim([0, BOXSIZE])
    fig.suptitle("Particle trajectory (purple is early, red is later)")

    if not os.path.exists("images"):
        os.mkdir("images")

    fig.savefig("images/test.png", facecolor="white")
    Ek = np.array(kinetic_energy(u_history), dtype=float)
    print(f"sum absoulute diff of Ek {np.sum(np.abs(np.diff(Ek)))}")
    return fig


if __name__ == '__main__':
    main()
