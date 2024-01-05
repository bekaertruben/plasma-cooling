import numpy as np
from typing import Optional

from fields import Fields
from simulation_parameters import SimulationParameters


def apply_periodicity(x: np.ndarray, edges: np.ndarray):
    """ Wraps positions to the simulation box defined by edges """
    return np.mod(x, edges)


def lorentz_factor(u: np.ndarray):
    """ Compute the lorentz factor of particles with velocity u. """
    return np.sqrt(1 + np.sum(np.square(u), axis=-1))


def boris_push(
        xci: np.ndarray,
        u0: np.ndarray,
        fields: Fields,
        sim_params: SimulationParameters
):
    """ Borish push (Lorentz force) on particles with velocity u0 and position x0.

    Arguments
    ---------
    x0: np.ndarray (shape: (N,3))
        particle positions at step n
    u0: np.ndarray (shape: (N,3))
        particle velocities at step n - 1/2
    fields: Fields
        the electric and magnetic fields (unaltered by the pusher)
    sim_params: SimulationParameters
        simulation parameters

    Returns
    -------
    xnext: np.ndarray (shape: (N,3))
        particle positions at step n + 1
    unext: np.ndarray (shape: (N,3))
        particle velocities at step n + 1/2
    Eci: np.ndarray (shape: (N,3))
        interpolated electric field at step n
    Bci: np.ndarray (shape: (N,3))
        interpolated magnetic field at step n
    """
    cc = sim_params.cc

    Eci, Bci = fields.interpolate(xci)

    dummy = 0.5 * sim_params.q_over_m * fields.Bnorm
    e0 = Eci * dummy
    b0 = Bci * dummy

    # half acceleration
    u1prime = cc * u0 + e0

    # first half magnetic rotation
    gamma1 = lorentz_factor(u1prime / cc)[..., np.newaxis]
    f = 2. / (1. + np.sum(np.square(b0/(cc * gamma1)), axis=-1)
              )[..., np.newaxis]
    u2prime = (u1prime + np.cross(u1prime/(cc * gamma1), b0, axis=-1))*f

    # second half magnetic rotation + half acceleration
    u3prime = u1prime + np.cross(u2prime/(cc * gamma1), b0, axis=-1) + e0
    unext = u3prime / cc

    return unext, Eci, Bci


def radiate_synchrotron(
        u0: np.ndarray,
        u1: np.ndarray,
        Eci: np.ndarray,
        Bci: np.ndarray,
        Bnorm: float,
        sim_params: SimulationParameters
):
    """
    Compute the radiative drag on a particle due to synchrotron radiation.

    Arguments
    ---------
    u0: np.ndarray (shape: (N,3))
        particle velocities at step n - 1/2
    u1: np.ndarray (shape: (N,3))
        particle velocities at step n + 1/2 from the unmodified Boris pusher
    Eci: np.ndarray (shape: (N,3))
        interpolated electric field at step n
    Bci: np.ndarray (shape: (N,3))
        interpolated magnetic field at step n
    Bnorm: float
        average z magnetic field
    sim_params: SimulationParameters
        simulation parameters (require gamma_syn, beta_rec, cc)

    Returns
    -------
    unext: np.ndarray (shape: (N,3))
        synchrotron dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)[..., np.newaxis]
    betaci = uci / gci

    Ebar = Eci + np.cross(betaci, Bci, axis=-1)

    beta_dot_e = np.einsum("ij,ij->i", betaci, Eci)[..., np.newaxis]

    kappa_R = np.cross(Ebar, Bci, axis=-1) + beta_dot_e * Eci
    chi_R_sq = np.sum(np.square(Ebar), axis=-1)[:, np.newaxis] - beta_dot_e**2

    prefactor = Bnorm * sim_params.beta_rec / \
        (sim_params.cc * sim_params.gamma_syn**2)

    unext = u0 + prefactor * (kappa_R - chi_R_sq * gci * uci)

    gnext = lorentz_factor(unext)
    return unext


def radiate_inversecompton(
        u0: np.ndarray,
        u1: np.ndarray,
        Bnorm: float,
        sim_params: SimulationParameters
):
    """
    Compute the radiative drag on a particle due to inverse compton radiation.

    Arguments
    ---------
    u0: np.ndarray (shape: (N, 3))
        particle velocities at step n - 1/2

    u1: np.ndarray (shape: (N, 3))
        particle velocities at step n + 1/2 from the unmodified Boris pusher

    Bnorm: float
        average z magnetic field

    sim_params: SimulationParameters
        simulation parameters (require gamma_ic, cc)

    Returns
    -------
    unext: np.ndarray (shape: (N, 3))
        inverse compton dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)[..., np.newaxis]

    dummy = Bnorm * sim_params.beta_rec / \
        (sim_params.cc * sim_params.gamma_ic**2)

    unext = u0 - dummy * uci * gci
    return unext


def push(
        x0: np.ndarray,
        u0: np.ndarray,
        fields: Fields,
        sim_params: SimulationParameters
):
    """ Combine the unmodified Boris pusher with radiative drag (inverse compton and synchrotron).
    This pusher assumes a dominant contribution from the Lorentz force, such that no modified pusher is needed.

    Arguments
    ---------
    x0: np.ndarray (shape: (N,3))
        particle positions at step n
    u0: np.ndarray (shape: (N,3))
        particle velocities at step n - 1/2
    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS)
    sim_params: SimulationParameters
        simulation parameters

    Returns
    -------
    xnext: np.ndarray (shape: (3,N))
        particle positions at step n + 1

    unext: np.ndarray (shape: (3,N))
        particle velocities at step n + 1/2

    """
    g0 = lorentz_factor(u0)[..., np.newaxis]
    xci = x0 + u0 / (2 * g0)
    u_lorentz, Eci, Bci = boris_push(x0, u0, fields, sim_params)

    syn_drag = sim_params.gamma_syn != None
    ic_drag = sim_params.gamma_ic != None

    if syn_drag:
        u_syn = radiate_synchrotron(
            u0, u_lorentz, Eci, Bci, fields.Bnorm, sim_params)

    if ic_drag:
        u_ic = radiate_inversecompton(u0, u_lorentz, fields.Bnorm, sim_params)

    if not syn_drag and not ic_drag:
        unext = u_lorentz

    elif syn_drag and not ic_drag:
        unext = u_lorentz + u_syn - u0

    elif ic_drag and not syn_drag:
        unext = u_lorentz + u_ic - u0

    elif ic_drag and syn_drag:
        unext = u_lorentz + u_syn + u_ic - 2 * u0

    gnext = lorentz_factor(unext)[..., np.newaxis]
    xnext = xci + unext / (2 * gnext)
    xnext = apply_periodicity(xnext, sim_params.edges_cells)
    return xnext, unext


def transferred_power(
        velocity: np.ndarray,
        Eci: Optional[np.ndarray] = None,
        Bci: Optional[np.ndarray] = None,
        fields: Optional[dict] = None,
        position: Optional[np.ndarray] = None
):
    """
    Get the parallel and perpendicular power transferred from field to particle.

    Arguments
    ---------
    charge: float or list[float]
        particle charge in Coulomb

    velocity: np.ndarray (shape: (..., 3))
        particle velocities at step n + 1/2

    Eci: Optional[np.ndarray] (shape: (..., 3)) (default: None)
        interpolated electric field at steps n. If None, fields and position must be passed.

    Bci: Optional[np.ndarray] (shape: (..., 3)) (default: None)
        interpolated magnetic field at steps n. If None, fields and position must be passed.

    fields: Optional[dict] (default: None)
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS).
        If None, Eci and Bci must be passed.

    position: Optional[np.ndarray] (shape: (..., 3)) (default: None)
        particle positions at steps n. If None, Eci and Bci must be passed.

    Returns
    -------
    Ppar / Pperp: np.ndarray (shape: (...))
        ratio parallel over perpendicular power

    """
    if (Eci is None or Bci is None) and (fields is None or position is None):
        raise ValueError(
            "Eci and Bci have to be passed, or fields and position. No consistent arguments are given.")

    if Eci is None or Bci is None:
        fields_ci = {key: interpolate_field(
            position, value) for key, value in fields.items()}
        Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
        Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])
        if len(position.shape) >= 2:
            Eci = np.swapaxes(Eci, 0, 1)
            Bci = np.swapaxes(Bci, 0, 1)

    # Epar = np.diag(Eci.T @ Bci) * Bci / \
    #     np.linalg.norm(Bci, axis=len(Bci.shape)-1) ** 2
    normalization = np.linalg.norm(Bci, axis=-2) ** 2
    Epar = np.einsum("...ij,...ij,...ik->...ik", Eci, Bci, Bci)
    Epar /= normalization[:, np.newaxis, :]
    Eperp = Eci - Epar

    Ppar = np.einsum("...ij,...ij -> ...i", velocity, Epar)
    Pperp = np.einsum("...ij,...ij -> ...i", velocity, Eperp)

    return Ppar / Pperp


def pitch_angle(
        u: np.ndarray,
        Bci: Optional[np.ndarray] = None,
        fields: Optional[dict] = None,
        position: Optional[np.ndarray] = None
):
    """
    Get the pitch angle of the particle.

    Arguments
    ---------
    u: np.ndarray (shape: (...,3))
        particle velocities at step n + 1/2

    Bci: Optional[np.ndarray] (shape: (...,3)) (default: None)
        interpolated magnetic field at step n. If None, fields and position must be passed.

    fields: Optional[dict] (default: None)
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS). If None, Bci must be passed.

    position: Optional[np.ndarray] (shape: (...,3)) (default: None)
        particle positions at step n. If None, Bci must be passed.

    Returns
    -------
    pitch_angle: np.ndarray (shape: (...,3))
        pitch angle of the particle

    """
    if (Bci is None) and (fields is None or position is None):
        raise ValueError(
            "Bci has to be passed, or fields and position. No consistent arguments are given.")

    if Bci is None:
        fields_ci = {key: interpolate_field(position, fields[key])
                     for key in ["bx", "by", "bz"]}
        Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])

    return np.arccos(np.einsum("...ij, ...ij -> ...i", u, Bci) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(Bci, axis=-1)))
