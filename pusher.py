import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib import figure, gridspec
import os
from typing import Optional, Union
from initialization import *
from tqdm import tqdm

from constants import *


def apply_periodicity(X0: np.ndarray, edges: np.ndarray):
    """
    X0: (3,N)

    edges: (3,)
        assumes boundaries of [(0,edge) for edge in edges]
    """

    return np.mod(X0, edges[:, np.newaxis])


def interpolate_field(x0: np.ndarray, F: np.ndarray):
    """
    Interpolate F at x0

    Arguments
    ---------
    x0: np.ndarray (shape: (*,3,N))
        positions at which to interpolate F. Optional zeroth axis for multiple sets of positions (different sim iterations)

    F: np.ndarray (shape: (N_CELLS, N_CELLS, N_CELLS))
        field to interpolate

    edges: np.ndarray (shape: (3,))
        edges of the simulation box in meters

    Returns
    -------
    F_interp: np.ndarray (shape: (*,N))
        interpolated field at x0

    """
    if len(x0.shape) == 3:
        # (3,*,N), if no copy a view is returned
        x = np.array(np.swapaxes(x0, 0, 1), copy=True)
        # weird bug I dont understand: if the below two lines are not seperated the results are wrong
    elif len(x0.shape) == 2:
        x = np.array(x0)
    else:
        raise ValueError(f"Invalid shape for x0 {x0.shape}")

    F_interp = map_coordinates(F, x, order=1, mode='grid-wrap')
    return F_interp


def lorentz_factor(u: np.ndarray):
    """
    Compute the lorentz factor of particles with velocity u.

    Arguments
    ---------
    u: np.ndarray (shape: (*,3,N))
        particle velocities. Dimension -2 is spatial

    Returns
    -------
    gamma: np.ndarray (shape: (*,N))
        lorentz factor
    """

    return np.sqrt(1 + np.sum(np.square(u), axis=-2))


def boris_push(x0: np.ndarray, u0: np.ndarray, fields: dict, bnorm: float, cc: float = CC):
    """
    Borish push (Lorentz force) on particles with velocity u0 and position x0.

    Arguments
    ---------
    x0: np.ndarray (shape: (3,N))
        particle positions at step n

    u0: np.ndarray (shape: (3,N))
        particle velocities at step n - 1/2

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS)

    Returns
    -------
    xnext: np.ndarray (shape: (3,N))
        particle positions at step n + 1

    unext: np.ndarray (shape: (3,N))
        particle velocities at step n + 1/2

    Eci: np.ndarray (shape: (3,N))
        interpolated electric field at step n

    Bci: np.ndarray (shape: (3,N))
        interpolated magnetic field at step n

    """
    if set(fields.keys()) != set(FIELDNAMES):
        raise KeyError(
            f"`fields` keys ({fields.keys()}) does not corresponds with {FIELDNAMES}")

    xci = x0 + u0 / (2 * lorentz_factor(u0))

    # interpolate fields. if bottleneck could make it multiprocessed
    fields_ci = {key: interpolate_field(xci, value)
                 for key, value in fields.items()}

    # fields_ci[key] is a (N,) shaped array

    # Fci shape: (3,N)
    Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])

    dummy = 0.5 * Q_OVER_M * bnorm
    E0 = Eci * dummy
    # dummy /= cc
    B0 = Bci * dummy

    # half acceleration
    u1prime = cc * u0 + E0

    # first half magnetic rotation
    gamma1 = lorentz_factor(u1prime / cc)
    f = 2. / (1. + np.sum(np.square(B0/(cc * gamma1)), axis=0))
    u2prime = (u1prime + np.cross(u1prime/(cc * gamma1), B0, axis=0))*f

    # second half magnetic rotation + half acceleration
    u3prime = u1prime + np.cross(u2prime/(cc * gamma1), B0, axis=0) + E0
    unext = u3prime / cc
    xnext = xci + unext / (2 * lorentz_factor(unext))

    xnext = apply_periodicity(xnext, np.array(EDGES_CELLS))

    return xnext, unext, Eci, Bci


def radiate_synchrotron(u0: np.ndarray,
                        u1: np.ndarray,
                        Eci: np.ndarray,
                        Bci: np.ndarray,
                        Bnorm: float,
                        beta_rec: float,
                        gamma_syn: float,
                        cc: float = CC):
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

    beta_rec: float
        fiducial magnetic energy extraction rate

    gamma_syn: float
        typical lorentz factor for synchrotron drag

    Returns
    -------
    unext: np.ndarray (shape: (3,N))
        synchrotron dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)
    betaci = uci/gci

    Ebar = Eci + np.cross(betaci, Bci, axis=0)

    beta_dot_e = np.einsum("ji,ji->i", betaci, Eci)

    kappa_R = np.cross(Ebar, Bci, axis=0) + beta_dot_e * Eci
    chi_R_sq = np.abs(np.sum(np.square(Ebar), axis=0) - np.square(beta_dot_e))

    prefactor = Bnorm * beta_rec / (cc * gamma_syn**2)

    unext = u0 + prefactor * (kappa_R - gci * chi_R_sq * uci)
    gafter = lorentz_factor(unext)
    return unext


def radiate_inversecompton(u0: np.ndarray, u1: np.ndarray, Bnorm: float, beta_rec: float, gamma_ic: float, cc: float = CC):
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

    beta_rec: float
        fiducial magnetic energy extraction rate

    gamma_ic: float
        typical lorentz factor for inverse compton drag

    Returns
    -------
    unext: np.ndarray (shape: (3,N))
        inverse compton dragged particle velocities at step n + 1/2

    """
    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)

    prefactor = Bnorm * beta_rec / (cc * gamma_ic**2)

    unext = u0 - prefactor * uci * gci
    gafter = lorentz_factor(unext)
    return unext


def push(x0: np.ndarray,
         u0: np.ndarray,
         fields: dict,
         gamma_drag: dict,
         beta_rec: float,
         Bnorm: Optional[float] = None,
         cc: float = CC):
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

    gamma_drag: dict
        dictionary with keys "syn" and "ic" and values of the typical lorentz factors for synchrotron and inverse compton drag. If a key is missing, the corresponding drag is not applied.

    edges_meter: np.ndarray (shape: (3,))
        edges of the simulation box in meters

    Bnorm: Optional[float]
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

    xnext, u_lorentz, Eci, Bci = boris_push(x0, u0, fields, Bnorm, cc)

    syn_drag = "syn" in gamma_drag.keys()
    ic_drag = "ic" in gamma_drag.keys()

    if not syn_drag and not ic_drag:
        return xnext, u_lorentz

    if syn_drag:
        u_syn = radiate_synchrotron(
            u0, u_lorentz, Eci, Bci, Bnorm, beta_rec, gamma_drag["syn"], cc)

    if ic_drag:
        u_ic = radiate_inversecompton(
            u0, u_lorentz, Bnorm, beta_rec, gamma_drag["ic"], cc)

    if syn_drag and not ic_drag:
        unext = u_lorentz + u_syn - u0
        return xnext, unext

    elif ic_drag and not syn_drag:
        unext = u_lorentz + u_ic - u0
        return xnext, unext

    unext = u_lorentz + u_syn + u_ic - 2 * u0
    return xnext, unext


def transferred_power(velocity: np.ndarray,
                      Eci: Optional[np.ndarray] = None,
                      Bci: Optional[np.ndarray] = None,
                      fields: Optional[dict] = None,
                      position: Optional[np.ndarray] = None):
    """
    Get the parallel and perpendicular power transferred from field to particle.

    Arguments
    ---------
    charge: float or list[float]
        particle charge in Coulomb

    velocity: np.ndarray (shape: (**, 3, N))
        particle velocities at step n + 1/2

    Eci: Optional[np.ndarray] (shape: (**, 3, N)) (default: None)
        interpolated electric field at steps n. If None, fields and position must be passed.

    Bci: Optional[np.ndarray] (shape: (**, 3, N)) (default: None)
        interpolated magnetic field at steps n. If None, fields and position must be passed.

    fields: Optional[dict] (default: None)
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS).
        If None, Eci and Bci must be passed.

    position: Optional[np.ndarray] (shape: (**, 3, N)) (default: None)
        particle positions at steps n. If None, Eci and Bci must be passed.

    Returns
    -------
    Ppar / Pperp: np.ndarray (shape: (**, N))
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
    #     np.linalg.norm(Bci, axis=len(Bci.shape)-2) ** 2
    normalization = np.linalg.norm(Bci, axis=-2) ** 2
    Epar = np.einsum("...ji,...ji,...ki->...ki", Eci, Bci, Bci)
    Epar /= normalization[:, np.newaxis, :]
    Eperp = Eci - Epar

    Ppar = np.einsum("...ji,...ji -> ...i", velocity, Epar)
    Pperp = np.einsum("...ji,...ji -> ...i", velocity, Eperp)

    return Ppar / Pperp


def pitch_angle(u: np.ndarray, Bci: Optional[np.ndarray] = None, fields: Optional[dict] = None, position: Optional[np.ndarray] = None):
    """
    Get the pitch angle of the particle.

    Arguments
    ---------
    u: np.ndarray (shape: (**,3,N))
        particle velocities at step n + 1/2

    Bci: Optional[np.ndarray] (shape: (**,3,N)) (default: None)
        interpolated magnetic field at step n. If None, fields and position must be passed.

    fields: Optional[dict] (default: None)
        keys must include 'ex','ey','ez','bx','by','bz'. The values are 3D arrays of shape (N_CELLS, N_CELLS, N_CELLS). If None, Bci must be passed.

    position: Optional[np.ndarray] (shape: (**,3,N)) (default: None)
        particle positions at step n. If None, Bci must be passed.

    Returns
    -------
    pitch_angle: np.ndarray (shape: (**,N))
        pitch angle of the particle

    """
    if (Bci is None) and (fields is None or position is None):
        raise ValueError(
            "Bci has to be passed, or fields and position. No consistent arguments are given.")

    if Bci is None:
        fields_ci = {key: interpolate_field(position, fields[key])
                     for key in ["bx", "by", "bz"]}
        Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])

    return np.arccos(np.einsum("...ji, ...ji -> ...i", u, Bci) / (np.linalg.norm(u, axis=-2) * np.linalg.norm(Bci, axis=-2)))
