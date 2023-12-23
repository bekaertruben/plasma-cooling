from constants import *
import numpy as np
from typing import Optional
from boris_pusher import lorentz_factor


# TODO: verify difference between relativistic velocity and regular (u = \gamma v). We are very much using velocity I believe
def radiate_synchrotron(u0: np.ndarray, u1: np.ndarray, Eci: np.ndarray, Bci: np.ndarray, Bnorm: Optional[float] = None):
    if Bnorm is None:
        Bnorm = 1.

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


def radiate_inversecompton(u0: np.ndarray, u1: np.ndarray, Eci: np.ndarray, Bci: np.ndarray, Bnorm: Optional[float] = None):
    if Bnorm is None:
        Bnorm = 1.

    uci = 0.5 * (u0 + u1)
    gci = lorentz_factor(uci)
    betaci = uci/gci

    prefactor = - np.abs(Q_OVER_M) * Bnorm * BETA_REC / (C * GAMMA_IC**2)

    unext = u0 + DT * prefactor * betaci * gci ** 2
    return unext
