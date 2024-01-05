import numpy as np
from dataclasses import dataclass
from typing import Optional
from warnings import warn


@dataclass
class SimulationParameters:
    """ Contains all the information about the simulation parameters.

    Attributes
    ----------
    n_cells : int
        Number of cells in each direction.
        The edges_cells property is [n_cells, n_cells, n_cells].
        (This is for generality, in the future we may want to have different number of cells in each direction)
    cc : float
        Numerical speed of light.
    q_over_m : float
        Charge to mass ratio.
    gamma_syn : Optional[float]
        Synchrotron drag coefficient. If set to None, no synchrotron drag is applied.
    gamma_ic : Optional[float]
        Inverse compton drag coefficient. If set to None, no inverse compton drag is applied.
    beta_rec : float
        Fiducial magnetic energy extraction rate.
    """

    n_cells: int = 160
    cc: float = 0.45
    q_over_m: float = -1.
    gamma_syn: Optional[float] = 10.
    gamma_ic: Optional[float] = 10.
    beta_rec: float = 0.1 # TODO: check if this may be independently defined from gamma_syn and gamma_ic

    def __post_init__(self):
        if self.cc > 0.5:
            warn(f"Numerical velocity of light (CC={self.cc}) is too high (> 0.5), simulation will be unstable")

    @property
    def edges_cells(self):
        return np.array([self.n_cells, self.n_cells, self.n_cells])


if __name__ == "__main__":
    print(SimulationParameters(cc=0.9))