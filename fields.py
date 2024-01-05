import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
import h5py
from typing import Optional


class Fields:
    """ Contains all the field information.

    Attributes
    ----------
    edges_cells : numpy.ndarray
        Gives the shape of the simulation space in number of cells.
    E, B : numpy.ndarray
        The electric and magnetic fields. The shape is (*edges_cells, 3).
    Bnorm : float
        The average magnetic field strength, by which all fields are normalized.
    """

    def __init__(self,
                edges_cells: np.ndarray,
                E : np.ndarray,
                B : np.ndarray,
                Bnorm: float,
            ) -> None:
        self.edges_cells = edges_cells
        self.E = E
        self.B = B
        self.Bnorm = Bnorm
        
        # The wrapped fields are needed for interpolation
        self._E_wrapped = np.pad(E, ((0, 1), (0, 1), (0, 1), (0,0)), mode='wrap') 
        self._B_wrapped = np.pad(B, ((0, 1), (0, 1), (0, 1), (0,0)), mode='wrap')
    
    @classmethod
    def uniform_fields(cls, edges_cells: np.ndarray, E0: np.ndarray = np.zeros(3), B0: np.ndarray = np.zeros(3)):
        """ Creates a uniform fields object """
        assert E0.shape == (3,), "E0 must be a 3D vector"
        assert B0.shape == (3,), "B0 must be a 3D vector"
        assert edges_cells.shape == (3,) and edges_cells.dtype == int, "edges_cells must be an array of 3 integers"
        B_norm = B0[-1]
        if B_norm != 0:
            E = np.ones([*edges_cells, 3]) * E0 / B_norm
            B = np.ones([*edges_cells, 3]) * B0 / B_norm
        else:
            B_norm = 1
            E = np.ones([*edges_cells, 3]) * E0
            B = np.ones([*edges_cells, 3]) * B0
        return cls(edges_cells, E, B, B_norm)
    
    @classmethod
    def from_file(cls, path: str = "data/flds.tot.00410"):
        """ Loads the turbulent fields provided by Daniel """
        with h5py.File(path, 'r') as file:
            ex = np.array(file["/ex"]).T
            ey = np.array(file["/ey"]).T
            ez = np.array(file["/ez"]).T
            bx = np.array(file["/bx"]).T
            by = np.array(file["/by"]).T
            bz = np.array(file["/bz"]).T

        Bnorm = np.mean(bz)

        E = np.moveaxis([ex, ey, ez], 0, -1) / Bnorm
        B = np.moveaxis([bx, by, bz], 0, -1) / Bnorm

        return cls(ex.shape, E, B, Bnorm)

    def interpolate(self, positions: np.ndarray, wrap=True) -> np.ndarray:
        """ Interpolates the fields at the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            Positions of the particles (..., 3).
        wrap : bool
            Whether to wrap the positions around the simulation space.

        Returns
        -------
        E_interp, B_interp : numpy.ndarray
            Interpolated electric and magnetic fields (..., 3).
        """
        positions = np.asarray(positions)
        if positions.shape == (3,):
            positions = positions[np.newaxis, :]
        assert positions.shape[-1] == 3, "Positions must be a (..., 3) array"

        if wrap:
            positions = np.mod(positions, self.edges_cells)
        else:
            assert np.all(positions >= 0) and np.all(positions < self.edges_cells), "Positions must be within the simulation space"

        range_x = np.arange(self.edges_cells[0] + 1)
        range_y = np.arange(self.edges_cells[1] + 1)
        range_z = np.arange(self.edges_cells[2] + 1)

        E_interpolator = RegularGridInterpolator((range_x, range_y, range_z), self._E_wrapped)
        B_interpolator = RegularGridInterpolator((range_x, range_y, range_z), self._B_wrapped)

        return E_interpolator(positions), B_interpolator(positions)


if __name__ == "__main__":
    """ A simple test of the fields class. """
    edges_cells = np.array([100, 100, 100])
    fields = Fields.uniform_fields(edges_cells, np.array([1, 0, 0]), np.array([0, 0, 1]))
    fields.E[0, 0] = [0, 0, 0]
    Ei, Bi = fields.interpolate([-0.5, 0, 0], wrap=True)
    print(Ei) # Should be [[0.5, 0, 0]]