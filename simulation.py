import numpy as np
import pandas as pd
import h5py
import os
from typing import Optional

from simulation_parameters import SimulationParameters
from maxwell_juttner import MaxwellJuttnerDistribution
from fields import Fields
import pusher


# Load precomputed Maxwell-Jüttner gamma samples
if os.path.exists('data/MJ_gammas.csv'):
    MJ_gammas = pd.read_csv('data/MJ_gammas.csv', index_col=0)
else:
    MJ_gammas = pd.DataFrame()


class Simulation():
    """ Wrapper class for the simulation.

    Attributes
    ----------
    N : int
        Number of particles.
    T : float
        Initial temperature of the particles.
    parameters : SimulationParameters
        Simulation parameters.
    fields : dict
        Dictionary containing the fields.
        Keys are the names of the fields, values are numpy arrays of shape (3, N).
    positions : numpy.ndarray
        Positions of the particles (3, N) at the current iteration.
    velocities : numpy.ndarray
        Proper velocities of the particles (3, N) at the current iteration.

    Methods
    -------
    generate_particles():
        Generate `N` particles with temperature `T` and add them to the simulation.
    step()
        Perform one iteration of the simulation.
    run(steps: int, num_sapshots: int):
        Generator running the simulation for `steps` iterations and yielding the positions and velocities every iteration.
        If `num_snapshots` is not None, it will instead return the positions and velocities only that many times.
    """

    N: int
    T: float
    parameters: SimulationParameters
    fields: Fields
    positions: np.ndarray
    velocities: np.ndarray

    def __init__(self, 
            N: int,
            T: float,
            parameters: SimulationParameters,
            fields: Optional[Fields] = None,
            skip_particle_generation: bool = False
    ) -> None:
        self.N = N
        self.T = T
        self.parameters = parameters
        if fields:
            self.fields = fields
        else:
            self.fields = Fields.uniform_fields(self.parameters.edges_cells)

        if not skip_particle_generation:
            self.generate_particles()

    def generate_particles(self) -> None:
        """ Generate `N` particles with temperature `T` and add them to the simulation. """
        # Sample particle positions uniformly over the simulation space
        self.positions = np.random.rand(
            self.N, 3) * self.parameters.edges_cells[np.newaxis, :]

        # Sample Lorentz factors from a thermal Maxwell-Jüttner distribution
        idx = f'T={self.T}'
        if idx in MJ_gammas.index and self.N <= MJ_gammas.loc[idx].size:
            gammas = MJ_gammas.loc[idx].sample(self.N).values
        else:
            mj = MaxwellJuttnerDistribution(
                T=self.T,
                # Approximation only works for large temperatures
                approximation_order=1 if self.T > 100 else None
            )
            gammas = mj.sample(self.N)
        us = np.sqrt(gammas**2 - 1)

        # Sample directions from uniform spherical distribution
        directions = np.random.randn(self.N, 3)
        directions /= np.linalg.norm(directions, axis=-1)[:, np.newaxis]

        self.velocities = us[:, np.newaxis] * directions

    def step(self) -> None:
        """ Perform one iteration of the simulation. """
        self.positions, self.velocities = pusher.push(
            self.positions,
            self.velocities,
            self.fields,
            self.parameters
        )

    def run(self, steps: int, num_snapshots: Optional[int] = None):
        """ Generator running the simulation for `steps` iterations and yielding the positions and velocities every iteration.
        If `num_snapshots` is not None, it will instead return the positions and velocities only that many times.
        """
        if num_snapshots:
            snapshots = np.linspace(0, steps-1, num_snapshots, dtype=int)
        else:
            snapshots = np.arange(steps)

        for i in range(steps):
            self.step()
            if i in snapshots:
                nth_snapshot = np.where(snapshots == i)
                yield nth_snapshot, self.positions, self.velocities

    def save(self, path: str) -> None:
        """ Save the simulation to a file. """
        with h5py.File(path, 'w') as file:
            file.create_dataset("positions", data=self.positions)
            file.create_dataset("velocities", data=self.velocities)
            file.create_dataset("fields/Bnorm", data=self.fields.Bnorm)
            file.create_dataset("fields/E", data=self.fields.E)
            file.create_dataset("fields/B", data=self.fields.B)
            file.create_dataset("parameters", data=self.parameters.as_array)

    @classmethod
    def load(cls, path: str):
        """ Load a simulation from a file. """
        with h5py.File(path, 'r') as file:
            positions = file['positions'][()]
            velocities = file['velocities'][()]

            edges_cells = np.asarray(positions.shape[:-1])
            Bnorm = file['fields/Bnorm'][()]
            E = file['fields/E'][()]
            B = file['fields/B'][()]

            parameters = file['parameters'][()]

        sim = cls(
            N=positions.shape[0],
            T=0,
            parameters=SimulationParameters(*parameters),
            fields=Fields(edges_cells, E, B, Bnorm),
            skip_particle_generation=True
        )
        sim.positions = positions
        sim.velocities = velocities

        return sim
