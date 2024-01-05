import numpy as np
import h5py
from typing import Optional

from simulation_parameters import SimulationParameters
from maxwell_juttner import MaxwellJuttnerDistribution
from fields import Fields
import pusher


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

    def __init__(self, N: int, T: float, parameters: SimulationParameters, fields: Optional[Fields] = None) -> None:
        self.N = N
        self.T = T
        self.parameters = parameters
        if fields:
            self.fields = fields
        else:
            self.fields = Fields.uniform_fields(self.parameters.edges_cells)
        
        self.generate_particles()
    
    def generate_particles(self) -> None:
        """ Generate `N` particles with temperature `T` and add them to the simulation. """
        # Sample particle positions uniformly over the simulation space
        self.positions = np.random.rand(self.N, 3) * self.parameters.edges_cells[np.newaxis, :]

        # Sample Lorentz factors from a thermal Maxwell-Jüttner distribution
        mj = MaxwellJuttnerDistribution(
            T = self.T,
            approximation_order = 1 if self.T > 100 else None # Approximation only works for large temperatures
        )
        gammas = mj.sample(self.N)
        us = np.sqrt(gammas**2 - 1)

        # Sample directions from uniform spherical distribution
        phi = np.random.uniform(0, np.pi*2, size=self.N)
        costheta = np.random.uniform(-1, 1, size=self.N)
        theta = np.arccos(costheta)
        directions = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ]).T

        self.velocities = us[:, np.newaxis] * directions

    def step(self) -> None:
        """ Perform one iteration of the simulation. """
        self.positions, self.velocities = pusher.push(
            self.positions,
            self.velocities,
            self.fields,
            self.parameters
        )

    def run(self, steps: int, num_snapshots: int = None):
        """ Generator running the simulation for `steps` iterations and yielding the positions and velocities every iteration.
        If `num_snapshots` is not None, it will instead return the positions and velocities only that many times.
        """
        if num_snapshots:
            snapshots = np.linspace(0, steps, num_snapshots, dtype=int)
        else:
            snapshots = np.arange(steps)

        for i in range(steps):
            self.step()
            if i in snapshots:
                yield i, self.positions, self.velocities
        
    def save(self, path: str) -> None:
        """ Save the simulation to a file. """
        with h5py.File(path, 'w') as file:
            file.create_dataset("positions", data=self.positions)
            file.create_dataset("velocities", data=self.velocities)
            file.create_dataset("fields", data=self.fields)
            file.create_dataset("parameters", data=self.parameters)



if __name__ == "__main__":
    from tqdm import tqdm

    sim = Simulation(
        N = 100,
        T = 1,
        fields = Fields.uniform_fields(np.array([100, 100, 100]), B0 = np.array([0, 0, 1])),
        # fields = Fields.from_file(),
        parameters = SimulationParameters(gamma_syn = None, gamma_ic = None, cc = 0.45),
    )

    sim.positions[0] = [50, 50, 50]
    sim.velocities[0] = [10, 0, 1]

    g = pusher.lorentz_factor(sim.velocities)

    iterations = 100
    x_hist = np.zeros((iterations+1, 3))
    x_hist[0] = sim.positions[0]
    for i, positions, velocities in tqdm(sim.run(iterations), total=iterations, desc="Running simulation"):
        x_hist[i+1] = positions[0]
    
    print(x_hist[0], x_hist[-1])

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.style.use("ggplot")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_title("Particle trajectory [red -> purple]")

    c = np.linspace(1, 0, iterations+1)
    for i in range(iterations):
        ax.plot(x_hist[i:i+2, 0], x_hist[i:i+2, 1], x_hist[i:i+2, 2], color=cm.rainbow(c[i]))

    plt.legend()
    plt.show()