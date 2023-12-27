import numpy as np
import pusher
from constants import *
import initialization as init
from typing import Optional, Union
from warnings import warn


class Simulation():
    def __init__(self, boxsize: float = BOXSIZE, n_cells: int = N_CELLS, simtime: Optional[float] = None) -> None:
        self.boxsize = boxsize
        self.n_cells = n_cells
        self.dx = boxsize / (4*n_cells)
        self.dt = CC * self.dx / C
        if simtime is None:
            self.simtime = 5 * boxsize / C
        else:
            self.simtime = simtime
        self.iterations = int(self.simtime/self.dt)
        self.edges_cells = np.array([n_cells, n_cells, n_cells])
        self.edges_meter = np.ones(3) * boxsize

    def begin(self,
              n_particles: int,
              temperature: float,
              fields: Union[str, dict] = "pic",
              gamma_drag: dict = {"syn": GAMMA_SYN, "ic": GAMMA_IC},
              beta_rec: float = BETA_REC) -> None:

        if fields == "pic":
            self.fields, self.b_norm = init.load_fields()
        else:
            self.fields = fields
            self.b_norm = np.mean(fields["bz"])
        if temperature > 1e8:
            warn(
                f"Temperature {temperature} is too high (> 1e8), particles will break relativity")

        self.n_particles = n_particles

        self.positions = init.sample_pos_uniform(n_particles, self.edges_meter)
        self.velocities = init.sample_velocity_thermal(
            n_particles, temperature)

        self.pos_history = np.zeros((self.iterations, 3, self.n_particles))
        self.vel_history = np.zeros((self.iterations, 3, self.n_particles))

        self.pos_history[0] = self.positions
        self.vel_history[0] = self.velocities

        self.Ek = np.zeros((self.iterations, self.n_particles))
        self.transferred_power = {'par': np.zeros(
            (self.iterations, self.n_particles)), 'perp': np.zeros((self.iterations, self.n_particles))}

        self.gamma_drag = gamma_drag
        self.beta_rec = beta_rec

    def _iteration(self, i: int) -> None:
        self.positions, self.velocities = pusher.push(
            self.positions, self.velocities, self.fields, self.gamma_drag, self.dt, self.edges_meter, self.beta_rec, self.b_norm)
        self.pos_history[i] = self.positions
        self.vel_history[i] = self.velocities

    def _diagnose(self, i: int) -> None:
        pass
