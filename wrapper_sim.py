import numpy as np
from tqdm import tqdm
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
              beta_rec: float = BETA_REC,
              number_of_saves: int = 100) -> None:

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
        if number_of_saves == -1:
            self.number_of_saves = self.iterations
        else:
            self.number_of_saves = number_of_saves

        self.pos_history = np.zeros(
            (self.number_of_saves, 3, self.n_particles))
        self.vel_history = np.zeros(
            (self.number_of_saves, 3, self.n_particles))

        self.pos_history[0] = self.positions
        self.vel_history[0] = self.velocities

        self.Ek = np.zeros((self.number_of_saves, self.n_particles))
        self.transferred_power = {'par': np.zeros(
            (self.number_of_saves, self.n_particles)), 'perp': np.zeros((self.number_of_saves, self.n_particles))}

        self.gamma_drag = gamma_drag
        self.beta_rec = beta_rec

    def _iteration(self, i: int, save: bool = False) -> None:
        self.positions, self.velocities = pusher.push(
            self.positions, self.velocities, self.fields, self.gamma_drag, self.dt, self.edges_meter, self.beta_rec, self.b_norm)

        if save:
            self.pos_history[(self.number_of_saves * i) //
                             self.iterations] = self.positions
            self.vel_history[(self.number_of_saves * i) //
                             self.iterations] = self.velocities

    def run(self) -> None:
        for i in tqdm(range(1, self.iterations)):
            self._iteration(
                i, save=(i % (self.iterations // self.number_of_saves) == 0))
        self.Ek = pusher.kinetic_energy(self.vel_history)
        self.transferred_power["par"], self.transferred_power["perp"] = pusher.transferred_power(
            E_CHARGE, self.vel_history, fields=self.fields, position=self.pos_history)

    def end(self) -> None:
        pass


def main():
    sim = Simulation(simtime=BOXSIZE/C)
    sim.begin(3, 1e8, number_of_saves=-1)
    sim.run()
    sim.end()


if __name__ == '__main__':
    main()
