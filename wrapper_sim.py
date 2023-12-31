from matplotlib import figure, gridspec
import os
import pickle
import numpy as np
from tqdm import tqdm
import pusher
from constants import *
import initialization as init
from typing import Optional, Union
from warnings import warn


class Simulation():
    def __init__(self, n_cells: int = N_CELLS, iterations: int = 1000, cc: float = CC) -> None:
        self.n_cells = n_cells
        if cc > 0.5:
            warn(
                f"Sim velocity of light {cc} is too high (> 0.5), simulation will be unstable")
        self.cc = cc
        self.iterations = iterations
        self.edges_cells = np.array([n_cells, n_cells, n_cells])

    def begin(self,
              n_particles: int,
              temperature: float,
              gamma_drag: dict = {"syn": GAMMA_SYN, "ic": GAMMA_IC},
              number_of_saves: int = 100,
              beta_rec: float = BETA_REC,
              fields: Union[str, dict] = "pic",
              ) -> None:

        if fields == "pic":
            self.fields, self.b_norm = init.load_fields()
        else:
            raise (ValueError("Fields different from `pic` not implemented."))

        self.n_particles = n_particles

        self.positions = init.sample_pos_uniform(n_particles, self.edges_cells)
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
        # self.transferred_power_fraction = np.zeros(
        # (self.number_of_saves, self.n_particles))

        self.gamma_drag = gamma_drag
        self.beta_rec = beta_rec

    def _iteration(self, i: int, save: bool = False) -> None:
        self.positions, self.velocities = pusher.push(
            self.positions, self.velocities, self.fields, self.gamma_drag, self.beta_rec, self.b_norm, self.cc)

        if save:
            self.pos_history[(self.number_of_saves * i) //
                             self.iterations] = self.positions
            self.vel_history[(self.number_of_saves * i) //
                             self.iterations] = self.velocities

    def run(self) -> None:
        for i in tqdm(range(1, self.iterations)):
            self._iteration(
                i, save=(i % (self.iterations // self.number_of_saves) == 0))
        # self.transferred_power_fraction = pusher.transferred_power(
            # self.vel_history, fields=self.fields, position=self.pos_history)
        pass

    to_pickle = ['n_cells', 'cc', 'iterations', 'b_norm',
                 'n_particles', 'number_of_saves', 'gamma_drag', 'beta_rec']

    def _attrmeta(self) -> dict:
        attr = {key: self.__getattribute__(key)
                for key in Simulation.to_pickle}
        return attr

    def end(self, name: str, prefix: str = "pickles") -> None:
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        path = prefix + "/" + name
        if not os.path.exists(path):
            os.mkdir(path)

        with open(path+"/meta.pkl", "wb") as f:
            pickle.dump(self._attrmeta(), f)

        np.save(path+"/xhist.npy", self.pos_history)
        np.save(path+"/uhist.npy", self.vel_history)

    def load(self, name: str, prefix: str = "pickles"):
        path = prefix + "/" + name

        with open(path + "/meta.pkl", "rb") as f:
            attr = pickle.load(f)

        for key, value in attr.items():
            self.__setattr__(key, value)

        self.pos_history = np.load(path + "/xhist.npy")
        self.vel_history = np.load(path + "/uhist.npy")


def main():
    sim = Simulation(iterations=3)
    sim.begin(1, 0.3, number_of_saves=2)
    sim.run()
    sim.end(name="test")


def main2():
    sim = Simulation()
    sim.load("test")
    print(sim.pos_history.shape)


def plot_fields():
    fields, _ = init.load_fields()
    fig = figure.Figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 2)
    axE = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    Eimshow = axE.imshow(fields["ey"][0, :, :], origin="lower")
    Bimshow = axB.imshow(fields["by"][0, :, :], origin="lower")

    fig.colorbar(Eimshow, ax=axE)
    fig.colorbar(Bimshow, ax=axB)

    fig.savefig("images/fields.png", facecolor="white")


def interpolate_fields_test():
    fields, _ = init.load_fields()
    # pos = np.linspace()


if __name__ == '__main__':
    main2()
    # plot_fields()
