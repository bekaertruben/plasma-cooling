from matplotlib import figure, gridspec
import numpy as np
from tqdm import tqdm
import pusher
from constants import *
import initialization as init
from typing import Optional, Union
from warnings import warn


class Simulation():
    def __init__(self, n_cells: int = N_CELLS, iterations: int = int(1e3), cc: float = CC) -> None:
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
              fields: Union[str, dict] = "pic",
              gamma_drag: dict = {"syn": GAMMA_SYN, "ic": GAMMA_IC},
              beta_rec: float = BETA_REC,
              number_of_saves: int = 100) -> None:

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
        self.transferred_power_fraction = np.zeros(
            (self.number_of_saves, self.n_particles))

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
        self.transferred_power_fraction = pusher.transferred_power(
            self.vel_history, fields=self.fields, position=self.pos_history)
        pass

    def end(self) -> None:
        pass


def plot_trajectories(time: np.ndarray, pos: np.ndarray, vel: np.ndarray, fields: np.ndarray) -> figure.Figure:
    linestyles = ["-", "--", "-.", ":"]
    colors = ["k", "c", "m", "y", "r", "g", "b"]

    fig = figure.Figure(figsize=(7, 5))
    gs = gridspec.GridSpec(2, 2, fig)

    axp = fig.add_subplot(gs[0, 0])
    axu = fig.add_subplot(gs[1, 0], sharex=axp)
    axE = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1], sharex=axE)

    for i, x in enumerate(np.swapaxes(pos, 0, -1)):
        for j in range(3):
            axp.plot(time, x[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    for i, u in enumerate(np.swapaxes(vel, 0, -1)):
        for j in range(3):
            axu.plot(time, u[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    fields_ci = {key: pusher.interpolate_field(pos, value)
                 for key, value in fields.items()}
    Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])
    Eci = np.swapaxes(Eci, 0, 1)

    Bci = np.swapaxes(Bci, 0, 1)

    for i, E in enumerate(np.swapaxes(Eci, 0, -1)):
        for j in range(3):
            axE.plot(time, E[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    for i, B in enumerate(np.swapaxes(Bci, 0, -1)):
        for j in range(3):
            axB.plot(time, B[j], ls=linestyles[i %
                     len(linestyles)], color=colors[j], lw=0.5)

    axp.set_ylabel("Position [m]")
    axu.set_ylabel("Velocity [m/s]")
    axE.set_ylabel("Electric Field")
    axB.set_ylabel("Magnetic Field")
    axB.set_xlabel("Time [s]")
    axE.set_xlabel("Time [s]")
    axp.set_xlabel("Time [s]")
    axu.set_xlabel("Time [s]")
    axp.set_title("Particle Trajectories")
    axu.set_title("Particle Velocities")
    axE.set_title("Electric Field")
    axB.set_title("Magnetic Field")

    fig.suptitle("Time evolution (Black: x, Cyan: y, Magenta: z)")

    return fig


def main():
    sim = Simulation(simtime=5*BOXSIZE/C, cc=0.45)
    sim.begin(1, 1e8, number_of_saves=-1)
    sim.run()
    # sim.end()

    fig = plot_trajectories(np.linspace(
        0, sim.simtime, sim.iterations), sim.pos_history, sim.vel_history, sim.fields)

    fig.savefig("images/time_evolution.png", facecolor="white")


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
    main()
    # plot_fields()
