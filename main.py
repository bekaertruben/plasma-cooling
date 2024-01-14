from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters

import numpy as np

N = 10_000
iterations = 1000
saves = 100
simulation_box_size = 100

sim = Simulation(
    N=N,
    T=0.3,
    fields=Fields.uniform_fields(
        edges_cells=np.array([simulation_box_size, simulation_box_size, simulation_box_size]),
        E0=np.array([0., 0., 0.]),
        B0=np.array([0., 0., 1.]),
    ),
    parameters=SimulationParameters(
        gamma_syn=20.,
        gamma_ic=20.,
        cc=0.45,
        particle_lifetime=5,
        n_cells=simulation_box_size,
    ),
)

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.style.use("ggplot")

    x_hist = np.zeros((saves, N, 3))
    u_hist = np.zeros((saves, N, 3))

    for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
        x_hist[i] = positions
        u_hist[i] = velocities

    fig = plt.figure(figsize=(10, 5))
    t = np.linspace(0, iterations, saves)

    ux_drift = u_hist[..., 0].mean(axis=1)
    uy_drift = u_hist[..., 1].mean(axis=1)
    uz_drift = u_hist[..., 2].mean(axis=1)

    plt.plot(t, ux_drift, label="$u_x$")
    plt.plot(t, uy_drift, label="$u_y$")
    plt.plot(t, uz_drift, label="$u_z$")

    plt.xlabel("Time")
    plt.ylabel("Drift velocity")
    plt.legend()
    plt.show()