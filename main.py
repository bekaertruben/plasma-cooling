from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use("ggplot")

N = 10_000
iterations = 1000
saves = 100

sim = Simulation(
    N=N,
    T=0.3,
    fields=Fields.from_file(),
    parameters=SimulationParameters(
        gamma_syn=3.,
        gamma_ic=3.,
        cc=0.45,
        particle_lifetime=5,
    ),
)

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