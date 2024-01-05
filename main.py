from simulation import Simulation
from fields import Fields
from simulation_parameters import SimulationParameters

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use("ggplot")

N = 1000
iterations = 10_000
saves = 100

sim = Simulation(
    N=N,
    T=10,
    # fields = Fields.uniform_fields(np.array([100, 100, 100]), B0 = np.array([0, 0, 1])),
    fields=Fields.from_file(),
    parameters=SimulationParameters(
        gamma_syn=None, gamma_ic=10., cc=0.45),
)

x_hist = np.zeros((saves, N, 3))
u_hist = np.zeros((saves, N, 3))

for i, positions, velocities in tqdm(sim.run(iterations, saves), total=saves, desc="Running simulation"):
    x_hist[i] = positions
    u_hist[i] = velocities

# # Plot a particle trajectory:
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='3d')
# ax.set_title("Particle trajectory [red -> purple]")

# c = np.linspace(1, 0, saves)
# for i in range(saves):
#     ax.scatter(x_hist[i, 0, 0], x_hist[i, 0, 1], x_hist[i, 0, 2], color=cm.rainbow(c[i]))
# plt.legend()
# plt.show()

# Plot the spread of particle velocities:
fig = plt.figure(figsize=(10, 5))
t = np.linspace(0, iterations, saves)
us = np.linalg.norm(u_hist, axis=-1)
plt.errorbar(t, us.mean(axis=-1), us.std(axis=-1), fmt='o', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Particle velocity spread")
plt.show()