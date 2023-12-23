import numpy as np
C = 3e8  # meter/sec
CC = 0.45
BOXSIZE = 1e8  # meter
# nx = 160 downsampled from 4*160
N_CELLS = 160
DX = BOXSIZE / (4*N_CELLS)
DT = CC * DX / C
T = 5 * BOXSIZE / C  # time to simulate in seconds
ITERATIONS = int(T/DT)
EDGES_CELLS = np.array([N_CELLS, N_CELLS, N_CELLS])
EDGES_METER = np.ones(3) * BOXSIZE


Q_OVER_M = -1
