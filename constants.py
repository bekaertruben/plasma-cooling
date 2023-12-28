import numpy as np
from scipy.constants import electron_mass as ELECTRON_MASS
from scipy.constants import Boltzmann as KB
from scipy.constants import elementary_charge as E_CHARGE

R_E = 2.81794e-15  # meter (classical electron radius)
C = np.double(2.99792458 * 1e8)  # meter/sec
BOXSIZE = 1e8  # meter
CC = 0.45

# nx = 160 downsampled from 4*160
N_CELLS = 160

DX = BOXSIZE / (4*N_CELLS)
DT = CC * DX / C
T = 5 * BOXSIZE / C  # time to simulate in seconds
ITERATIONS = int(T/DT)

EDGES_CELLS = np.array([N_CELLS, N_CELLS, N_CELLS])
EDGES_METER = np.ones(3) * BOXSIZE

Q_OVER_M = -1

# TODO: the bottom two parameters are linked, should verify if we can define them independently (see tristan wiki: radiation)
GAMMA_SYN = 10.  # synchrotron cooling rate
GAMMA_IC = 10.  # synchrotron cooling rate
BETA_REC = 1.0  # fiducial magnetic energy extraction rate

FIELDNAMES = ["ex", "ey", "ez", "bx", "by", "bz"]
