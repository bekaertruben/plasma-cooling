import numpy as np
import h5py
from scipy import ndimage

# Read the 3D turbulence electric and magnetic field data:
filename = "data/flds.tot.00410"
prec = "float32"
f = h5py.File(filename, 'r')
# By default the dimensions are stored as (z, y, x) so tranpose the data:
ex = np.array(f["/ex"], dtype=prec).T
ey = np.array(f["/ey"], dtype=prec).T
ez = np.array(f["/ez"], dtype=prec).T
bx = np.array(f["/bx"], dtype=prec).T
by = np.array(f["/by"], dtype=prec).T
bz = np.array(f["/bz"], dtype=prec).T
f.close()

(nx, ny, nz) = ex.shape    # the number of grid points in x, y, z


#
# Some notes and tips:
#
#
# The simulation domain is a cube with nx = ny = nz.
# The boundary conditions are periodic in all 3 directions! Particle motion needs
# to take this into account.
#
# For simulation units check the Tristan-MP PIC code wiki page. Note that the
# E & B data is multiplited by "Bnorm" before being saved to file. If you use
# the code's implementation of the Boris algorithm for guidance, omit the step
# where E & B are rescaled by Bborm because this was already done when the data
# was written to the "flds.tot.00410" file.
#
# For consistency it is best to adopt the normalizations and units of Tristan-MP.
# Check the code's wiki page and code on github for a convenient normalization of
# the radiative drag force and the definition of the "cool_gamma_syn" and "cool_gamma_ic"
# parameters that quantify the strength of synchrotron and inverse-Compton
# cooling, respectively. For that part, you do need to know the value of "Bnorm".
# In the provided data "Bnorm" equals the mean of bz (i.e., Bborm = np.mean(bz)).
#
#
# Note also that the data stored in the file has been downsampled by a factor of 4 in each
# dimension. That is, the original spacing between the xyz grid points was 4 times smaller than
# in the provided data file and the original data size was 4*nx * 4*ny * 4*nz.
#
# For reference, the "numerical speed of light" used in the Tristan-MP simulation (check wiki)
# was CC = 0.45. The CC param effectively controls the integration time step. A smaller
# CC means that light travels a shorter distance between two time steps. You can use CC=0.45
# as a reference value for your numerical integration algorithm. You can also measure time
# in units of the box light crossing time, which in the default units
# amounts to 4 * nx / CC time steps (factor 4 because of the downsampling).
#
# Use electrons as particles so the "q_over_m" parameter (see code wiki and github) is -1.
#
#


# Distribute particles initially uniformly over the box. For example:
N = 50000
x = np.random.rand(N) * nx
y = np.random.rand(N) * ny
z = np.random.rand(N) * nz
print(nx, ny, nz)

# Tip: use map_coordinates() function from scipy to interpolate
# the E and B fields to the particle positions to find the
# Loretz force on each particle at every step:
ex0 = ndimage.map_coordinates(ex, [x, y, z], order=1)
ey0 = ndimage.map_coordinates(ey, [x, y, z], order=1)
ez0 = ndimage.map_coordinates(ez, [x, y, z], order=1)
