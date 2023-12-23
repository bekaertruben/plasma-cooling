import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib import figure, gridspec
import os
from typing import Optional
from tqdm import tqdm

from constants import *


def apply_periodicity(X0: np.ndarray, edges: np.ndarray):
    """
    X0: (N,3)

    edges: (3,)
        assumes boundaries of [(0,edge) for edge in edges]
    """

    return np.mod(X0, edges[:, np.newaxis])


def interpolate_field(x0: np.ndarray, F: np.ndarray):
    """
    x0: (3,N)
    F: (nx,ny,nz)
    """
    return map_coordinates(F, x0, order=1, mode='grid-wrap')


def lorentz_factor(u: np.ndarray):
    """
    u: (3,N)
    """

    return np.sqrt(1 + np.sum(np.square(u)/C**2, axis=0))


def boris_push(x0: np.ndarray, u0: np.ndarray, fields: dict):
    """
    Borish Pusher
    -----

    x0: (3,N)
    u0: (3,N)

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'
    """
    if set(fields.keys()) != set(FIELDNAMES):
        raise KeyError(
            f"`fields` keys ({fields.keys()}) does not corresponds with {FIELDNAMES}")

    xci = x0 + u0*DT / (2 * lorentz_factor(u0))

    # interpolate fields. if bottleneck could make it multiprocessed
    fields_ci = {key: interpolate_field(xci, value)
                 for key, value in fields.items()}

    # fields_ci[key] is a (N,) shaped array

    # Fci shape: (3,N)
    Eci = np.array([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.array([fields_ci[key] for key in ["bx", "by", "bz"]])

    umin = u0 + Q_OVER_M * DT * Eci / 2

    # == lorentz_factor(uplus) == lorentz_factor(uci) see paper
    g = lorentz_factor(umin)

    t = Bci * Q_OVER_M * DT / (2 * g)

    s = 2*t / (1 + np.linalg.norm(t)**2)

    uplus = umin + \
        np.cross((umin + np.cross(umin, t, axis=0)),
                 s, axis=0)

    unext = uplus + Q_OVER_M * DT * Eci / 2
    xnext = xci + unext * DT / (2 * g)

    xnext = apply_periodicity(xnext, EDGES_METER)

    return xnext, unext


def load_fields(path: str = "data/flds.tot.00410"):
    "from Daniel"
    import h5py
    prec = "float32"
    f = h5py.File(path, 'r')
    ex = np.array(f["/ex"], dtype=prec).T
    ey = np.array(f["/ey"], dtype=prec).T
    ez = np.array(f["/ez"], dtype=prec).T
    bx = np.array(f["/bx"], dtype=prec).T
    by = np.array(f["/by"], dtype=prec).T
    bz = np.array(f["/bz"], dtype=prec).T
    f.close()

    fields = {
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "bx": bx,
        "by": by,
        "bz": bz
    }

    return fields


def uniform_B(bdir: str = "z", val: Optional[float] = None):
    directions = ["x", "y", "z"]
    if bdir not in directions:
        raise ValueError(f"Direction {bdir} not in {directions}")
    fields = {key: np.zeros((N_CELLS, N_CELLS, N_CELLS)) for key in FIELDNAMES}
    if val is not None:
        fields[f"b{bdir}"] += val
    else:
        fields[f"b{bdir}"] += 50
    return fields


def init_random_x(N: int):
    x = np.random.rand(3, N) * EDGES_METER[:, np.newaxis]
    return x


def init_random_u(N: int, set_c: Optional[float] = None):
    u = np.random.rand(3, N)  # initialize direction vectors
    u /= np.linalg.norm(u, axis=0)  # normalize them to norm 1

    if set_c is not None:
        return u*set_c

    return u * (np.random.rand(N) * C)[:, np.newaxis]


def kinetic_energy(u_history: list[np.array], mass: float = 1.0):
    Ek = []
    for u in tqdm(u_history):
        Ek.append(mass * lorentz_factor(u)*C**2)
    return Ek


def Rangeframe(*P: list[tuple[np.ndarray, np.ndarray]], args: dict = {}, scatter: bool = True):
    fig = figure.Figure()
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    for i in range(len(P)):
        data = P[i]
        pl = ax.plot(data[0], data[1], zorder=1, label=args)
        if scatter:
            ax.scatter(data[0], data[1], s=64, zorder=2, color='white')
            ax.scatter(data[0], data[1], s=8, zorder=3,
                       color=pl[0].get_color())
        try:
            pl[0].set_label(args['plotlabel'][i])
        except:
            pass
    try:
        ax.set_xlabel(args['xlabel'])
    except:
        pass
    try:
        ax.set_ylabel(args['ylabel'])
    except:
        pass
    try:
        ax.set_title(args['axtitle'])
    except:
        pass
    try:
        fig.suptitle(args['suptitle'])
    except:
        pass
    return fig, gs, ax


def main():

    # fields = load_fields()
    fields = uniform_B()

    N_PARTICLES = 1
    # x = init_random_x(N_PARTICLES)
    # u = init_random_u(N_PARTICLES)
    # manual
    x = np.asarray([[5], [5], [5]])*1e7

    u = init_random_u(N_PARTICLES, 0.999*C)
    print(u)

    x_history = []
    y_history = []
    z_history = []

    u_history = []

    for _ in tqdm(range(ITERATIONS)):
        x, u = boris_push(x, u, fields)

        x_history.append(x[0])
        y_history.append(x[1])
        z_history.append(x[2])
        u_history.append(u)

    # fig = plt.figure(figsize=(4, 3.5))
    fig = figure.Figure(figsize=(4, 3.5))
    ax = fig.add_subplot(projection="3d")

    for i in range(N_PARTICLES):
        ax.scatter(np.asarray(x_history)[:, i], np.asarray(y_history)[:, i], np.asarray(z_history)[:, i],
                   c=np.arange(ITERATIONS), cmap="rainbow", s=.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_xlim([0, BOXSIZE])
    ax.set_ylim([0, BOXSIZE])
    ax.set_zlim([0, BOXSIZE])
    fig.suptitle("Particle trajectory (purple is early, red is later)")

    if not os.path.exists("images"):
        os.mkdir("images")

    fig.savefig("images/test.png", facecolor="white")
    Ek = np.array(kinetic_energy(u_history), dtype=float)
    print(f"sum absoulute diff of Ek {np.sum(np.abs(np.diff(Ek)))}")
    return fig


if __name__ == '__main__':
    main()
