import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib import figure


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

    return np.sqrt(1 + np.sum(np.square(u), axis=0))


def boris_push(x0: np.ndarray, u0: np.ndarray, fields: dict, dt: float, q_over_m: float):
    """
    Borish Pusher
    -----

    x0: (3,N)
    u0: (3,N)

    fields: dict
        keys must include 'ex','ey','ez','bx','by','bz'
    """

    xci = x0 + u0*dt / (2 * lorentz_factor(u0))

    # interpolate fields if bottleneck could make it multiprocessed
    fields_ci = {key: interpolate_field(xci, value)
                 for key, value in fields.items()}

    # fields_ci[key] is a (N,) shaped array

    # Fci shape: (3,N)
    Eci = np.asarray([fields_ci[key] for key in ["ex", "ey", "ez"]])
    Bci = np.asarray([fields_ci[key] for key in ["bx", "by", "bz"]])

    umin = u0 + q_over_m * dt * Eci / 2

    # == lorentz_factor(uplus) == lorentz_factor(uci) see paper
    g = lorentz_factor(umin)

    t = Bci * q_over_m * dt / (2 * g)

    s = 2*t / (1 + np.linalg.norm(t)**2)

    uplus = umin + \
        np.cross((umin + np.cross(umin, t, axisa=0, axisb=0, axisc=0)),
                 s, axisa=0, axisb=0, axisc=0)

    unext = uplus + q_over_m * dt * Eci / 2
    xnext = xci + unext * dt / (2 * g)

    edges = np.asarray(fields["ex"].shape)

    xnext = apply_periodicity(xnext, edges)

    return xnext, unext


def load_fields(path: str = "data/flds.tot.00410"):
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


def main():
    from tqdm import tqdm
    CC = 0.25
    DT = 1
    DX = DT / CC

    Q_OVER_M = -1

    # fields = load_fields()
    fieldnames = ["ex", "ey", "ez", "bx", "by", "bz"]
    n = 100
    fields = {key: np.zeros((n, n, n)) for key in fieldnames}
    fields["bz"] += 2e-3
    # fields["ex"] += 1
    edges = np.asarray(fields["ex"].shape)

    IT = int(10000*DT)
    # IT = 5
    N = 1
    # x = np.random.rand(3, N) * edges[:, np.newaxis]
    x = np.asarray([[20], [20], [20]])
    # u = np.random.rand(3, N)
    u = np.asarray([[0.00005], [0.], [0.001]])

    x_history = []
    y_history = []
    z_history = []

    for _ in tqdm(range(IT)):
        x, u = boris_push(x, u, fields, DT, Q_OVER_M)

        x_history.append(x[0])
        y_history.append(x[1])
        z_history.append(x[2])

    fig = figure.Figure(figsize=(4, 3.5))
    ax = fig.add_subplot(projection="3d")

    for i in range(N):
        ax.scatter(np.asarray(x_history)[:, i], np.asarray(y_history)[:, i], np.asarray(z_history)[:, i],
                   c=np.arange(IT), cmap="rainbow", s=.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ax.set_xlim([20-0.04, 20+0.04])
    # ax.set_ylim([20-0.01, 20+0.07])
    # ax.set_zlim([20, 40])
    fig.suptitle("purple is early, red is later")

    fig.savefig("images/xytest.png", facecolor="white")


if __name__ == '__main__':
    main()
