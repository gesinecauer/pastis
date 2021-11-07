import numpy as np


def get_new_bead(random_state, distance=1., noise=0.1, prev_bead=None):
    if prev_bead is None:
        prev_bead = np.zeros(3)
    dis_sq = distance ** 2
    x_sq = random_state.uniform(0, dis_sq)
    x = np.sqrt(x_sq) * random_state.choice([-1, 1])
    y_sq = random_state.uniform(0, dis_sq - x_sq)
    y = np.sqrt(y_sq) * random_state.choice([-1, 1])
    z_sq = dis_sq - x_sq - y_sq
    z = np.sqrt(z_sq) * random_state.choice([-1, 1])
    coord_diffs = np.array([x, y, z])
    random_state.shuffle(coord_diffs)
    coord_noise = noise * random_state.randn(*(1, 3))
    return coord_diffs + coord_noise + prev_bead

