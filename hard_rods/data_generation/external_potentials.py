"""
Random external potentials of the training and validation profiles (Appendix A).

Each species gets 2-5 random sinusoids plus random plateaus (ellipses or rectangles, 10% of
them hard walls) and, with probability 20% each, a hard strip along two box edges. The
chemical potential is common to both species. Saved: u_i = beta (V_i - mu), [2, H, W].
"""

import os

import numpy as np
from numba import njit

from ..config import profile_path


@njit
def _sinusoid(x, y, system_height, system_width, A, phase, wave_vector_0, wave_vector_1):
    """One sinusoidal term of the potential at site (y, x)."""
    return A * np.sin(2 * np.pi * (x * wave_vector_1 / system_width + y * wave_vector_0 / system_height) + phase)


@njit
def _plateau(x, y, system_height, system_width, plateau_type, a, b, height, x_centre, y_centre):
    """Value of one plateau at site (y, x): 0 is an ellipse, 1 a rectangle (periodic distances)."""
    dx = np.abs(x - x_centre)
    dy = np.abs(y - y_centre)
    dx = np.minimum(dx, system_width - dx)
    dy = np.minimum(dy, system_height - dy)
    if plateau_type == 0:   # ellipse
        if (dx / a)**2 + (dy / b)**2 < 1:
            return height
        else:
            return 0
    if plateau_type == 1:   # rectangle
        if dx < a and dy < b:
            return height
        else:
            return 0


@njit
def _potential_at(y, x, sin_parameters, plateau_parameters):
    """The full potential (all sinusoids and plateaus) at site (y, x)."""
    num_sin = np.shape(sin_parameters)[0]
    num_plateaus = np.shape(plateau_parameters)[0]
    result = 0
    for i in range(num_sin):
        result += _sinusoid(x, y, sin_parameters[i, 0], sin_parameters[i, 1], sin_parameters[i, 2],
                            sin_parameters[i, 3], sin_parameters[i, 4], sin_parameters[i, 5])
    for i in range(num_plateaus):
        result += _plateau(x, y, plateau_parameters[i, 0], plateau_parameters[i, 1],
                           plateau_parameters[i, 2], plateau_parameters[i, 3],
                           plateau_parameters[i, 4], plateau_parameters[i, 5],
                           plateau_parameters[i, 6], plateau_parameters[i, 7])
    return result


def random_potential(k, system_height, system_width, with_spots):
    """One random potential beta V of a single species, [H, W], without the chemical potential."""
    num_sin = np.random.randint(2, 6)
    if with_spots:
        if k >= 6:
            num_plateaus = np.random.randint(50, 70)
        else:
            num_plateaus = np.random.randint(10, 15)
    else:
        num_plateaus = 0
    sin_parameters = np.zeros((num_sin, 6))
    plateau_parameters = np.zeros((num_plateaus, 8))

    for n in range(num_sin):
        A = np.random.uniform(0.2, 1)               # amplitude
        phase = np.random.uniform() * 2 * np.pi     # phase
        if k >= 6:
            q_max = 10
        if k == 4:
            q_max = 20
        wave_vector = np.random.randint(-q_max, q_max, 2)
        sin_parameters[n, :] = np.array([system_height, system_width, A, phase,
                                         wave_vector[0], wave_vector[1]])

    for n in range(num_plateaus):
        a, b = np.random.uniform(1, 6, 2)           # semi-axes / half-widths
        plateau_type = np.random.randint(2)         # 0: ellipse, 1: rectangle
        wall_or_normal = np.random.uniform()
        if wall_or_normal > 0.9:                    # hard wall
            height = np.inf
        else:                                       # plateau of moderate height
            if np.random.uniform() > 0.65:
                height = np.random.uniform(0.2, 2)
            else:
                height = np.random.uniform(0.2, 1)
                height = height * -1
        x_centre = np.random.randint(system_width)
        y_centre = np.random.randint(system_height)
        plateau_parameters[n, :] = np.array([system_height, system_width, plateau_type, a, b,
                                             height, x_centre, y_centre])
    potential = np.zeros((system_height, system_width))
    for i in range(system_height):
        for j in range(system_width):
            potential[i, j] = _potential_at(i, j, sin_parameters, plateau_parameters)

    # hard strips of width 8 along two box edges, each with probability 20%
    border = np.random.uniform(0, 1)
    if border > 0.8:
        potential[0:8, :] = np.inf
    border = np.random.uniform(0, 1)
    if border > 0.8:
        potential[:, 0:8] = np.inf

    return potential


def random_potential_pair(k, system_height, system_width, unique_potential, with_spots, mu=None):
    """(u_0, u_1) of one profile; mu=None draws the chemical potential at random."""
    if mu is None:
        if k >= 6:
            mu = np.random.uniform(-5, 1)
        else:
            mu = np.random.uniform(-5, 1.3)

    if unique_potential:
        u_0 = random_potential(k, system_height, system_width, with_spots) - mu
        u_1 = random_potential(k, system_height, system_width, with_spots) - mu
        return u_0, u_1
    else:
        u = random_potential(k, system_height, system_width, with_spots) - mu
        return u, u


def save_potentials(cfg, split, number_of_potentials):
    """Save number_of_potentials random potentials to vext/ of the 'training' or 'validation' data."""
    directory = cfg.profile_dir(split)
    os.makedirs(os.path.join(directory, "vext"), exist_ok=True)
    for i in range(number_of_potentials):
        u_0_1 = random_potential_pair(cfg.k, cfg.grid_size, cfg.grid_size, cfg.unique_potentials,
                                      cfg.with_spots, cfg.mu)
        np.save(profile_path(directory, "vext", i), u_0_1)
    print(f"created {number_of_potentials} {split} potentials in {directory}")
