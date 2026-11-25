"""
Grand-canonical Monte Carlo of hard rods on a periodic lattice in an external field (Sec. 3.1).

Insertion and deletion moves with the grand-canonical acceptance rules; the anchor density is
averaged over each run and over independent parallel runs. Species 0 (vertical) occupies the
sites (y, x) ... (y + k - 1, x), species 1 (horizontal) the sites (y, x) ... (y, x + k - 1).
"""

import os

import numpy as np
from numba import njit, prange

from ..config import profile_path


@njit(inline='always')
def overlap_horizontal(y, x, occ, W, k):
    """True if a horizontal rod anchored at (y, x) would overlap an occupied site."""
    end = x + k
    if end <= W:
        for xx in range(x, end):
            if occ[y, xx]:
                return True
    else:
        for xx in range(x, W):
            if occ[y, xx]:
                return True
        wrap = end - W
        for xx in range(wrap):
            if occ[y, xx]:
                return True
    return False


@njit(inline='always')
def overlap_vertical(y, x, occ, H, k):
    """True if a vertical rod anchored at (y, x) would overlap an occupied site."""
    end = y + k
    if end <= H:
        for yy in range(y, end):
            if occ[yy, x]:
                return True
    else:
        for yy in range(y, H):
            if occ[yy, x]:
                return True
        wrap = end - H
        for yy in range(wrap):
            if occ[yy, x]:
                return True
    return False


@njit(boundscheck=False)
def main_loop(grid_height, grid_width, k, n_steps, u_0_1, warmup_steps):
    """One run; returns the anchor density, sampled every 100 steps after warmup_steps, [2, H, W]."""
    final_density = np.zeros((2, grid_height, grid_width))
    u_0 = u_0_1[0]
    u_1 = u_0_1[1]
    meas_step_size = 100    # steps between two recorded configurations
    # how often each site has been the anchor of a vertical (0) / horizontal (1) rod
    rod_counter_type_0 = np.zeros((grid_height, grid_width), dtype=np.uint32)
    rod_counter_type_1 = np.zeros((grid_height, grid_width), dtype=np.uint32)
    # anchor (y, x) and orientation (True: horizontal) of every rod; at most H W / k rods fit
    rods_coordinates = np.zeros((grid_height * grid_width // k, 2), dtype=np.int32)
    rods_orientation = np.zeros((grid_height * grid_width // k), dtype=np.bool_)
    counter = 0             # number of rods in the system
    occin = np.zeros((grid_height, grid_width), dtype=np.bool_)    # occupied sites
    measure_counter = 0

    # 2 H W exp(-u), as it enters both acceptance probabilities
    msquared_times_2 = grid_height * grid_width * 2
    exponentiated_potential_0 = msquared_times_2 * np.exp(-u_0)
    exponentiated_potential_1 = msquared_times_2 * np.exp(-u_1)

    for it in range(n_steps):
        if it % meas_step_size == 0 and it > warmup_steps:
            # record the current configuration
            for n in range(counter):
                if rods_orientation[n]:
                    y, x = rods_coordinates[n]
                    rod_counter_type_1[y, x] += 1
                else:
                    y, x = rods_coordinates[n]
                    rod_counter_type_0[y, x] += 1
            measure_counter += 1

        add_attempt = np.random.random() < 0.5

        # insertion attempt
        if add_attempt:
            ry = np.random.randint(0, grid_height)
            rx = np.random.randint(0, grid_width)
            maybe_horiz = np.random.random() < 0.5
            if not maybe_horiz and not overlap_vertical(ry, rx, occin, grid_height, k) and exponentiated_potential_0[ry, rx] > np.random.random() * (counter + 1):   # vertical
                    rods_coordinates[counter, 0] = ry
                    rods_coordinates[counter, 1] = rx
                    rods_orientation[counter] = False
                    counter += 1
                    for i in range(k):
                        occin[(ry + i) % grid_height, rx] = True
            elif maybe_horiz and not overlap_horizontal(ry, rx, occin, grid_width, k) and exponentiated_potential_1[ry, rx] > np.random.random() * (counter + 1):   # horizontal
                    rods_coordinates[counter, 0] = ry
                    rods_coordinates[counter, 1] = rx
                    rods_orientation[counter] = True
                    counter += 1
                    for i in range(k):
                        occin[ry, (rx + i) % grid_width] = True

        # deletion attempt
        elif not add_attempt and counter > 0:
            index = np.random.randint(0, counter)      # a random rod
            ry = rods_coordinates[index, 0]
            rx = rods_coordinates[index, 1]
            if rods_orientation[index]:                # horizontal
                z = exponentiated_potential_1[ry, rx]
            else:                                      # vertical
                z = exponentiated_potential_0[ry, rx]

            if counter > np.random.random() * z:
                if rods_orientation[index]:
                    for i in range(k):
                        occin[ry, (rx + i) % grid_width] = False
                else:
                    for i in range(k):
                        occin[(ry + i) % grid_height, rx] = False

                # the last rod in the list takes the place of the deleted one
                last_idx = counter - 1
                rods_coordinates[index] = rods_coordinates[last_idx]
                rods_orientation[index] = rods_orientation[last_idx]
                counter -= 1
    final_density[0] = rod_counter_type_0 / measure_counter
    final_density[1] = rod_counter_type_1 / measure_counter
    return final_density


@njit(parallel=True)
def simulate_runs(u_0_1, grid_height, grid_width, n_steps, parallel_simulations, k, warmup_steps):
    """Independent runs in parallel; every run's density separately, [runs, 2, H, W]."""
    all_densities = np.zeros((parallel_simulations, 2, grid_height, grid_width))
    for parallel_run in prange(parallel_simulations):
        all_densities[parallel_run] = main_loop(grid_height, grid_width, k, n_steps, u_0_1,
                                                warmup_steps)
    return all_densities


@njit(parallel=True)
def simulate(u_0_1, grid_height, grid_width, n_steps, parallel_simulations, k):
    """One profile: the density averaged over the runs, its c1 target and its standard error."""
    all_densities = simulate_runs(u_0_1, grid_height, grid_width, n_steps, parallel_simulations,
                                  k, warmup_steps=10 ** 5)
    all_densities_average = np.sum(all_densities, axis=0) / parallel_simulations

    if parallel_simulations < 2:
        raise ValueError("the standard error of the average needs at least two runs")
    mean_reshaped = all_densities_average.reshape(1, 2, grid_height, grid_width)
    diff = all_densities - mean_reshaped
    var_of_mean = np.sum(diff * diff, axis=0) / (parallel_simulations * (parallel_simulations - 1))
    all_densities_sem = np.sqrt(var_of_mean)
    all_c1s = c1_target(all_densities_average, u_0_1)
    return all_densities_average, all_c1s, all_densities_sem


@njit
def c1_target(rho, u):
    """c1 target ln(rho) + u, with rho clipped at 1e-12 (+inf inside hard walls)."""
    rho_safe = np.clip(rho, 1e-12, None)
    return np.log(rho_safe) + u


def run_gcmc_profiles(cfg, split, number_of_profiles):
    """Simulate every potential of the 'training' or 'validation' data; save rho, c1 and rho_sem."""
    directory = cfg.profile_dir(split)
    for quantity in ("rho", "c1", "rho_sem"):
        os.makedirs(os.path.join(directory, quantity), exist_ok=True)

    for i in range(number_of_profiles):
        u_0_1 = np.load(profile_path(directory, "vext", i))
        rho, c1, rho_sem = simulate(u_0_1, cfg.grid_size, cfg.grid_size, cfg.gcmc_steps,
                                    cfg.gcmc_parallel_runs, cfg.k)
        np.save(profile_path(directory, "rho", i), rho)
        np.save(profile_path(directory, "c1", i), c1)
        np.save(profile_path(directory, "rho_sem", i), rho_sem)
        print(f"finished the GCMC simulation of {split} profile {i}")
    print(f"finished the GCMC simulations of the {split} data")
