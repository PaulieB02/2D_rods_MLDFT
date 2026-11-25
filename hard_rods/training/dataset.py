"""
Training windows (Sec. 3.1): periodically wrapped density windows around random lattice sites,
with the c1 target at their centre. Sites where both targets are infinite (hard walls) are skipped.
"""

import numpy as np
import torch
from numba import njit
from torch.utils.data import Dataset

from ..config import profile_path
from ..utils import packing_fraction


@njit
def generate_training_windows(profile, num_window_bins, random_bools):
    """Periodic windows [n, 2, w, w] around the sites selected by random_bools (row-major)."""
    num_windows = np.sum(random_bools)
    profile_windows = np.zeros((num_windows, 2, num_window_bins, num_window_bins), dtype="float32")
    pad = num_window_bins // 2
    profile_padded = np.zeros((2, profile.shape[1] + 2 * pad, profile.shape[2] + 2 * pad), dtype="float32")

    profile_padded[:, pad:profile.shape[1] + pad, pad:profile.shape[2] + pad] = profile
    if pad > 0:
        profile_padded[:, :pad, pad:profile.shape[2] + pad] = profile[:, -pad:, :]                          # lower rows
        profile_padded[:, pad + profile.shape[1]:, pad:profile.shape[2] + pad] = profile[:, :pad, :]        # upper rows
        profile_padded[:, :, :pad] = profile_padded[:, :, -2 * pad:-pad]                                     # left columns
        profile_padded[:, :, pad + profile.shape[1]:] = profile_padded[:, :, pad:2 * pad]                    # right columns
    idx = 0
    dimprofile0 = profile.shape[2]
    for i in range(profile.shape[1]):
        for j in range(dimprofile0):
            if random_bools[i * dimprofile0 + j]:
                profile_windows[idx, :, :, :] = profile_padded[:, i:i + num_window_bins, j:j + num_window_bins]
                idx += 1
    return profile_windows


def load_training_windows(directory, rho_window_size, c1_window_size, number_of_files,
                          usage_rate, max_packing_fraction, k):
    """Density windows and the matching c1 windows (size 1: the centre value) of the profiles in directory."""
    rho_profiles = []
    c1_profiles = []
    for i in range(number_of_files):
        rho_profiles.append(np.load(profile_path(directory, "rho", i)).astype("float32"))
        c1_profiles.append(np.load(profile_path(directory, "c1", i)).astype("float32"))

    rho_windows_both_species = []
    c1_windows_both_species = []
    for rho, c1 in zip(rho_profiles, c1_profiles):
        if packing_fraction(rho, k) >= max_packing_fraction:
            continue
        random_bools = (np.random.rand((c1.shape[1] * c1.shape[2])) < usage_rate)
        # the same sites for both, so the windows line up index for index
        rho_windows = generate_training_windows(rho, rho_window_size, random_bools)
        c1_windows = generate_training_windows(c1, c1_window_size, random_bools)
        idx = 0
        for i in range(c1.shape[1]):
            for j in range(c1.shape[2]):
                if random_bools[i * c1.shape[2] + j]:
                    c1_finite = np.isfinite(c1[0, i, j]) or np.isfinite(c1[1, i, j])
                    if c1_finite:
                        rho_windows_both_species.append(rho_windows[idx])
                        c1_windows_both_species.append(c1_windows[idx])
                    idx += 1
    rho_windows_all = torch.from_numpy(np.stack(rho_windows_both_species))
    c1_windows_all = torch.from_numpy(np.stack(c1_windows_both_species))
    return rho_windows_all, c1_windows_all


class WindowDataset(Dataset):
    """(density window, c1 target) pairs of one folder of GCMC profiles; usage_rate is the
    fraction of the lattice sites of each profile used as window centres."""

    def __init__(self, directory, number_of_profiles, rho_window_size, c1_window_size,
                 max_packing_fraction, k, usage_rate=0.5):
        self.rho_windows, self.c1_windows = load_training_windows(
            directory, rho_window_size, c1_window_size, number_of_profiles, usage_rate,
            max_packing_fraction, k)
        self.num_items = self.c1_windows.shape[0]

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return self.rho_windows[idx], self.c1_windows[idx]
