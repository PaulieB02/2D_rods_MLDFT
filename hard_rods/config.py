"""
Settings of the k = 8 and k = 4 runs, and the repository layout.

The two data sets were generated and trained on different computers with different settings;
each RunConfig records the values used for its rod length.

Conventions: species 0 are vertical and species 1 horizontal rods; fields are arrays
[2, H, W] (species, y, x); the stored external potential is u_i = beta (V_i - mu), +inf at
hard walls; c1_i = -d(beta F_ex)/d rho_i. The data layout is described in the README.
"""

import dataclasses
import os
from typing import Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURE_DIR = os.path.join(REPO_ROOT, "figures")
FINITE_SIZE_GCMC_DIR = os.path.join(REPO_ROOT, "finite_size_gcmc")

# "c1_regularised" is the direct-c1 architecture trained with the mixed-derivative penalty
NETWORK_TYPES = ("phi", "c1", "c1_regularised")

# architecture of each network type (names in functionals.networks.ARCHITECTURES)
ARCHITECTURE_OF = {"phi": "Sarpedon", "c1": "Diomedes", "c1_regularised": "Diomedes"}

# chemical potentials beta*mu of the bulk equation of state
EOS_MU_VALUES = np.linspace(-5.0, 2, 50)


@dataclasses.dataclass(frozen=True)
class TrainingSettings:
    """Hyperparameters of one network's training."""
    num_epochs: int
    batch_size: int
    learning_rate: float


@dataclasses.dataclass(frozen=True)
class PicardSettings:
    """Settings of evaluation.picard.picard_minimise."""
    alpha: float                  # initial mixing parameter
    maxiter: int
    tol: float                    # on the mean squared residual over the allowed sites
    S_initial: Optional[float] = None   # initial order parameter (bulk scans)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """All settings of one run (one rod length)."""
    k: int                        # rod length
    grid_size: int                # periodic box of grid_size x grid_size sites
    run_dir: str                  # data directory of this run

    # training and validation data
    n_training: int               # training profiles generated (before the eta filter)
    n_validation: int             # validation profiles on disk
    gcmc_steps: int               # Monte Carlo steps per run
    gcmc_parallel_runs: int       # independent runs averaged per profile
    with_spots: bool = True       # add plateaus to the random potentials
    unique_potentials: bool = True  # independent potentials for the two species
    mu: Optional[float] = None    # fixed beta*mu; None draws it at random for every profile

    # training
    max_packing_fraction_training: float = 0.7     # profiles with eta >= this are not used
    training: dict = dataclasses.field(default_factory=dict)   # network type -> TrainingSettings

    # evaluation of the validation profiles
    picard_validation: PicardSettings = PicardSettings(alpha=1e-7, maxiter=10000, tol=1e-10)
    max_packing_fraction_comparison: float = 0.8   # profiles above this are left out of the error plots

    # bulk equation of state: GCMC, and Picard for the networks and for LC
    eos_gcmc_grid_size: int = 100
    eos_gcmc_steps: int = 10 ** 10
    eos_gcmc_parallel_runs: int = 26
    picard_eos: PicardSettings = PicardSettings(alpha=0.05, maxiter=10000, tol=1e-10, S_initial=0.1)
    picard_eos_lc: PicardSettings = PicardSettings(alpha=0.05, maxiter=100000, tol=1e-10,
                                                   S_initial=0.1)

    # GCMC critical packing fraction, written by finite_size_gcmc/export_critical_packing_fraction.py;
    # None for a rod length without a transition
    eta_c_file: Optional[str] = None

    def window_size(self, network_type):
        """Convolution window: k + 1 for the Phi-network, 2k + 1 for the direct-c1 network."""
        return self.k + 1 if network_type == "phi" else 2 * self.k + 1

    def profile_dir(self, split):
        """Folder of the "training" or "validation" profiles."""
        if split not in ("training", "validation"):
            raise ValueError(f"split must be 'training' or 'validation', not {split!r}")
        return os.path.join(self.run_dir, f"{split}_data")

    def model_path(self, network_type):
        return os.path.join(self.run_dir, "models", f"{network_type}_network.pth")

    def evaluation_dir(self, functional):
        return os.path.join(self.run_dir, "evaluation", functional)

    def bulk_path(self, filename):
        return os.path.join(self.run_dir, "bulk", filename)

    def tensorboard_dir(self, network_type):
        return os.path.join(self.run_dir, "tensorboard", network_type)


def profile_path(directory, quantity, index):
    """File of one profile, e.g. profile_path(cfg.profile_dir("validation"), "rho", 3)."""
    return os.path.join(directory, quantity, f"{quantity}_{index:03d}.npy")


def read_critical_packing_fraction(path):
    """eta_c from a file written by finite_size_gcmc/export_critical_packing_fraction.py."""
    with open(path) as handle:
        for line in handle:
            key, _, value = line.partition("=")
            if key.strip() == "eta_c" and not line.lstrip().startswith("#"):
                return float(value)
    raise ValueError(f"no 'eta_c = ...' line in {path}")


K8 = RunConfig(
    k=8,
    grid_size=100,
    run_dir=os.path.join(DATA_DIR, "k8"),
    n_training=1000,          # 872 have eta < 0.7
    n_validation=100,
    gcmc_steps=10 ** 10,
    gcmc_parallel_runs=28,
    training={
        "phi": TrainingSettings(num_epochs=200, batch_size=1024, learning_rate=1e-3),
        "c1": TrainingSettings(num_epochs=200, batch_size=512, learning_rate=1e-4),
        "c1_regularised": TrainingSettings(num_epochs=100, batch_size=128, learning_rate=1e-4),
    },
    eos_gcmc_grid_size=100,
    eos_gcmc_steps=10 ** 10,
    eos_gcmc_parallel_runs=26,
    picard_eos=PicardSettings(alpha=0.05, maxiter=10000, tol=1e-10, S_initial=0.1),
    eta_c_file=os.path.join(FINITE_SIZE_GCMC_DIR, "results", "final",
                            "critical_packing_fraction.txt"),
)

K4 = RunConfig(
    k=4,
    grid_size=50,
    run_dir=os.path.join(DATA_DIR, "k4"),
    n_training=4000,          # 3492 have eta < 0.7
    n_validation=100,
    gcmc_steps=int(0.25 * 10 ** 10),
    gcmc_parallel_runs=22,
    training={
        "phi": TrainingSettings(num_epochs=200, batch_size=1024, learning_rate=1e-3),
        "c1": TrainingSettings(num_epochs=200, batch_size=128, learning_rate=1e-4),
        "c1_regularised": TrainingSettings(num_epochs=100, batch_size=128, learning_rate=1e-4),
    },
    eos_gcmc_grid_size=50,
    eos_gcmc_steps=int(0.25 * 10 ** 10),
    eos_gcmc_parallel_runs=22,
    # alpha = 0.005 for the networks; the stored LC scan used picard_eos_lc (alpha = 0.05)
    picard_eos=PicardSettings(alpha=0.005, maxiter=10000, tol=1e-10, S_initial=0.1),
    eta_c_file=None,
)

RUNS = {8: K8, 4: K4}
