# Neural density functional for two-dimensional lattice hard rods

This repository contains code, data and trained networks accompanying

**A Neural Density Functional for Spatial Structure and Orientational Ordering in Two-Dimensional Lattice Hard Rods**
*Paul Bitzer, Jens Weimar and Martin Oettel, [arxiv:2611.19553 [cond-mat.stat-mech]](https://arxiv.org/abs/2611.19553)*


Three convolutional networks learn the one-body direct correlation function c1 of hard rods of
length *k* = 8 and *k* = 4 on a periodic square lattice from grand-canonical Monte Carlo (GCMC)
density profiles: the Φ-network, the direct-c<sup>(1)</sup> network and the direct-c<sup>(1)</sup> network
with the mixed-derivative penalty. The Lafuente–Cuesta lattice
functional serves as the analytical reference.

## Setup

```bash
git clone https://github.com/PaulieB02/2D_rods_MLDFT.git
cd 2D_rods_MLDFT
```

With [uv](https://docs.astral.sh/uv/), `uv sync` installs Python 3.13 and the exact package
versions of `uv.lock`. Without uv, create a Python 3.13 environment, run
`pip install -r requirements.txt` and use `python` instead of `uv run` below. The evaluation
runs on CUDA or, much slower, on the CPU.

## Instructions

```bash
uv run reproduce_paper_figures.py --plots-only   # redraw every figure from the stored data (seconds)
uv run reproduce_paper_figures.py                # evaluate the trained networks, then redraw (~1 h on a GPU)
```

Nothing is simulated or trained: the script evaluates the trained networks (`data/k*/models/`)
on the stored GCMC data and writes the figures to `figures/`:

| figure | article | computed by `hard_rods/evaluation/` | drawn by `hard_rods/plotting/` |
|---|---|---|---|
| `k8_profile_deviations.pdf`, `k4_…` | Sec. 4.1, App. A | `validation_profiles.py` | `profile_errors.py` |
| `k8_phi_profile003.pdf` | Sec. 4.1 | `validation_profiles.py` | `profile_map.py` |
| `k8_equation_of_state.pdf`, `k4_…` | Sec. 4.2, App. A | `equation_of_state.py` | `equation_of_state.py` |
| `k8_free_energy_landscapes.pdf` | Sec. 4.3 | `landscapes.py` | `landscapes.py` |
| `k8_order_parameter_two_routes.pdf` | Sec. 4.4 | `phase_diagram.py` | `order_parameter.py` |
| `k8_free_energy_derivatives_vs_eta.pdf` | Sec. 4.5 | `landau.py` | `landau.py` |
| `k8_c2_symmetry_histogram.pdf` | App. B | `c2_symmetry.py` | `c2_symmetry.py` |
| `k8_network_comparison_c1.pdf`, `k4_…` | not in the article | `validation_profiles.py` | `profile_errors.py` |

`run_pipeline.py` contains the rest of the workflow: random external potentials, GCMC of the
profiles and of the bulk, training and evaluation (`--help` lists the stages; the GCMC takes days
on a many-core machine). It does not overwrite existing data unless `--overwrite` is given.
Potentials and GCMC are not seeded, so a new run reproduces the data statistically, not bit for
bit. The *k* = 8 and *k* = 4 runs were made on different computers with partly different
settings; all of them are recorded in `hard_rods/config.py`.

The critical packing fraction of the isotropic–nematic transition comes from a finite-size GCMC
analysis, which has its own README in [`finite_size_gcmc/`](finite_size_gcmc/README.md).

## Data

`data/k8` (100 × 100 lattice) and `data/k4` (50 × 50) contain the trained networks (`models/`),
the GCMC validation profiles (`validation_data/`), the evaluated validation profiles of every
functional (`evaluation/`) and the tables of the homogeneous system (`bulk/`, each described in
the module that writes it). Profiles are stored as one `.npy` file per profile and quantity,
arrays `[species, y, x]` with species 0 vertical and species 1 horizontal rods: `vext`
(u = β(V − μ), +inf at hard walls), `rho` (density of rod anchors, i.e. of the lower or left
ends, averaged over independent GCMC runs), `rho_sem` (its standard error) and `c1` (the
training target ln ρ + u). The packing fraction is η = *k*(ρ₀ + ρ₁), averaged over all sites.

The GCMC training profiles (about 1.25 GB), needed only for retraining, are attached to the
release `v1.0` as `training_data.tar.gz`. Download and extract it in the repository root:

```bash
curl -LO https://github.com/PaulieB02/2D_rods_MLDFT/releases/download/v1.0/training_data.tar.gz
tar xzf training_data.tar.gz    # creates data/k8/training_data/ and data/k4/training_data/
rm training_data.tar.gz
```

## Reproducibility

`--plots-only` redraws the figures identically to the article. A full re-evaluation reproduces
the stored data exactly for the direct-c1 networks and LC, but not for the Φ-network: its c1
comes from a GPU backward pass, which with PyTorch's default (fastest) cuDNN routines varies by
about 10⁻⁷ between runs. Deterministic mode (`torch.backends.cudnn.deterministic = True`) would
make runs on one machine repeatable at about 20% more time; it was off for the stored data and
is left off. The Picard iteration amplifies the differences, slightly for converged profiles and
strongly for the 10 Φ-network profiles of *k* = 8 that do not converge within 10⁴ steps (in one
of three re-runs, one of their errors came out four times larger). The Φ-network's values in
the c2 histogram are round-off themselves, so a re-evaluation also changes that histogram's bins.

## Citation

Please cite the article above. Code and data are released under the MIT licence (`LICENSE`).
