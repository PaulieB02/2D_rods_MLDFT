#!/usr/bin/env python3
"""
The full pipeline that produced the data and the networks, stage by stage:

    python run_pipeline.py --k 8 --stages potentials gcmc bulk-gcmc train evaluate

  potentials   random external potentials for the training and validation profiles
  gcmc         GCMC simulation of every potential (validation first, then training)
  bulk-gcmc    GCMC of the homogeneous bulk for the equation of state
  train        train the networks (--networks; default all three)
  evaluate     evaluate the trained networks, as reproduce_paper_figures.py does

Settings come from hard_rods/config.py. The GCMC stages take days and training hours; all
results are included in data/. Stages that would overwrite existing data stop unless
--overwrite is given; --run-dir writes a new run elsewhere. The finite-size GCMC analysis of
the transition is in finite_size_gcmc/.
"""

import argparse
import dataclasses
import os
import sys
import time

from hard_rods.config import ARCHITECTURE_OF, NETWORK_TYPES, RUNS, profile_path
from hard_rods.utils import default_device

STAGES = ("potentials", "gcmc", "bulk-gcmc", "train", "evaluate")


def refuse_to_overwrite(path, stage, overwrite):
    """Stop if path exists, unless overwriting was asked for."""
    if os.path.exists(path) and not overwrite:
        sys.exit(f"stage '{stage}': {path} already exists. Pass --overwrite to replace the "
                 f"existing data, or --run-dir to write a new run elsewhere.")


def create_run_folders(cfg):
    """The folder layout of a run directory (see hard_rods/config.py)."""
    for split in ("training", "validation"):
        for quantity in ("vext", "rho", "rho_sem", "c1"):
            os.makedirs(os.path.join(cfg.profile_dir(split), quantity), exist_ok=True)
    os.makedirs(os.path.join(cfg.run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(cfg.run_dir, "bulk"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True, choices=sorted(RUNS), help="rod length")
    parser.add_argument("--stages", nargs="+", required=True, choices=STAGES)
    parser.add_argument("--networks", nargs="+", choices=NETWORK_TYPES, default=list(NETWORK_TYPES),
                        help="networks to train (default: all three)")
    parser.add_argument("--run-dir", default=None, help="run directory (default: data/k<k>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing existing simulation data and networks")
    parser.add_argument("--device", default=None, help="torch device (default: cuda, mps or cpu)")
    args = parser.parse_args()

    cfg = RUNS[args.k]
    if args.run_dir is not None:
        cfg = dataclasses.replace(cfg, run_dir=os.path.abspath(args.run_dir))
    device = args.device or default_device()
    create_run_folders(cfg)
    print(f"run directory {cfg.run_dir} (k = {cfg.k}, {cfg.grid_size} x {cfg.grid_size})")

    if "potentials" in args.stages:
        from hard_rods.data_generation.external_potentials import save_potentials
        refuse_to_overwrite(profile_path(cfg.profile_dir("validation"), "vext", 0), "potentials",
                            args.overwrite)
        save_potentials(cfg, "training", cfg.n_training)
        save_potentials(cfg, "validation", cfg.n_validation)

    if "gcmc" in args.stages:
        from hard_rods.data_generation.gcmc import run_gcmc_profiles
        refuse_to_overwrite(profile_path(cfg.profile_dir("validation"), "rho", 0), "gcmc",
                            args.overwrite)
        start = time.time()
        run_gcmc_profiles(cfg, "validation", cfg.n_validation)
        run_gcmc_profiles(cfg, "training", cfg.n_training)
        print(f"finished the GCMC simulations in {time.time() - start:.0f} s")

    if "bulk-gcmc" in args.stages:
        from hard_rods.data_generation.bulk_gcmc import eos_gcmc_path, run_bulk_gcmc_scan
        refuse_to_overwrite(eos_gcmc_path(cfg), "bulk-gcmc", args.overwrite)
        start = time.time()
        run_bulk_gcmc_scan(cfg)
        print(f"finished the bulk GCMC scan in {time.time() - start:.0f} s")

    if "train" in args.stages:
        import torch
        from hard_rods.functionals.networks import ARCHITECTURES
        from hard_rods.training.train import train_model
        for network_type in args.networks:
            refuse_to_overwrite(cfg.model_path(network_type), "train", args.overwrite)
        for network_type in args.networks:
            print(f"training the {network_type} network ...")
            architecture = ARCHITECTURES[ARCHITECTURE_OF[network_type]]
            net = architecture(cfg.window_size(network_type)).to(device)
            train_model(net, network_type, cfg, device)
            # the padding is not part of the state dict; restored for completeness
            net.enable_padding()
            torch.save(net.state_dict(), cfg.model_path(network_type))
            print(f"saved {cfg.model_path(network_type)}")

    if "evaluate" in args.stages:
        from reproduce_paper_figures import evaluate
        evaluate(cfg, device)


if __name__ == "__main__":
    main()
