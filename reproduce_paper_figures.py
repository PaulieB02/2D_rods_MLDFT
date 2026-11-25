#!/usr/bin/env python3
"""
Evaluate the trained networks and redraw every result figure of the article into figures/.

    python reproduce_paper_figures.py                evaluate, then draw (about 1 h on one GPU)
    python reproduce_paper_figures.py --plots-only   draw from the evaluation data on disk
    python reproduce_paper_figures.py --k 8          only the k = 8 system

Nothing is simulated or trained: the GCMC data, the trained networks and the finite-size GCMC
result are read from data/ and finite_size_gcmc/. The evaluation overwrites data/k*/evaluation/
and the evaluated tables in data/k*/bulk/. The README lists the figures and explains why the
Phi-network results are reproduced only up to GPU round-off.
"""

import argparse
import time

from hard_rods.config import FIGURE_DIR, RUNS
from hard_rods.utils import default_device


def evaluate(cfg, device):
    """Every network evaluation the figures of the run cfg need."""
    from hard_rods.evaluation.equation_of_state import (lafuente_cuesta_equation_of_state,
                                                        network_equation_of_state)
    from hard_rods.evaluation.validation_profiles import evaluate_lafuente_cuesta, evaluate_network

    print(f"\n===== k = {cfg.k}: validation profiles")
    for network_type in ("c1", "phi", "c1_regularised"):
        evaluate_network(cfg, network_type, device)
    evaluate_lafuente_cuesta(cfg, device)

    print(f"\n===== k = {cfg.k}: bulk equation of state")
    network_equation_of_state(cfg, device)
    lafuente_cuesta_equation_of_state(cfg, device)

    if cfg.k == 8:
        # bulk phase behaviour: k = 8 only
        from hard_rods.evaluation.c2_symmetry import compute_c2_losses
        from hard_rods.evaluation.landau import compute_landau_analysis
        from hard_rods.evaluation.landscapes import compute_landscapes
        from hard_rods.evaluation.phase_diagram import first_derivative_diagram, global_minimum_diagram

        print(f"\n===== k = {cfg.k}: phase diagram")
        global_minimum_diagram(cfg, device)
        first_derivative_diagram(cfg, device)
        print(f"\n===== k = {cfg.k}: free-energy landscapes")
        compute_landscapes(cfg, device)
        print(f"\n===== k = {cfg.k}: Landau coefficients")
        compute_landau_analysis(cfg, device)
        print(f"\n===== k = {cfg.k}: c2 symmetry")
        compute_c2_losses(cfg, device)


def draw(cfg, figure_dir):
    """Every figure of the run cfg. Returns the paths written."""
    from hard_rods.plotting.equation_of_state import plot_equation_of_state
    from hard_rods.plotting.profile_errors import plot_profile_errors

    print(f"\n===== k = {cfg.k}: figures")
    paths = [plot_profile_errors(cfg, figure_dir, quantity="rho"),
             plot_profile_errors(cfg, figure_dir, quantity="c1"),
             plot_equation_of_state(cfg, figure_dir)]
    if cfg.k == 8:
        from hard_rods.plotting.c2_symmetry import plot_c2_histogram
        from hard_rods.plotting.landau import plot_landau
        from hard_rods.plotting.landscapes import plot_landscapes
        from hard_rods.plotting.order_parameter import plot_order_parameter
        from hard_rods.plotting.profile_map import plot_profile_map

        paths += [plot_profile_map(cfg, figure_dir, profile=3, network_type="phi"),
                  plot_landscapes(cfg, figure_dir),
                  plot_order_parameter(cfg, figure_dir),
                  plot_landau(cfg, figure_dir),
                  plot_c2_histogram(cfg, figure_dir)]
    return [p for p in paths if p is not None]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, nargs="+", choices=sorted(RUNS), default=[8, 4],
                        help="rod lengths to process (default: 8 4)")
    parser.add_argument("--plots-only", action="store_true",
                        help="skip the evaluation and draw from the data on disk")
    parser.add_argument("--device", default=None, help="torch device (default: cuda, mps or cpu)")
    parser.add_argument("--figure-dir", default=FIGURE_DIR, help="output folder of the figures")
    args = parser.parse_args()

    device = args.device or default_device()
    written = []
    for k in args.k:
        cfg = RUNS[k]
        if not args.plots_only:
            start = time.time()
            evaluate(cfg, device)
            print(f"\nk = {k}: evaluation took {(time.time() - start) / 60:.1f} min")
        written += draw(cfg, args.figure_dir)

    print(f"\n{len(written)} figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
