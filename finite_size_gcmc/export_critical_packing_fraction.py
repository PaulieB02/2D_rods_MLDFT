#!/usr/bin/env python3
"""
Write the critical packing fraction of the finite-size analysis to a plain text file.

eta_c is the "estimate" of the bootstrap report of peak_uncertainty.py (fit window
-0.25 <= beta*mu <= -0.10). The figures of the main analysis read it from
results/final/critical_packing_fraction.txt; floats are written with repr(), so they are read
back exactly.

    python3 export_critical_packing_fraction.py [REPORT.json] [--out FILE]
"""
import argparse
import json
from pathlib import Path

FINAL = Path(__file__).resolve().parent / 'results' / 'final'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('report', type=Path, nargs='?', default=FINAL / 'peak_bootstrap_window15_block50000.json',
                   help='bootstrap report written by peak_uncertainty.py')
    p.add_argument('--out', type=Path, default=FINAL / 'critical_packing_fraction.txt')
    args = p.parse_args()

    report = json.loads(args.report.read_text())
    estimate, interval = report['estimate'], report['percentile_95']
    mus = report['fitted_mus']
    lines = [
        '# Critical packing fraction of the isotropic-nematic transition of hard rods with k = 8',
        '# on a periodic 100 x 100 lattice, from the finite-size GCMC analysis in finite_size_gcmc/:',
        '# the mean packing fraction at the maximum of Var(|S|), located by a weighted local quadratic',
        f'# fit over {min(mus)} <= beta*mu <= {max(mus)} ({len(mus)} chemical potentials), with a',
        f'# {report["bootstrap"]}-sample bootstrap of {report["block_sweeps"]}-sweep blocks.',
        f'# Written by export_critical_packing_fraction.py from {args.report.name}.',
        '# eta_c_ci95 is the statistical 95% bootstrap interval; with the fit-window sensitivity the',
        '# working uncertainty is about +-0.002 (see RESULTS.md).',
        f'eta_c = {estimate["eta"]!r}',
        f'eta_c_ci95 = {interval["eta"][0]!r} {interval["eta"][1]!r}',
        f'mu_c = {estimate["mu"]!r}',
        f'max_var_abs_S = {estimate["variance"]!r}',
    ]
    args.out.write_text('\n'.join(lines) + '\n')
    print(f'eta_c = {estimate["eta"]!r} written to {args.out}')


if __name__ == '__main__':
    main()
