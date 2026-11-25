# Finite-size GCMC reference

Supporting simulations for the isotropic–nematic transition of rods of length
*k* = 8 on a periodic **100 × 100 lattice**, without an external potential.
The transition is located from the maximum of `Var(|S|)`, where
`S = (N0 - N1)/(N0 + N1)` is calculated from the instantaneous global rod counts.
The stored finite-size estimate is **η_c ≈ 0.611**, with a working uncertainty
of **±0.002**.

## Stored results

- [summary.csv](results/final/summary.csv): mean packing fraction, mean |S|,
  variance of |S|, and standard errors.
- [Plots](results/final/order_and_variance.pdf) and
  [transition-region enlargement](results/final/order_and_variance_zoom.png).
- [peak_estimate.json](results/final/peak_estimate.json): peak estimate,
  bootstrap intervals, and fit-window sensitivity.
- [critical_packing_fraction.txt](results/final/critical_packing_fraction.txt):
  the value read by the main repository's plotting routines.

See [METHODS.md](METHODS.md) for sampling and error analysis, and
[RESULTS.md](RESULTS.md) for numerical checks and the full reproduction commands.
Exact run parameters and seeds are retained with the raw data.

## Build and run

Requires `make`, a C++17 compiler, and Python 3 with NumPy and Matplotlib.
The Python dependencies are included in the [repository setup](../README.md#setup).
Run the following commands from this directory.

```sh
make          # build the C++ sampler
make check    # validate against exhaustive enumeration
```

A short example scan, with fewer sweeps than the stored reference:

```sh
python3 run_scan.py --mu-min -1.5 --mu-max 1.5 --points 21 \
  --replicas 1 --burn 3000 --sweeps 12000 --block 500 \
  --workers 4 --out results/example
python3 analyze.py results/example --out results/example --reblock-sweeps 2000
```

The default sampler uses conditional row/column heat-bath updates, following
[Kundu et al., Phys. Rev. E 87, 032103 (2013), Sec. II](https://arxiv.org/pdf/1211.2536).
`--workers` sets the number of parallel chains (up to 20). Use `--help` for
other options.

To regenerate the stored reference plots and export the packing fraction:

```sh
python3 analyze.py results/final --out results/final --reblock-sweeps 50000 \
  --peak-estimate results/final/peak_estimate.json --zoom-min 0.58 --zoom-max 0.64
python3 export_critical_packing_fraction.py
```
