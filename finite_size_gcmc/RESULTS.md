# Finite-size reference: results

System: periodic **100 × 100 square lattice**, **k = 8**, no external potential,
equal activities `z0 = z1 = exp(mu)`. The observable is the absolute value of the
instantaneous **global** order parameter `(N0-N1)/(N0+N1)`.

## Peak

A weighted local quadratic interpolation of the variance as a function of
chemical potential, over `-0.25 <= mu <= -0.10`, gives:

| Quantity | Estimate | Statistical 95% block-bootstrap interval |
|---|---:|---:|
| Chemical potential `beta*mu` | -0.18978 | [-0.19914, -0.18195] |
| Mean packing fraction at the peak | 0.611367 | [0.609946, 0.612369] |
| Maximum variance of absolute S | 0.061481 | [0.060894, 0.062065] |

The bootstrap uses 3,000 resamples of consecutive 50,000-sweep blocks. All
resampled interpolations have a bracketed maximum. Widening the fit window to
`[-0.275,-0.10]` gives eta = 0.611368 and variance = 0.061469. Narrowing it to
`[-0.235,-0.115]` gives eta = 0.611717 and variance = 0.061660.

For a practical comparison, use **eta_peak ≈ 0.611 ± 0.002** and
**max Var(|S|) ≈ 0.0615 ± 0.001**. These rounded working uncertainties allow
for sampling variation and the observed fit-window sensitivity. They are not
separate rigorously constructed confidence intervals.

The largest *measured grid point* is at `mu=-0.205`, eta = 0.609171, with
variance 0.061693 ± 0.000647 (one standard error). The peak is broad, so selecting
one noisy grid maximum is less informative than the local interpolation. The
very small error on the mean eta at a fixed mu is **not** the uncertainty of
the peak location.

The dashed line in the final figures marks the main interpolated peak. The
shaded strip is its statistical 95% interval in mean packing fraction.

## Sampling retained in the final curve

| Stage | mu grid | Replicas per mu | Burn-in sweeps per chain | Production sweeps per chain |
|---|---|---:|---:|---:|
| `low_density` | -5 to -2, 7 points | 2 | 10,000 | 100,000 |
| `overview` | -1.5 to 1.5, 21 points | 2 | 20,000 | 200,000 |
| `refined` | -0.45 to -0.10, 15 points | 4 | 20,000 | 1,000,000 |
| `peak_refined` | -0.25 to -0.10, 11 points | 6 | 50,000 | 1,000,000 |

There are **182 independent chains** and **135,800,000 production sweeps** in
the final reference. Measurements are made every full heat-bath sweep, before
any averaging or species cancellation. Replicas start from empty, vertically
aligned, and horizontally aligned configurations. A sweep updates both species
on the entire lattice. The raw blocks contain sums of eta, eta squared, |S|,
S squared, S to the fourth power, and signed S.

Pointwise standard errors use a delete-one-block jackknife with 50,000-sweep
blocks. Reblocking the first refined scan at 20,000, 50,000 and 100,000 sweeps
gave comparable errors. For example, at mu=-0.20 the variance standard errors
were 0.000877, 0.000908 and 0.000845, respectively. The signed mean is retained
as a diagnostic but is not used to define the plotted order parameter.

## Independent checks

Both samplers pass exhaustive-enumeration checks of eta, mean |S|, and its
variance for a periodic 4 × 4 dimer lattice at mu=-1, 0 and 0.8. A separate
ideal-monomer check reproduces `eta = 2 exp(mu)/(1+2 exp(mu))`. Occupancy grids
are checked against the rod lists and rejected if overlaps or mismatches occur.

The full-size reference was also checked at mu=-0.175 with ordinary single-rod
insertion/deletion. Four independent local chains each discard 100,000 sweeps
and measure 1,000,000 sweeps. These checks are not pooled into the final curve.

| Observable at mu=-0.175 | Heat bath | Single-rod insertion/deletion |
|---|---:|---:|
| Mean eta | 0.613230 ± 0.000060 | 0.613039 ± 0.000130 |
| Mean absolute S | 0.532064 ± 0.003917 | 0.521110 ± 0.008834 |
| Variance of absolute S | 0.061614 ± 0.000678 | 0.061916 ± 0.001328 |

Errors in this table are one standard error. The variance difference is 0.20
combined standard errors. The eta and mean-|S| differences are 1.33 and 1.13
combined standard errors, respectively.

## Files and reproduction

- `results/final/summary.csv`: all curve points and pointwise standard errors.
- `results/final/peak_estimate.json`: main peak, bootstrap intervals, and fit-window sensitivity.
- `results/final/chain_diagnostics.csv`: first/second-half comparisons for each chain.
- `results/final/sampler_comparison.json`: the full-size independent check.
- `results/final/sources.json`: raw-chain paths and source/binary SHA256 hashes.
- Each stage's `manifest.json`: parameters and wall-clock duration.
- Each raw chain's `.done.json`: exact executable command and seed.

The final analysis can be repeated with the commands below. `collect_results.py` links the
completed chains of the four stages into `results/final/raw`; that directory is already
populated (with copies of the chains), so the first command is only needed when starting from
the stage directories alone.

```sh
python3 collect_results.py results/low_density results/overview results/refined \
  results/peak_refined --out results/final
python3 analyze.py results/final --out results/final --reblock-sweeps 50000
python3 peak_uncertainty.py results/final --block-sweeps 50000 --bootstrap 3000 \
  --mu-min -0.25 --mu-max -0.10
python3 analyze.py results/final --out results/final --reblock-sweeps 50000 \
  --peak-estimate results/final/peak_estimate.json --zoom-min 0.58 --zoom-max 0.64
python3 export_critical_packing_fraction.py
```

The last command writes `results/final/critical_packing_fraction.txt`, the eta_c used in the
figures of the article. `peak_estimate.json` collects the main fit
(`peak_bootstrap_window15_block50000.json`) together with the fit-window sensitivity
(windows 13 and 16) and the rounded working uncertainties.

The initial `results/coarse` scan is retained as a record of the first rough
estimate. It is superseded by the longer runs above.
