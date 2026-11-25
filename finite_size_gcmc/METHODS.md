# Short methods note: finite-size GCMC reference

## System and definition of a step

The simulated system is a periodic **100 × 100 square lattice** containing hard
rods of length **k = 8**, with two orientations and no external potential. Each
rod has activity `z = exp(u)`, where `u = beta*mu` is common to both orientations.

The main sampler uses exact conditional row/column heat-bath updates. All rods
of one orientation are removed, and the available intervals are repopulated
according to their exact one-dimensional grand-canonical distribution, keeping
the other orientation fixed. Both orientations are then updated. Their update
order is randomized, and periodic lines are treated explicitly.

**One heat-bath sweep updates all 100 rows and all 100 columns:** 200 conditional
line updates. This is the unit used for the step counts below. The method follows
[Kundu et al., Sec. II](https://arxiv.org/pdf/1211.2536), without high-density flip
moves. The independent single-rod check uses a different unit: one local sweep
is 10,000 attempted insertions/deletions.

## Number of sweeps and independent runs

| Scan | Range of u | Chemical potentials | Chains per potential | Discarded warmup per chain | Production per chain |
|---|---|---:|---:|---:|---:|
| Low density | -5 to -2 | 7 | 2 | 10,000 | 100,000 |
| Overview | -1.5 to 1.5 | 21 | 2 | 20,000 | 200,000 |
| First refinement | -0.45 to -0.10 | 15 | 4 | 20,000 | 1,000,000 |
| Focused peak scan | -0.25 to -0.10 | 11 | 6 | 50,000 | 1,000,000 |

These scans contribute **182 independent chains**, **135,800,000 production
sweeps**, and **5,480,000 discarded warmup sweeps**. Overlapping chemical-potential
points are pooled, giving 48 distinct points in the final curve. Each chain has
its own random seed. Across replicas, initial states include empty lattices and
lattices aligned in either orientation.

The initial coarse scan (21 chains, 3,000 warmup + 12,000 production sweeps each)
was used to find the region of interest and is excluded from the final curve.
The single-rod validation runs are also excluded from that curve.

## What was measured

After **every production sweep**, the global rod counts give

$$
S_t=\frac{N_0(t)-N_1(t)}{N_0(t)+N_1(t)},\qquad
\eta_t=\frac{8[N_0(t)+N_1(t)]}{100^2}.
$$

The absolute value is taken for each configuration, before averaging. At each
chemical potential, the reported estimates are

$$
\widehat q=\frac{1}{M}\sum_t |S_t|,\qquad
\widehat v=\frac{1}{M}\sum_t S_t^2-\widehat q^2.
$$

Moments are pooled across the independent chains before calculating the variance.
Individual block variances are not averaged. The empty configuration is assigned
S = 0, with its frequency recorded. Signed S and its fourth moment are also saved.

Measurements are accumulated online. For the final heat-bath scans, the raw files
store sums over consecutive **2,000-sweep blocks**, retaining every measurement's
contribution without storing 135.8 million individual configurations.

## Error bars on the curves

For analysis, 25 consecutive raw blocks are joined into **50,000-sweep blocks**,
without crossing chain boundaries. At each chemical potential, these blocks are
treated as approximately independent. For B blocks, each block is omitted in
turn and the complete estimate is recalculated from the remaining data.

For either q or v, the delete-one-block jackknife standard error is

$$
\mathrm{SE}(\widehat\theta)=
\sqrt{\frac{B-1}{B}\sum_{b=1}^{B}
\left(\widehat\theta_{(-b)}-
\overline{\widehat\theta}_{(-)}\right)^2}.
$$

The overbar denotes the mean of the B leave-one-block-out estimates.
For the variance, both moments are recalculated together after deleting a block.
This preserves their statistical dependence. The point estimate itself remains
the pooled-moment estimate above.

**The plotted vertical error bars are ±1 standard error**, not 95% intervals.
Errors on mean packing fraction are retained in `summary.csv` but are not plotted.
To check the blocking choice, the first refined scan was also analyzed with
20,000- and 100,000-sweep blocks. At u = -0.20, the variance standard errors were
0.000877, 0.000908, and 0.000845 for 20,000, 50,000, and 100,000 sweeps, respectively.
No precise integrated autocorrelation time was estimated.

## Peak location and its uncertainty

The peak is estimated by fitting a quadratic to the variance versus **chemical
potential**, using the 15 sampled points in `-0.25 <= u <= -0.10`. Weighted least
squares minimizes the sum of squared residuals divided by their variance. The
weights come from the block-bootstrap standard errors described below.

The quadratic's vertex gives u at the peak and its fitted maximum variance.
The corresponding mean packing fraction is obtained by linear interpolation of
the measured mean eta versus u.

For uncertainty, **3,000 block-bootstrap datasets** are generated. Independently
at each chemical potential, the pooled 50,000-sweep blocks are sampled with
replacement, keeping the same number of blocks. Each selected block carries all
its moments together. Each bootstrap dataset is used to recompute the moments,
refit the quadratic with the fixed weights, and interpolate eta at its vertex.
The 2.5th and 97.5th percentiles give the statistical 95% intervals:

| Quantity | Main estimate | Statistical 95% interval |
|---|---:|---:|
| u at the peak | -0.189780 | [-0.199138, -0.181952] |
| Mean eta at the peak | 0.611367 | [0.609946, 0.612369] |
| Maximum variance of absolute S | 0.061481 | [0.060894, 0.062065] |

All 3,000 bootstrap fits had a maximum inside the fitting window. These intervals
are conditional on the local quadratic approximation. Changing the window to
`[-0.275,-0.10]` or `[-0.235,-0.115]` gives peak eta values 0.611368 and 0.611717.

The suggested **0.611 ± 0.002** for peak eta and **0.0615 ± 0.001** for maximum
variance are rounded working uncertainties allowing for this fit sensitivity.
They are **neither one-standard-error bars nor formally constructed 95%
intervals**. The dashed line in the final plots marks the main fitted peak, and
the gray strip shows its statistical 95% interval in eta.

## Checks and retained records

Both samplers were checked against exhaustive enumeration of a periodic 4 × 4
dimer system, and the heat-bath sampler passed an independent ideal-monomer test.
Four full-size single-rod chains at u = -0.175 each used 100,000 discarded and
1,000,000 production sweeps. Their variance agrees with the heat-bath reference
to 0.20 combined standard errors. First/second-half summaries are retained for
each chain, alongside the initial states and sign-change counts.

See [RESULTS.md](RESULTS.md) for the numerical checks, [README.md](README.md) for
NixOS/run instructions, and [summary.csv](results/final/summary.csv) for the curve.
Exact commands and seeds are recorded in each stage's `manifest.json` and each
chain's `.done.json`.
