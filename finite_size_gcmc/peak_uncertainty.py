#!/usr/bin/env python3
"""Local peak interpolation and a bootstrap of consecutive Monte Carlo blocks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze import SUMS, observable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--block-sweeps',type=int,default=50000)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--half-window',type=int,default=2)
    p.add_argument('--mu-min',type=float)
    p.add_argument('--mu-max',type=float)
    p.add_argument('--seed',type=int,default=712331)
    args=p.parse_args()
    with (args.input/'summary.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    mu=np.array([float(r['mu']) for r in rows])
    variance=np.array([float(r['var_abs_s']) for r in rows])
    top=int(np.argmax(variance))
    chosen=np.arange(max(0,top-args.half_window),min(len(mu),top+args.half_window+1))
    if args.mu_min is not None or args.mu_max is not None:
        mask=np.ones(len(mu),dtype=bool)
        if args.mu_min is not None: mask &= mu>=args.mu_min-1e-9
        if args.mu_max is not None: mask &= mu<=args.mu_max+1e-9
        chosen=np.flatnonzero(mask)
    if len(chosen)<3:
        raise SystemExit('Peak is not bracketed by at least three chemical potentials')
    x=mu[chosen]; center=mu[top]
    rng=np.random.default_rng(args.seed)
    boot_values=[]; measured=[]; nblocks=[]
    for chemical_potential in x:
        blocks=[]
        for f in sorted((args.input/'raw').glob('*.csv')):
            if not f.with_suffix('.done.json').exists():
                continue
            with f.open() as handle:
                raw_rows=list(csv.DictReader(handle))
            if not raw_rows or abs(float(raw_rows[0]['mu'])-chemical_potential)>1e-8:
                continue
            per=int(raw_rows[0]['block_sweeps'])
            group=max(1,int(np.ceil(args.block_sweeps/per)))
            raw=np.array([[float(r[key]) for key in SUMS] for r in raw_rows])
            N=len(raw)//group
            if not N: raise SystemExit('Chain shorter than requested analysis block')
            blocks.extend(raw[:N*group].reshape(N,group,-1).sum(axis=1))
        blocks=np.asarray(blocks)
        measured.append(observable(blocks.sum(axis=0)))
        indices=rng.integers(0,len(blocks),size=(args.bootstrap,len(blocks)))
        boot_values.append(observable(blocks[indices].sum(axis=1)))
        nblocks.append(len(blocks))
    measured=np.asarray(measured)
    boot_values=np.asarray(boot_values).transpose(1,0,2)
    errors=boot_values[:,:,2].std(axis=0,ddof=1)
    weights=1/np.maximum(errors,1e-10)
    def peak(values):
        coeff=np.polyfit(x-center,values[:,2],2,w=weights)
        if coeff[0]>=0: return None
        location=center-coeff[1]/(2*coeff[0])
        if not x[0]<=location<=x[-1]: return None
        return [location,float(np.interp(location,x,values[:,0])),float(np.polyval(coeff,location-center))]
    estimate=peak(measured)
    if estimate is None: raise SystemExit('Local quadratic does not have a bracketed maximum')
    samples=np.array([v for values in boot_values if (v:=peak(values)) is not None])
    if len(samples)<.95*args.bootstrap:
        print('WARNING: more than 5% of bootstrap peaks are unbracketed. Widen the fit window before interpreting the intervals.')
    if len(samples)<2:
        raise SystemExit('Insufficient bracketed bootstrap peaks')
    result=dict(method='weighted local quadratic; non-overlapping block bootstrap',
                fitted_mus=x.tolist(),block_sweeps=args.block_sweeps,blocks_per_mu=nblocks,
                bootstrap=args.bootstrap,accepted_bootstrap=len(samples),
                estimate=dict(mu=estimate[0],eta=estimate[1],variance=estimate[2]),
                standard_error=dict(zip(['mu','eta','variance'],samples.std(axis=0,ddof=1).tolist())),
                percentile_95=dict(zip(['mu','eta','variance'],np.quantile(samples,[.025,.975],axis=0).T.tolist())),
                note='Statistical uncertainty conditional on this interpolation window. Compare window sizes for fit sensitivity.')
    output=args.input/f'peak_bootstrap_window{len(chosen)}_block{args.block_sweeps}.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
