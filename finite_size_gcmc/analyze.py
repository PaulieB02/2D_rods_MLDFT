#!/usr/bin/env python3
"""Pool raw configuration moments, estimate errors by block jackknife, plot."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SUMS = ['samples','sum_eta','sum_eta2','sum_abs_s','sum_s2','sum_s4','sum_s']


def observable(sums):
    n,e,e2,q,q2,q4,s = np.moveaxis(np.asarray(sums),-1,0)
    return np.stack((e/n,q/n,q2/n-(q/n)**2,1-(q4/n)/(3*(q2/n)**2),s/n),axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reblock-sweeps',type=int,default=2000)
    p.add_argument('--peak-min',type=float)
    p.add_argument('--peak-max',type=float)
    p.add_argument('--peak-estimate',type=Path,help='Bootstrap report used for the plotted peak marker')
    p.add_argument('--zoom-min',type=float)
    p.add_argument('--zoom-max',type=float)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    groups = defaultdict(list)
    files = sorted(set(f.resolve() for d in args.inputs for f in (d/'raw').glob('*.csv') if f.with_suffix('.done.json').exists()))
    for f in files:
        with f.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        mu = round(float(rows[0]['mu']),10)
        groups[mu].append((f,rows))
    if not groups:
        raise SystemExit('No completed chains found')
    table=[]
    diagnostics=[]
    for mu,chains in sorted(groups.items()):
        blocks=[]
        for f,rows in chains:
            raw = np.array([[float(row[key]) for key in SUMS] for row in rows])
            per = int(rows[0]['block_sweeps'])
            # Equal-duration blocks preserve the sampling autocorrelation inside each block.
            group = max(1,int(np.ceil(args.reblock_sweeps/per)))
            nblocks = len(raw)//group
            if nblocks < 2:
                group=max(1,len(raw)//2)
                nblocks=len(raw)//group
            coarse = raw[:nblocks*group].reshape(nblocks,group,-1).sum(axis=1)
            blocks.extend(coarse)
            half=len(raw)//2
            first=observable(raw[:half].sum(axis=0)); second=observable(raw[half:].sum(axis=0))
            diagnostics.append(dict(file=str(f),mu=mu,initial=rows[0]['initial'],sampler=rows[0]['sampler'],
                                    eta_first=first[0],eta_second=second[0],abs_s_first=first[1],abs_s_second=second[1],
                                    var_first=first[2],var_second=second[2],analysis_blocks=nblocks,
                                    analysis_block_sweeps=group*per,sign_changes=sum(int(r['sign_changes']) for r in rows)))
        blocks=np.asarray(blocks)
        total=blocks.sum(axis=0)
        value=observable(total)
        loo=observable(total-blocks)
        B=len(blocks)
        sem=np.sqrt((B-1)/B*np.sum((loo-loo.mean(axis=0))**2,axis=0)) if B>1 else value*np.nan
        row=dict(mu=mu,L=int(chains[0][1][0]['L']),k=int(chains[0][1][0]['k']),
                 eta=value[0],eta_sem=sem[0],mean_abs_s=value[1],mean_abs_s_sem=sem[1],
                 var_abs_s=value[2],var_abs_s_sem=sem[2],binder=value[3],binder_sem=sem[3],
                 signed_mean=value[4],signed_mean_sem=sem[4],replicas=len(chains),samples=int(total[0]),blocks=B)
        table.append(row)
    def save_csv(path,rows):
        with path.open('w') as out:
            w=csv.DictWriter(out,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save_csv(args.out/'summary.csv',table)
    save_csv(args.out/'chain_diagnostics.csv',diagnostics)
    x=np.array([r['eta'] for r in table]); mu=np.array([r['mu'] for r in table])
    q=np.array([r['mean_abs_s'] for r in table]); qerr=np.array([r['mean_abs_s_sem'] for r in table])
    var=np.array([r['var_abs_s'] for r in table]); verr=np.array([r['var_abs_s_sem'] for r in table])
    mask=np.ones(len(table),dtype=bool)
    if args.peak_min is not None: mask &= mu>=args.peak_min
    if args.peak_max is not None: mask &= mu<=args.peak_max
    selected=np.flatnonzero(mask)
    top=int(selected[np.argmax(var[mask])])
    result=dict(grid_max=table[top],reblock_sweeps=args.reblock_sweeps,
                note='Finite-size peak of <S^2>-<|S|>^2. Errors: block jackknife. No thermodynamic-limit extrapolation.')
    # A local fit is an interpolation, not a replacement for the measured grid maximum.
    nearest=selected[np.argsort(np.abs(mu[selected]-mu[top]))[:5]]
    nearest=np.sort(nearest)
    if len(nearest)>=3:
        center=mu[top]
        coeff=np.polyfit(mu[nearest]-center,var[nearest],2,w=1/np.maximum(verr[nearest],1e-8))
        if coeff[0]<0:
            vertex=center-coeff[1]/(2*coeff[0])
            if mu[nearest].min()<=vertex<=mu[nearest].max():
                result['local_quadratic']=dict(mu=float(vertex),eta=float(np.interp(vertex,mu,x)),
                                               variance=float(np.polyval(coeff,vertex-center)),
                                               fitted_mus=mu[nearest].tolist())
    (args.out/'peak.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(6,7),sharex=True,layout='constrained')
    axes[0].errorbar(x,q,yerr=qerr,fmt='o-',markersize=3,capsize=2)
    axes[1].errorbar(x,var,yerr=verr,fmt='o-',markersize=3,capsize=2)
    axes[0].set_ylabel(r'$\langle |S|\rangle$'); axes[1].set_ylabel(r'$\mathrm{Var}(|S|)$')
    axes[1].set_xlabel(r'Mean packing fraction $\langle\eta\rangle$')
    peak_eta=x[top]
    interval=None
    if args.peak_estimate:
        report=json.loads(args.peak_estimate.read_text())
        peak_eta=report['estimate']['eta']
        interval=report.get('percentile_95',{}).get('eta')
    for ax in axes:
        ax.grid(alpha=.25); ax.axvline(peak_eta,color='k',ls='--',lw=.8)
        if interval: ax.axvspan(*interval,color='gray',alpha=.15)
    axes[0].set_title(f'Periodic {table[0]["L"]} × {table[0]["L"]} lattice, k = {table[0]["k"]}')
    fig.savefig(args.out/'order_and_variance.png',dpi=150)
    fig.savefig(args.out/'order_and_variance.pdf')
    if args.zoom_min is not None or args.zoom_max is not None:
        axes[1].set_xlim(left=args.zoom_min,right=args.zoom_max)
        fig.savefig(args.out/'order_and_variance_zoom.png',dpi=150)
        fig.savefig(args.out/'order_and_variance_zoom.pdf')
    plt.close(fig)
    print('GRID MAXIMUM:',json.dumps(table[top],indent=2))
    if 'local_quadratic' in result: print('LOCAL QUADRATIC:',json.dumps(result['local_quadratic'],indent=2))
    print('Wrote',args.out/'summary.csv',args.out/'order_and_variance.png',flush=True)


if __name__=='__main__':
    main()
