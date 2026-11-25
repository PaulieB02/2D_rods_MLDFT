#!/usr/bin/env python3
"""Run independent single-threaded C++ chains, reserving four logical CPUs."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mu-min', type=float, default=-1.5)
    p.add_argument('--mu-max', type=float, default=1.5)
    p.add_argument('--points', type=int, default=21)
    p.add_argument('--mus', type=float, nargs='+')
    p.add_argument('--replicas', type=int, default=1)
    p.add_argument('--L', type=int, default=100)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--burn', type=int, default=3000)
    p.add_argument('--sweeps', type=int, default=12000)
    p.add_argument('--block', type=int, default=500)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--sampler', choices=['heatbath', 'local'], default='heatbath')
    p.add_argument('--initial', choices=['mixed', 'empty', 'vertical', 'horizontal'], default='mixed')
    p.add_argument('--verify-every', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.workers <= 20:
        p.error('workers must lie between 1 and 20')
    mus = args.mus or [args.mu_min + (args.mu_max-args.mu_min)*i/max(1,args.points-1)
                      for i in range(args.points)]
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    (out/'raw').mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    selected = cpus[4:] if len(cpus) > 4 else cpus
    prefix = [shutil.which('taskset'), '-c', ','.join(map(str,selected))] if selected and shutil.which('taskset') else []
    workers = min(args.workers, len(selected) or args.workers)
    manifest = dict(vars(args), out=str(out), mus=mus, cpus=selected,
                    started=datetime.datetime.now(datetime.timezone.utc).isoformat(), workers=workers)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    jobs = []
    for i,mu in enumerate(mus):
        for r in range(args.replicas):
            initial = ['empty','vertical','empty','horizontal'][r%4] if args.initial=='mixed' else args.initial
            seed = args.seed + 1000003*i + 9176*r
            name = f'mu_{mu:+.6f}_rep_{r:02d}_seed_{seed}'
            target = out/'raw'/f'{name}.csv'
            cmd = prefix+[str(root/'rods_gcmc'), '--L',str(args.L),'--k',str(args.k),
                          '--mu',str(mu),'--burn',str(args.burn),'--sweeps',str(args.sweeps),
                          '--block',str(args.block),'--stride',str(args.stride),'--seed',str(seed),
                          '--initial',initial,'--sampler',args.sampler,'--output',str(target),
                          '--verify-every',str(args.verify_every)]
            jobs.append((cmd,target))
    started = time.monotonic()
    def run(job):
        cmd,target = job
        done = target.with_suffix('.done.json')
        if done.exists():
            return target.name+' already complete'
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with target.with_suffix('.log').open('w') as log:
            result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        message = target.with_suffix('.log').read_text().strip()
        if result.returncode:
            raise RuntimeError(f'{target.name} failed: {message}')
        done.write_text(json.dumps(dict(command=cmd,summary=message),indent=2)+'\n')
        return message
    print(f'{len(jobs)} chains, {workers} workers, CPUs {selected}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run,job) for job in jobs]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            print(f'[{i}/{len(jobs)}] {future.result()}',flush=True)
    manifest['elapsed_seconds'] = time.monotonic()-started
    manifest['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    print(f'Finished in {manifest["elapsed_seconds"]:.1f} seconds.',flush=True)


if __name__=='__main__':
    main()
