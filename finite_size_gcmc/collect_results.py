#!/usr/bin/env python3
"""Collect completed chains by relative symlinks, without duplicating raw data."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    raw=args.out.resolve()/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    included=[]
    for directory in args.inputs:
        for f in sorted((directory/'raw').glob('*.csv')):
            if not f.with_suffix('.done.json').exists(): continue
            target=raw/f'{directory.name}_{f.name}'
            for source,destination in [(f,target),(f.with_suffix('.done.json'),target.with_suffix('.done.json'))]:
                if destination.is_symlink():
                    if destination.resolve()!=source.resolve():
                        raise RuntimeError(f'Conflicting source for {destination}')
                elif destination.exists():
                    raise RuntimeError(f'Refusing to replace regular file {destination}')
                else:
                    destination.symlink_to(os.path.relpath(source.resolve(),destination.parent))
            included.append(str(f.resolve()))
    root=Path(__file__).resolve().parent
    hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest()
            for f in ['rods_gcmc.cpp','rods_gcmc','run_scan.py','analyze.py','peak_uncertainty.py']}
    (args.out/'sources.json').write_text(json.dumps(dict(chains=included,sha256=hashes),indent=2)+'\n')
    print(f'Collected {len(included)} completed chains in {raw}')


if __name__=='__main__':
    main()
