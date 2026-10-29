#!/usr/bin/env python3
"""Check the SMPL-X NPZ fields expected by the externally maintained EMAGE tools.

This checks the saved file structure only; it does not call the EMAGE renderer/evaluator.
"""
import argparse
from pathlib import Path
import numpy as np

REQUIRED = ('poses', 'expressions', 'betas', 'trans', 'model', 'gender', 'mocap_frame_rate')


def check_file(path: Path):
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED).difference(data.files)
        if missing:
            raise ValueError(f'Missing NPZ fields: {sorted(missing)}')
        poses, expressions, betas, trans = (data[k] for k in ('poses', 'expressions', 'betas', 'trans'))
        if poses.ndim != 2 or poses.shape[1] != 165 or len(poses) == 0:
            raise ValueError(f'poses expected [T,165], received {poses.shape}')
        T = len(poses)
        expected = {'expressions': (T, 100), 'betas': (300,), 'trans': (T, 3)}
        for key, size in expected.items():
            if data[key].shape != size:
                raise ValueError(f'{key} expected {size}, received {data[key].shape}')
        for key in ('poses', 'expressions', 'betas', 'trans'):
            a = data[key]
            if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
                raise ValueError(f'{key} contains nonnumeric or nonfinite values')
        if str(data['model'].item()) != 'smplx2020':
            raise ValueError(f"model expected 'smplx2020', received {data['model'].item()}")
        if str(data['gender'].item()) != 'neutral':
            raise ValueError(f"gender expected 'neutral', received {data['gender'].item()}")
        if float(data['mocap_frame_rate'].item()) != 30:
            raise ValueError('mocap_frame_rate expected 30')
    return T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path, help='Single generated NPZ or directory of NPZ files')
    args = p.parse_args()
    paths = sorted(args.path.rglob('*.npz')) if args.path.is_dir() else [args.path]
    if not paths:
        p.error(f'No NPZ files found under {args.path}')
    invalid = []
    for path in paths:
        try:
            T = check_file(path)
            print(f'[OK] {path.name}  T={T}')
        except Exception as exc:
            invalid.append((path, exc))
            print(f'[ERROR] {path}: {exc}')
    print(f'Validated {len(paths)-len(invalid)}/{len(paths)} files')
    if invalid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
