"""Run pixel-only baseline, grid-only, hybrid, then separate labeled evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]


def run(command, log=None, env=None):
    if log:
        with log.open('w') as stream:
            subprocess.run(list(map(str, command)), cwd=ROOT/'web', check=True,
                           timeout=240, stdout=stream, stderr=subprocess.STDOUT, env=env)
    else:
        subprocess.run(list(map(str, command)), cwd=ROOT/'web', check=True, timeout=240, env=env)


def main():
    ap = argparse.ArgumentParser(__doc__)
    for arg in ['birch', 'overhead', 'plains', 'assets', 'birch-fixture', 'out']:
        ap.add_argument('--'+arg, type=Path, required=True)
    ap.add_argument('--overhead-reference', type=Path)
    ap.add_argument('--node', default='node')
    ap.add_argument('--esbuild', type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    vendor = args.out/'vendor'
    vendor.mkdir(exist_ok=True)
    for source, dest in [('dist/vendor/opencv.js','opencv.cjs'),
                         ('dist/vendor/grass-sift.js','grass-sift.cjs'),
                         ('dist/vendor/grass-sift.wasm','grass-sift.wasm')]:
        shutil.copy2(args.assets/source, vendor/dest)
    run([args.esbuild, ROOT/'prototype/baseline.ts', '--bundle',
         '--platform=node', '--format=esm', '--outfile='+str(args.out/'baseline.mjs')])
    images = dict(hill=ROOT/'web/public/examples/hill.png', birch=args.birch.resolve(),
                  overhead=args.overhead.resolve(), plains=args.plains.resolve())
    manifest = dict(images={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for name,p in images.items()},baseline_commit='0580d5970dca983456f5a3f129e1a50386bfad95',
                    artifacts={name:hashlib.sha256((args.assets/'artifacts'/name).read_bytes()).hexdigest()
                               for name in ['solver.wasm','detect-wasm.wasm']},
                    versions=dict(opencv=cv.__version__,python=sys.version))
    (args.out/'inputs.json').write_text(json.dumps(manifest, indent=2))
    ref = cv.imread(str(ROOT/'web/public/grass_block_top.png'), 0)
    for name, p in images.items():
        image = cv.imread(str(p))
        if image is None:
            raise ValueError('Cannot decode '+str(p))
        dest = args.out/'baseline'/name
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'gray.bin').write_bytes(image[:,:,1].tobytes())
        (dest/'input.json').write_text(json.dumps(dict(w=image.shape[1],h=image.shape[0],
                                                      gray=str(dest/'gray.bin'),ref=ref.tolist())))
        run([args.node,args.out/'baseline.mjs',dest/'input.json',dest/'result.json',
             vendor,args.assets/'artifacts/detect-wasm.wasm'],dest/'log.txt')
    for mode in ['grid-only','hybrid']:
        for name,p in images.items():
            dest = args.out/mode/name
            dest.mkdir(parents=True,exist_ok=True)
            cmd = [sys.executable,'-m','robust_detect.cli',p,'--texture',
                   ROOT/'web/public/grass_block_top.png','--out',dest]
            if mode == 'hybrid':
                cmd += ['--hybrid-seeds',args.out/'baseline'/name/'result.json']
            run(cmd,dest/'log.txt',env={**os.environ,'PYTHONPATH':str(ROOT/'prototype')})
    eval_args = [sys.executable,ROOT/'prototype/evaluate.py','--runs',args.out,
                 '--hill-fixture',ROOT/'web/public/examples/hill.json',
                 '--birch-fixture',args.birch_fixture]
    if args.overhead_reference:
        eval_args += ['--overhead-reference',args.overhead_reference.resolve()]
    run(eval_args)
    run([args.node,ROOT/'prototype/solver-regression.cjs',args.out,
         args.assets/'artifacts/solver.wasm'])


if __name__ == '__main__':
    main()
