"""Run opt-in existing-pipeline trials from pixels. No fixture/world coordinates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import cv2 as cv

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--images',type=Path,required=True,help='JSON object mapping names to screenshot paths')
    ap.add_argument('--bundle',type=Path,required=True,help='esbuild bundle of prototype/baseline.ts')
    ap.add_argument('--vendor',type=Path,required=True,help='opencv.cjs, grass-sift.cjs, grass-sift.wasm')
    ap.add_argument('--artifact',type=Path,required=True,help='Production detect-wasm.wasm')
    ap.add_argument('--texture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--node',default='node')
    args=ap.parse_args()
    ref=cv.imread(str(args.texture),cv.IMREAD_GRAYSCALE)
    if ref is None or ref.shape!=(16,16):raise ValueError('Expected 16x16 grass-top texture')
    report={}
    for name,path in json.loads(args.images.read_text()).items():
        if not name or Path(name).name!=name:raise ValueError('Unsafe image output name')
        image=cv.imread(path)
        if image is None:raise ValueError('Unreadable image '+path)
        dest=args.out/name
        dest.mkdir(parents=True,exist_ok=True)
        gray=dest/'gray.bin';gray.write_bytes(image[:,:,1].tobytes())
        inp=dest/'input.json'
        inp.write_text(json.dumps(dict(gray=str(gray.resolve()),ref=ref.tolist(),w=image.shape[1],h=image.shape[0])))
        with (dest/'log.txt').open('w') as log:
            subprocess.run([args.node,str(args.bundle.resolve()),str(inp.resolve()),str((dest/'result.json').resolve()),
                            str(args.vendor.resolve()),str(args.artifact.resolve())],
                           env={**os.environ,'IMPROVED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        result=json.loads((dest/'result.json').read_text())
        blocks=result['blocks']
        (dest/'formation.txt').write_text(str(len(blocks))+'\n'+'\n'.join(
            f"{b['x']} {b['y']} {b['z']} {b['rotation']} 0" for b in blocks)+'\n')
        report[name]=dict(tiles=len(blocks),seconds=result['seconds'],trials=result['trials'])
        (args.out/'report.json').write_text(json.dumps(report,indent=2))
        print(name,report[name]['tiles'],report[name]['seconds'],flush=True)

if __name__=='__main__':
    main()
