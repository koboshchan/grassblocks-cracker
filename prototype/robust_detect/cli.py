"""python -m robust_detect.cli IMAGE --texture TEXTURE --out DIR"""
import argparse
import json
from pathlib import Path
import time
import cv2 as cv
from .detector import detect


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('image', type=Path)
    ap.add_argument('--texture', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hybrid-seeds', type=Path, help='Baseline output; optional, separately labeled hybrid mode')
    args = ap.parse_args()
    im, texture = cv.imread(str(args.image)), cv.imread(str(args.texture), cv.IMREAD_GRAYSCALE)
    if im is None or texture is None or texture.shape != (16, 16):
        ap.error('Image must decode and texture must be 16x16')
    if im.shape[0] * im.shape[1] > 16 * 1024 * 1024:
        ap.error('Image exceeds 16-megapixel offline safety limit')
    cv.setNumThreads(1)
    seeds = json.loads(args.hybrid_seeds.read_text()).get('seeds', []) if args.hybrid_seeds else None
    start = time.perf_counter()
    result = detect(im, texture, seeds)
    result.update(seconds=time.perf_counter()-start, mode='hybrid' if seeds is not None else 'grid-only')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'result.json').write_text(json.dumps(result, indent=2))
    blocks = result['blocks']
    rows = [str(len(blocks))] + [f"{b['x']} {b['y']} {b['z']} {b['rotation']} 0" for b in blocks]
    (args.out/'formation.txt').write_text('\n'.join(rows)+'\n')
    overlay = im.copy()
    import numpy as np
    for b in blocks:
        q = np.int32(b['quad'])
        cv.polylines(overlay, [q], True, (50, 255, 80), 2)
        cv.putText(overlay, str(b['rotation']), tuple(q[0]), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
    if not cv.imwrite(str(args.out/'overlay.png'), overlay):
        raise RuntimeError('Overlay write failed')
    print(json.dumps(dict(mode=result['mode'], seconds=result['seconds'], blocks=len(blocks),
                          planes=len(result['planes']), meta=result['meta'])), flush=True)


if __name__ == '__main__':
    main()
