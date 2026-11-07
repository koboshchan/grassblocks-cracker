"""Evaluation only. Labeled fixture quads NEVER imported by detector."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def rotation(x, y, z):
    v = (x * 3129871) & 0xffffffff
    if v >> 31:
        v -= 1 << 32
    value = (v ^ (z * 116129781) ^ y) & ((1 << 64)-1)
    value = (value*value*42317861 + value*11) & ((1 << 64)-1)
    if value >> 63:
        value -= 1 << 64
    return ((((value >> 16) ^ 0x5deece66d)*0x5deece66d+11) & ((1 << 48)-1)) >> 46


def turn(x, z, n):
    for _ in range(n):
        x, z = -z, x
    return x, z


def paired(a, refs):
    q = np.array(a['quad'])
    best = None
    for ref in refs:
        r = np.array(ref['quad'])
        # Allow a cyclic corner start, but never allow mirrored geometry.
        for shift in range(4):
            r2 = np.roll(r, -shift, axis=0)
            error = float(np.linalg.norm(q-r2, axis=1).mean())
            scale = np.linalg.norm(r-np.roll(r, -1, axis=0), axis=1).min()
            if error < max(3., scale*.12) and (best is None or error < best[0]):
                best = error, ref, shift
    return best


def evaluation(result, refs, coords, fixture_north):
    pairs = [(b, paired(b, refs)) for b in result['blocks']]
    pairs = [(b, p) for b, p in pairs if p]
    matches = sum(b['rotation'] == (p[1]['rotation']-p[2]) % 4 for b, p in pairs)
    # Infer exported anchor from independently labeled geometry. Not from maximizing
    # hash agreement, so bad classifications cannot choose favorable world offsets.
    registrations = []
    for b, (_, ref, shift) in pairs:
        # A shifted corner is a different block origin, with the corresponding axis turn.
        ref_origin = [(0,0),(1,0),(1,1),(0,1)][shift]
        wx, wz = turn(ref['x']+ref_origin[0], ref['z']+ref_origin[1], fixture_north)
        north = (fixture_north-shift) % 4
        bx, bz = turn(b['x'], b['z'], north)
        registrations.append((coords[0]+wx-bx, coords[1]+ref['y']-b['y'],
                              coords[2]+wz-bz, north))
    anchor = Counter(registrations).most_common(1)
    anchor = anchor[0][0] if anchor else None
    hash_good = 0
    if anchor:
        x, y, z, north = anchor
        for b in result['blocks']:
            bx, bz = turn(b['x'], b['z'], north)
            hash_good += rotation(x+bx,y+b['y'],z+bz) == (b['rotation']+north) % 4
    return dict(tiles=len(result['blocks']), labeled_overlap=len(pairs),
                correct_labeled_rotations=int(matches), registered_anchor=anchor,
                hash_matches=int(hash_good), all_hash_match=bool(result['blocks']) and hash_good==len(result['blocks']))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--hill-fixture', type=Path, required=True)
    ap.add_argument('--birch-fixture', type=Path, required=True)
    ap.add_argument('--overhead-reference', type=Path, help='Independent classifier output, NOT world ground truth')
    args = ap.parse_args()
    cases = dict(hill=(json.loads(args.hill_fixture.read_text())['blocks'], (3538,90,-454),0),
                 birch=(json.loads(args.birch_fixture.read_text())['blocks'], (-4133,63,-2049),3))
    report = {}
    for mode in ['baseline','grid-only','hybrid']:
        report[mode] = {}
        for name in ['hill','birch','overhead','plains']:
            r = json.loads((args.runs/mode/name/'result.json').read_text())
            metrics = evaluation(r, *cases[name]) if name in cases else dict(tiles=len(r['blocks']))
            if name == 'overhead' and args.overhead_reference:
                reference = json.loads(args.overhead_reference.read_text())['blocks']
                overlaps = [(b, paired(b, reference)) for b in r['blocks']]
                overlaps = [(b, p) for b, p in overlaps if p]
                metrics['independent_overlap'] = len(overlaps)
                metrics['independent_rotation_agreement'] = sum(
                    b['rotation'] == (p[1]['rotation']-p[2]) % 4 for b, p in overlaps)
                metrics['note'] = 'Classifier agreement only; no overhead ground truth or known location'
            metrics['seconds'] = r['seconds']
            report[mode][name] = metrics
    (args.runs/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
