"""Offline grid-first experiment. Never supplies unproven cross-plane heights."""
from dataclasses import dataclass
import itertools
import cv2 as cv
import numpy as np

UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
UV = np.float32([[(x + .5, z + .5) for x in range(16)] for z in range(16)])


def project(H, xy):
    return cv.perspectiveTransform(np.float32([xy]), H)[0]


def ncc(a, b, mask):
    x, y = a[mask].astype(float), b[mask].astype(float)
    if len(x) < 64:
        return -1.
    x -= x.mean()
    y -= y.mean()
    return float(x @ y / max(1e-9, np.linalg.norm(x) * np.linalg.norm(y)))


@dataclass
class Classifier:
    texture: np.ndarray
    cutoff: float = .88
    margin: float = .35
    max_masked: float = .20

    def __post_init__(self):
        self.refs = [np.rot90(self.texture.astype(np.float32), -r).copy() for r in range(4)]

    def classify(self, image, quad):
        q = np.float32(quad)
        h, w = image.shape[:2]
        if not np.isfinite(q).all() or not cv.isContourConvex(q):
            return None
        if q.min() < 0 or q[:, 0].max() >= w - 1 or q[:, 1].max() >= h - 1:
            return None
        # Reject sub-texel ground foreshortening. Upsampling cannot restore lost detail.
        area = abs(cv.contourArea(q))
        alt = area / max(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1))
        if alt < 12 or area < 256:
            return None
        H = cv.getPerspectiveTransform(UNIT * 16, q)
        p = cv.perspectiveTransform(UV, H)
        patch = cv.remap(image.astype(np.float32), p[:, :, 0], p[:, :, 1], cv.INTER_LINEAR)
        if patch.ndim == 3:
            green = patch[:, :, 1]
        else:
            green = patch
        inner = np.zeros((16, 16), bool)
        inner[1:-1, 1:-1] = True
        scores = [ncc(green, r, inner) for r in self.refs]
        rotation = int(np.argmax(scores))
        # Fit per-channel affine tint/lighting, then remove ONLY gross, model-independent
        # RGB outliers. A high unmasked NCC floor prevents fitting an arbitrary occluder.
        valid = inner.copy()
        if patch.ndim == 3 and scores[rotation] >= .55:
            r = self.refs[rotation]
            X = np.stack([r[inner], np.ones(inner.sum())], axis=1)
            fitted = X @ np.linalg.lstsq(X, patch[inner], rcond=None)[0]
            residual = np.linalg.norm(patch[inner] - fitted, axis=1)
            med = np.median(residual)
            mad = np.median(abs(residual - med))
            valid[inner] = residual <= max(18., med + 3.5 * max(mad, 1.))
        masked = 1 - valid.sum() / inner.sum()
        if masked > self.max_masked:
            return None
        scores = [ncc(green, r, valid) for r in self.refs]
        order = np.argsort(scores)
        k = int(order[-1])
        score, margin = scores[k], scores[k] - scores[order[-2]]
        votes = 0
        for yy, xx in itertools.product([slice(1, 8), slice(8, 15)], repeat=2):
            mask = np.zeros_like(valid)
            mask[yy, xx] = valid[yy, xx]
            if mask.sum() < 30:
                continue
            # Small regional checks use a lower sample-count floor than global NCC.
            ss = []
            for r in self.refs:
                a, b = green[mask], r[mask]
                a, b = a - a.mean(), b - b.mean()
                ss.append(float(a @ b / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b))))
            votes += int(np.argmax(ss)) == k
        if score < self.cutoff or margin < self.margin or votes != 4:
            return None
        return dict(rotation=k, score=score, margin=margin, masked=masked,
                    votes=votes, quad=q.tolist())


def line_vps(image, max_lines=300):
    """Length-weighted deterministic RANSAC in homogeneous image coordinates."""
    h, w = image.shape[:2]
    scale = min(1., 1200 / max(h, w))
    g = cv.resize(image[:, :, 1], None, fx=scale, fy=scale)
    segments = cv.createLineSegmentDetector(cv.LSD_REFINE_STD).detect(g)[0]
    if segments is None:
        return [], dict(lines=0)
    segments = segments[:, 0] / scale
    length = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    ids = np.argsort(length)[::-1][:max_lines]
    segments, length = segments[ids], length[ids]
    keep = length > 18
    segments, length = segments[keep], length[keep]
    if len(segments) < 8:
        return [], dict(lines=len(segments))
    center = np.array([w / 2, h / 2])
    n = max(h, w)
    p = np.c_[(segments[:, :2] - center) / n, np.ones(len(segments))]
    q = np.c_[(segments[:, 2:] - center) / n, np.ones(len(segments))]
    lines = np.cross(p, q)
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    mid = (p + q) / 2
    available = np.ones(len(lines), bool)
    rng = np.random.default_rng(781)
    vps, groups = [], []
    for _ in range(3):
        idx = np.flatnonzero(available)
        if len(idx) < 5:
            break
        pairs = rng.choice(idx, (1200, 2))
        vp = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
        norms = np.linalg.norm(vp, axis=1)
        vp = vp[norms > 1e-8]
        vp /= np.linalg.norm(vp, axis=1)[:, None]
        # distance to the infinite line, divided by midpoint-to-VP ray length.
        dist = np.abs(lines @ vp.T)
        ray = vp[None, :, :2] - mid[:, None, :2] * vp[None, :, 2:3]
        angle = dist / np.maximum(np.linalg.norm(ray, axis=2), 1e-9)
        support = (angle < .012) & available[:, None]
        weights = np.minimum(length, 160)
        sums = support.T @ weights
        k = int(sums.argmax())
        group = support[:, k]
        if group.sum() < 5:
            break
        _, _, vt = np.linalg.svd(lines[group] * np.sqrt(weights[group, None]))
        v = vt[-1]
        # Return pixel coordinates with homogeneous infinity retained.
        v = np.array([n*v[0] + center[0]*v[2], n*v[1] + center[1]*v[2], v[2]])
        v /= np.linalg.norm(v)
        vps.append(v)
        groups.append(int(group.sum()))
        available[group] = False
    return vps, dict(lines=len(lines), vp_support=groups,
                     vanishing_points=[v.tolist() for v in vps])


def rectifications(image):
    """Ground orientation hypotheses; scale and phase are still unknown."""
    h, w = image.shape[:2]
    vps, meta = line_vps(image)
    result = []
    origin = np.array([w * .45, h * .76, 1.])
    for a, b in itertools.combinations(vps, 2):
        # Choose local basis with positive screen x and y where possible.
        va, vb = a.copy(), b.copy()
        da = va[:2] - origin[:2] * va[2]
        db = vb[:2] - origin[:2] * vb[2]
        if abs((da[0]*db[1] - da[1]*db[0]) / max(np.linalg.norm(da)*np.linalg.norm(db), 1e-9)) < .10:
            continue
        if abs(da[0]) < abs(db[0]):
            va, vb, da, db = vb, va, db, da
        va *= (1 if da[0] >= 0 else -1) * 80 / np.linalg.norm(da)
        vb *= (1 if db[1] >= 0 else -1) * 80 / np.linalg.norm(db)
        H = np.column_stack([va, vb, origin])
        if abs(np.linalg.det(H)) < 1e-9:
            continue
        result.append(H)
    # Fronto-parallel fallback for overhead scenes, not a fabricated VP.
    result.append(np.array([[80., 0, origin[0]], [0, 80., origin[1]], [0, 0, 1.]]))
    return result, meta


def rectify(image, H, size=800):
    # Avoid the projective horizon. This window covers ~10 blocks around the anchor
    # before texture-frequency scale calibration; candidates outside image are masked.
    pixels_per_unit = 64
    R = np.array([[1/pixels_per_unit, 0, -size/2/pixels_per_unit],
                  [0, 1/pixels_per_unit, -size/2/pixels_per_unit], [0, 0, 1.]])
    M = H @ R
    out = cv.warpPerspective(image, np.linalg.inv(M), (size, size),
                             flags=cv.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv.warpPerspective(np.ones(image.shape[:2], np.uint8),
                               np.linalg.inv(M), (size, size), flags=cv.INTER_NEAREST)
    return out, valid, M


def periods(gray, axis):
    # A grass tile is 16 texels; rectified texel edges produce spectral peaks.
    edge = np.mean(abs(np.diff(gray.astype(float), axis=axis)), axis=1-axis)
    edge -= edge.mean()
    power = abs(np.fft.rfft(edge)) ** 2
    freq = np.fft.rfftfreq(len(edge))
    allowed = np.flatnonzero((freq > 16/260) & (freq < 16/24))
    peaks = allowed[np.argsort(power[allowed])[-5:]]
    return sorted(set(int(round(16/freq[i])) for i in peaks))


def texture_proposals(image, classifier, max_proposals=120):
    """Rectify first, then anisotropic four-rotation template search."""
    transforms, meta = rectifications(image)
    candidates = []
    for plane_id, H in enumerate(transforms):
        rect, valid, M = rectify(image, H)
        green = rect[:, :, 1].astype(np.float32)
        sx, sy = periods(green, 1), periods(green, 0)
        # Frequency estimates are proposals only, never evidence of a block lattice.
        choices = [(x, y) for x in sx for y in sy if 24 <= x <= 260 and 24 <= y <= 260]
        for width, height in choices:
            for rotation, ref in enumerate(classifier.refs):
                template = cv.resize(ref, (width, height), interpolation=cv.INTER_NEAREST)
                corr = cv.matchTemplate(green, template, cv.TM_CCOEFF_NORMED)
                for _ in range(2):
                    _, score, _, (x, y) = cv.minMaxLoc(corr)
                    if score < .48:
                        break
                    qq = UNIT * [width, height] + [x, y]
                    q = project(M, qq)
                    candidates.append((score, q, plane_id))
                    cv.rectangle(corr, (max(0, x-width//2), max(0, y-height//2)),
                                 (x+width//2, y+height//2), -1, -1)
    candidates.sort(key=lambda c: -c[0])
    seeds = []
    for coarse, q, plane_id in candidates[:max_proposals]:
        r = classifier.classify(image, q)
        if r:
            r.update(proposal='rectified-lines', plane_hint=plane_id)
            seeds.append(r)
    meta.update(proposals=len(candidates), validated_seeds=len(seeds),
                rectifications=len(transforms))
    return seeds, meta


def plane_groups(seeds):
    """Do not merge unrelated height levels into one y=0 formation."""
    remaining = list(range(len(seeds)))
    groups = []
    while remaining:
        best = None
        for anchor in remaining:
            H = cv.getPerspectiveTransform(UNIT, np.float32(seeds[anchor]['quad']))
            inv = np.linalg.inv(H)
            members = []
            for idx in remaining:
                local = project(inv, seeds[idx]['quad'])
                offset = np.rint(local[0])
                error = np.max(np.linalg.norm(local - (UNIT + offset), axis=1))
                if error < .16:
                    members.append((idx, offset))
            if members and (best is None or len(members) > len(best[1])):
                best = H, members
        if best is None:
            break
        H, members = best
        src = np.concatenate([UNIT + offset for _, offset in members]).astype(np.float32)
        dst = np.concatenate([seeds[idx]['quad'] for idx, _ in members]).astype(np.float32)
        fitted, _ = cv.findHomography(src, dst, cv.RANSAC, 1.)
        if fitted is None:
            fitted = H
        groups.append(dict(H=fitted, seed_ids=[idx for idx, _ in members]))
        used = {idx for idx, _ in members}
        remaining = [idx for idx in remaining if idx not in used]
    return groups


def grow(image, classifier, H, radius=20):
    blocks = []
    for z in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            q = project(H, UNIT + [x, z])
            r = classifier.classify(image, q)
            if r:
                r.update(x=x, y=0, z=z, is_side=0)
                blocks.append(r)
    return blocks


def detect(image, texture, optional_seeds=None):
    classifier = Classifier(texture)
    seeds, meta = texture_proposals(image, classifier)
    # The optional mode is explicitly a hybrid ablation. Geometry comes from
    # the live baseline run, not stored fixture camera matrices or labeled rotations.
    for seed in optional_seeds or []:
        checked = classifier.classify(image, seed['quad'])
        if checked:
            checked['proposal'] = 'baseline-hybrid'
            seeds.append(checked)
    groups = plane_groups(seeds)
    planes = []
    for idx, group in enumerate(groups):
        blocks = grow(image, classifier, group['H'])
        planes.append(dict(id=idx, H=group['H'].tolist(), seed_ids=group['seed_ids'], blocks=blocks))
    planes.sort(key=lambda p: -len(p['blocks']))
    # Conservative export of one plane. Independent planes have unknown relative
    # integer heights/translations, so combining would silently corrupt formation.txt.
    blocks = planes[0]['blocks'] if planes else []
    return dict(blocks=blocks, planes=planes, seeds=seeds, meta=meta,
                limitation='Separate planes retained; cross-height registration unproven. Export largest plane only.')
