// @ts-nocheck

export const DEFAULTS = { accept: 0.88, acceptMargin: 0.35, relaxed: 0.78, coarse: 0.55 };

/* ---------- small linear algebra ---------- */
function solveLinear(A, b) {
  const n = b.length, m = A.map((r, i) => [...r, b[i]]);
  for (let j = 0; j < n; j++) {
    let k = j; for (let i = j + 1; i < n; i++) if (Math.abs(m[i][j]) > Math.abs(m[k][j])) k = i;
    [m[j], m[k]] = [m[k], m[j]];
    if (Math.abs(m[j][j]) < 1e-12) return null;
    const d = m[j][j]; for (let t = j; t <= n; t++) m[j][t] /= d;
    for (let i = 0; i < n; i++) if (i !== j) { const f = m[i][j]; if (f) for (let t = j; t <= n; t++) m[i][t] -= f * m[j][t]; }
  }
  return m.map(r => r[n]);
}
export function homography(src, dst) { // exact/least squares, h33 = 1; returns flat 9 or null
  const N = Array.from({ length: 8 }, () => new Array(8).fill(0)), r = new Array(8).fill(0);
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    const rows = [[x, y, 1, 0, 0, 0, -u * x, -u * y, u], [0, 0, 0, x, y, 1, -v * x, -v * y, v]];
    for (const q of rows) for (let a = 0; a < 8; a++) { r[a] += q[a] * q[8]; for (let b = 0; b < 8; b++) N[a][b] += q[a] * q[b]; }
  }
  const h = solveLinear(N, r); return h ? [...h, 1] : null;
}
export function applyH(H, x, y) { const d = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / d, (H[3] * x + H[4] * y + H[5]) / d]; }
export function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) return null;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
function mul3(a, b) { const o = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j]; return o; }
function jacobiEigen(S, n) { // symmetric, returns {vals, vecs(columns)}
  const A = S.map(r => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { vals: A.map((r, i) => r[i]), vecs: V };
}
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
const norm = v => Math.hypot(...v);

/* ---------- texture, sampling, verification ---------- */
export function makeRefs(ref) { // ref[y][x] 0..255 gray; 4 clockwise rotations as flat arrays
  const refs = [];
  for (let k = 0; k < 4; k++) { const a = []; for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { let xx = x, yy = y; for (let j = 0; j < k; j++) { const t = xx; xx = yy; yy = 15 - t; } a.push(ref[yy][xx]); } refs.push(a); }
  return refs;
}
function greenPlane(img) { const { width: w, height: h, data } = img, g = new Float32Array(w * h); for (let i = 0; i < g.length; i++) g[i] = data[4 * i + 1]; return g; }
function corr(a, b) {
  const n = a.length; let sa = 0, sb = 0, aa = 0, bb = 0, ab = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; ab += a[i] * b[i]; }
  const va = aa - sa * sa / n, vb = bb - sb * sb / n; if (va < 1e-9 || vb < 1e-9) return 0;
  return (ab - sa * sb / n) / Math.sqrt(va * vb);
}
const UNIT16 = [[0, 0], [16, 0], [16, 16], [0, 16]];
function samplePatch(g, w, h, quad) { // 16x16 bilinear patch, null if outside image
  const H = homography(UNIT16, quad); if (!H) return null;
  const out = new Float32Array(256);
  for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) {
    const [px, py] = applyH(H, i + .5, j + .5);
    if (!(px >= 0 && py >= 0 && px < w - 1 && py < h - 1)) return null;
    const xx = Math.floor(px), yy = Math.floor(py), fx = px - xx, fy = py - yy;
    out[j * 16 + i] = g[yy * w + xx] * (1 - fx) * (1 - fy) + g[yy * w + xx + 1] * fx * (1 - fy) + g[(yy + 1) * w + xx] * (1 - fx) * fy + g[(yy + 1) * w + xx + 1] * fx * fy;
  }
  return out;
}
const INNER = [], QUADS = [[], [], [], []];
for (let j = 1; j < 15; j++) for (let i = 1; i < 15; i++) { INNER.push(j * 16 + i); QUADS[(i >= 8 ? 1 : 0) + (j >= 8 ? 2 : 0)].push(j * 16 + i); }
export function classify(ctx, quad) {
  const patch = samplePatch(ctx.g, ctx.w, ctx.h, quad); if (!patch) return null;
  const a = INNER.map(i => patch[i]), sc = ctx.refs.map(r => corr(a, INNER.map(i => r[i])));
  let o = [0, 1, 2, 3].sort((p, q) => sc[q] - sc[p]); const k = o[0]; let votes = 0;
  for (let q = 0; q < 4; q++) {
    const ids = QUADS[q], ar = ids.map(i => patch[i]), ss = ctx.refs.map(r => corr(ar, ids.map(i => r[i])));
    if (ss.indexOf(Math.max(...ss)) === k) votes++;
  }
  return { rotation: k, score: sc[k], margin: sc[k] - sc[o[1]], votes };
}
function isConvex(q) { let s = 0; for (let i = 0; i < 4; i++) { const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4], z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]); if (!z) return false; if (!s) s = Math.sign(z); else if (Math.sign(z) !== s) return false; } return true; }
const minSide = q => Math.min(...q.map((p, i) => Math.hypot(p[0] - q[(i + 1) % 4][0], p[1] - q[(i + 1) % 4][1])));
const inImage = (q, w, h) => q.every(p => Number.isFinite(p[0]) && p[0] >= 0 && p[1] >= 0 && p[0] < w - 1 && p[1] < h - 1);
function gridQuad(H, x, z, y = 0) { return [[x, z], [x + 1, z], [x + 1, z + 1], [x, z + 1]].map(([a, b]) => applyH(H, a, b)); }
export function camQuad(P, x, y, z) { return [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => { const X = x + a, Z = z + b, w = P[8] * X + P[9] * y + P[10] * Z + P[11]; return [(P[0] * X + P[1] * y + P[2] * Z + P[3]) / w, (P[4] * X + P[5] * y + P[6] * Z + P[7]) / w]; }); }
/* camera-ray sanity: all four corners must be in front of the camera */
function camQuadOK(P, x, y, z) { for (const [a, b] of [[0, 0], [1, 0], [1, 1], [0, 1]]) if (P[8] * (x + a) + P[9] * y + P[10] * (z + b) + P[11] <= 0) return false; return true; }

/* ---------- batch classification (JS reference; WASM kernel plugs in via ctx.classifyBatch) ---------- */
export type Verdict = { rotation: number; score: number; margin: number; votes: number } | null;
export function jsClassifyBatch(ctx: any, quads: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n * 5);
  for (let i = 0; i < n; i++) {
    const q = [[quads[i*8], quads[i*8+1]], [quads[i*8+2], quads[i*8+3]], [quads[i*8+4], quads[i*8+5]], [quads[i*8+6], quads[i*8+7]]];
    const r = classify(ctx, q); if (!r) continue;
    out.set([1, r.rotation, r.score, r.margin, r.votes], i * 5);
  }
  return out;
}
function batchClassify(ctx: any, quads: number[][][]): Verdict[] {
  const n = quads.length, flat = new Float32Array(n * 8);
  quads.forEach((q, i) => q.forEach((p, k) => { flat[i*8+k*2] = p[0]; flat[i*8+k*2+1] = p[1]; }));
  const CH = 4096, res: Verdict[] = new Array(n);
  for (let s = 0; s < n; s += CH) {
    const m = Math.min(CH, n - s), sub = flat.subarray(s * 8, (s + m) * 8);
    const o = ctx.classifyBatch ? ctx.classifyBatch(sub, m) : jsClassifyBatch(ctx, sub, m);
    for (let i = 0; i < m; i++) res[s+i] = o[i*5] ? { rotation: o[i*5+1], score: o[i*5+2], margin: o[i*5+3], votes: o[i*5+4] } : null;
    if (ctx.onChunk) ctx.onChunk(s + m, n);
  }
  return res;
}

/* ---------- context ---------- */
export function makeContext(img, ref, opts?, hooks?) {
  return { classifyBatch: hooks && hooks.classifyBatch, onChunk: hooks && hooks.onChunk, w: img.width, h: img.height, g: greenPlane(img), refs: makeRefs(ref), ref, opts: Object.assign({}, DEFAULTS, opts || {}) };
}

/* ---------- stage 1: feature seeds (OpenCV.js) ---------- */
function mat8(cv, g, w, h) { const m = new cv.Mat(h, w, cv.CV_8UC1); for (let i = 0; i < w * h; i++) m.data[i] = g[i]; return m; }
function seedsFromFeatures(cv, ctx, progress) {
  const { w, h, g } = ctx, log = progress || (() => {});
  const img = mat8(cv, g, w, h), refSmall = new cv.Mat(16, 16, cv.CV_8UC1);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) refSmall.data[y * 16 + x] = ctx.ref[y][x];
  const found = [], free = [img, refSmall];
  // The texture is tiny and noisy, so feature scale matters: try several rendered sizes (and nearest/smooth upsampling);
  // every candidate is still gated by dense correlation later.
  try {
    for (const S of [64, 96, 128]) for (const interp of [cv.INTER_NEAREST, cv.INTER_LINEAR]) for (const feat of ['akaze', 'orb']) {
      const refBig = new cv.Mat(); cv.resize(refSmall, refBig, new cv.Size(S, S), 0, 0, interp);
      const det = feat === 'akaze' ? new cv.AKAZE() : new cv.ORB(6000);
      if (feat === 'orb') { try { det.setScaleFactor(1.15); det.setNLevels(12); det.setEdgeThreshold(8); det.setFastThreshold(5); } catch (e) {} }
      const kr = new cv.KeyPointVector(), ks = new cv.KeyPointVector(), dr = new cv.Mat(), ds = new cv.Mat(), none = new cv.Mat();
      det.detectAndCompute(refBig, none, kr, dr); det.detectAndCompute(img, none, ks, ds);
      let P = [], Q = [];
      if (dr.rows >= 4 && ds.rows >= 4) {
        const bf = new cv.BFMatcher(cv.NORM_HAMMING, false), mv = new cv.DMatchVectorVector();
        bf.knnMatch(ds, dr, mv, 2);
        for (let i = 0; i < mv.size(); i++) {
          const m = mv.get(i); if (m.size() < 2) continue; const a = m.get(0), b = m.get(1);
          if (a.distance < 0.85 * b.distance) { const r = kr.get(a.trainIdx).pt, s = ks.get(a.queryIdx).pt; P.push([r.x, r.y]); Q.push([s.x, s.y]); }
        }
        bf.delete(); mv.delete();
      }
      [kr, ks, dr, ds, none, det, refBig].forEach(o => o.delete());
      let got = 0;
      for (let it = 0; it < 30 && P.length >= 6; it++) {
        const sp = cv.matFromArray(P.length, 1, cv.CV_32FC2, P.flat()), dp = cv.matFromArray(Q.length, 1, cv.CV_32FC2, Q.flat()), mask = new cv.Mat();
        const Hm = cv.findHomography(sp, dp, cv.RANSAC, 4.0, mask, 4000, 0.995);
        sp.delete(); dp.delete();
        if (!Hm || Hm.empty()) { mask.delete(); if (Hm) Hm.delete(); break; }
        const H = Array.from(Hm.data64F); Hm.delete();
        const inl = []; let n = 0; for (let i = 0; i < P.length; i++) { const k = !!mask.data[i]; inl.push(k); if (k) n++; }
        mask.delete();
        if (n < 5) break;
        P = P.filter((_, i) => !inl[i]); Q = Q.filter((_, i) => !inl[i]);
        const q = UNIT16.map(([x, y]) => applyH(H, x * S / 16, y * S / 16));
        if (!isConvex(q) || !inImage(q, w, h) || minSide(q) < 14) continue;
        const area = Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0)) / 2;
        if (area > w * h * 0.25) continue;
        found.push(canonicalQuad(q)); got++;
      }
      log(feat + ' @' + S + (interp === cv.INTER_NEAREST ? 'n' : 'l') + ': ' + got + ' tile hypothesis(es)');
    }
  } finally { free.forEach(o => o.delete()); }
  return found;
}
function canonicalQuad(q) { // start from min x+y corner, keep winding
  let s = 0; q.forEach((p, i) => { if (p[0] + p[1] < q[s][0] + q[s][1]) s = i; });
  q = [0, 1, 2, 3].map(i => q[(s + i) % 4]);
  const area = q.reduce((a, p, i) => a + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0);
  return area < 0 ? [q[0], q[3], q[2], q[1]] : q; // always clockwise on screen (y down => positive area)
}

/* ---------- ECC refinement (OpenCV.js) ---------- */
function makeRefiner(cv, ctx) {
  const { w, h, g } = ctx, img = new cv.Mat(h, w, cv.CV_32FC1);
  for (let i = 0; i < w * h; i++) img.data32F[i] = g[i] / 255;
  const targets = [0, 1, 2, 3].map(k => { const m = new cv.Mat(128, 128, cv.CV_32FC1); for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) m.data32F[y * 128 + x] = ctx.refs[k][(y >> 3) * 16 + (x >> 3)] / 255; return m; });
  const crit = new cv.TermCriteria(cv.TermCriteria_COUNT | cv.TermCriteria_EPS, 60, 1e-5);
  const sq = [[0, 0], [128, 0], [128, 128], [0, 128]];
  const fn = function (quad, rot) {
    const base = homography(sq, quad); if (!base) return null; const bi = inv3(base); if (!bi) return null;
    const wm = new cv.Mat(), M = cv.matFromArray(3, 3, cv.CV_64FC1, bi), warped = new cv.Mat(), Wm = cv.matFromArray(3, 3, cv.CV_32FC1, [1, 0, 0, 0, 1, 0, 0, 0, 1]), none = new cv.Mat();
    try {
      cv.warpPerspective(img, warped, M, new cv.Size(128, 128), cv.INTER_LINEAR | cv.WARP_INVERSE_MAP * 0, cv.BORDER_REPLICATE);
      cv.findTransformECC(targets[rot], warped, Wm, cv.MOTION_HOMOGRAPHY, crit, none, 3);
      const Wh = Array.from(Wm.data32F), full = mul3(base, Wh), nq = sq.map(([x, y]) => applyH(full, x, y));
      return nq;
    } catch (e) { return null; } finally { [wm, M, warped, Wm, none].forEach(o => o.delete()); }
  };
  fn.free = () => { img.delete(); targets.forEach(t => t.delete()); };
  return fn;
}
function verify(ctx, quad, refine, thr, r0?) {
  let r = r0 === undefined ? classify(ctx, quad) : r0; if (!r) return null;
  if (refine && r.score > thr.coarse && r.margin > 0.25) {
    const nq = refine(quad, r.rotation);
    if (nq && isConvex(nq) && inImage(nq, ctx.w, ctx.h) && Math.max(...nq.map((p, i) => Math.hypot(p[0] - quad[i][0], p[1] - quad[i][1]))) < 3.5) {
      const r2 = classify(ctx, nq); if (r2 && r2.score > r.score) { quad = nq; r = r2; }
    }
  }
  return Object.assign({ quad }, r);
}

/* ---------- stage 2: ground-plane lattice from seeds ---------- */
export function planeFromSeeds(seeds) { // seeds: [{quad,...}] -> {H, ids} (H maps lattice -> pixels)
  let best = null; const unit = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const s of seeds) {
    const H0 = homography(unit, s.quad), I = H0 && inv3(H0); if (!I) continue;
    let ids = [], src = [], dst = [];
    for (let k = 0; k < seeds.length; k++) {
      const loc = seeds[k].quad.map(p => applyH(I, p[0], p[1])), off = [Math.round(loc[0][0]), Math.round(loc[0][1])];
      const err = Math.max(...loc.map((p, i) => Math.hypot(p[0] - (unit[i][0] + off[0]), p[1] - (unit[i][1] + off[1]))));
      if (err < 0.2) { ids.push(k); unit.forEach((u, i) => { src.push([u[0] + off[0], u[1] + off[1]]); dst.push(seeds[k].quad[i]); }); }
    }
    const H = src.length >= 4 ? homography(src, dst) : null;
    if (H && (!best || ids.length > best.ids.length)) best = { H, ids };
  }
  return best;
}
export function growPlane(ctx, H, refine, range, progress?) {
  const out = [], thr = ctx.opts, cand = [];
  for (let z = -range; z <= range; z++) for (let x = -range; x <= range; x++) {
    const q = gridQuad(H, x, z); if (!inImage(q, ctx.w, ctx.h) || !isConvex(q) || minSide(q) < 12) continue; cand.push({ q, x, z });
  }
  const v = batchClassify(ctx, cand.map(c => c.q));
  cand.forEach((c, i) => { const r = verify(ctx, c.q, refine, thr, v[i]); if (r && r.score >= thr.accept && r.margin >= thr.acceptMargin && r.votes === 4) out.push(Object.assign(r, { x: c.x, y: 0, z: c.z })); });
  return out;
}

/* ---------- stage 3: camera + voxel lattice ---------- */
function cameraFromPlane(H, w, h) {
  const cx = w / 2, cy = h / 2, v1 = [H[0] / H[6], H[3] / H[6]], v2 = [H[1] / H[7], H[4] / H[7]];
  if (!isFinite(v1[0] + v1[1] + v2[0] + v2[1])) return null;
  const f2 = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy)); if (!(f2 > 100)) return null;
  const f = Math.sqrt(f2), Ki = a => [(a[0] - cx * a[2]) / f, (a[1] - cy * a[2]) / f, a[2]];
  const a = Ki([H[0], H[3], H[6]]), c = Ki([H[1], H[4], H[7]]);
  let b = cross(c, a); const s = (norm(a) + norm(c)) / 2 / norm(b); b = b.map(v => v * s);
  const Kb = [f * b[0] + cx * b[2], f * b[1] + cy * b[2], b[2]];
  const P = [H[0], Kb[0], H[1], H[2], H[3], Kb[1], H[4], H[5], H[6], Kb[2], H[7], H[8]];
  // make +y point up on screen
  const o = [P[3], P[7], P[11]], u = [P[3] + P[1], P[7] + P[5], P[11] + P[9]];
  if (u[1] / u[2] > o[1] / o[2]) for (const i of [1, 5, 9]) P[i] = -P[i];
  return { P, f };
}
export function dltCamera(pts3, pix) { // normalised DLT for 3x4 P; returns flat 12 with P[11]=1 style scale
  const n = pts3.length, c3 = [0, 0, 0], c2 = [0, 0];
  pts3.forEach(p => p.forEach((v, i) => c3[i] += v / n)); pix.forEach(p => p.forEach((v, i) => c2[i] += v / n));
  let d3 = 0, d2 = 0; pts3.forEach(p => d3 += Math.hypot(p[0] - c3[0], p[1] - c3[1], p[2] - c3[2]) / n); pix.forEach(p => d2 += Math.hypot(p[0] - c2[0], p[1] - c2[1]) / n);
  const s3 = Math.sqrt(3) / (d3 || 1), s2 = Math.SQRT2 / (d2 || 1), S = Array.from({ length: 12 }, () => new Array(12).fill(0));
  for (let i = 0; i < n; i++) {
    const X = pts3[i].map((v, k) => (v - c3[k]) * s3), u = (pix[i][0] - c2[0]) * s2, v = (pix[i][1] - c2[1]) * s2, V = [...X, 1];
    const r1 = [...V, 0, 0, 0, 0, ...V.map(t => -u * t)], r2 = [0, 0, 0, 0, ...V, ...V.map(t => -v * t)];
    for (const r of [r1, r2]) for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) S[a][b] += r[a] * r[b];
  }
  const { vals, vecs } = jacobiEigen(S, 12); let mi = 0; vals.forEach((v, i) => { if (v < vals[mi]) mi = i; });
  const Pn = vecs.map(r => r[mi]); // 12 (3x4 row-major) in normalised space
  // P = T2^-1 * Pn * T3
  const T3 = [[s3, 0, 0, -s3 * c3[0]], [0, s3, 0, -s3 * c3[1]], [0, 0, s3, -s3 * c3[2]], [0, 0, 0, 1]];
  const T2i = [[1 / s2, 0, c2[0]], [0, 1 / s2, c2[1]], [0, 0, 1]], out = new Array(12).fill(0);
  const tmp = [0, 1, 2].map(i => [0, 1, 2, 3].map(j => { let s = 0; for (let k = 0; k < 4; k++) s += Pn[i * 4 + k] * T3[k][j]; return s; }));
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 3; k++) s += T2i[i][k] * tmp[k][j]; out[i * 4 + j] = s; }
  const sc = out[11]; return Math.abs(sc) < 1e-12 ? null : out.map(v => v / sc);
}
function reprojErr(P, p3, pix) { let s = 0; for (let i = 0; i < p3.length; i++) { const [X, Y, Z] = p3[i], w = P[8] * X + P[9] * Y + P[10] * Z + P[11], u = (P[0] * X + P[1] * Y + P[2] * Z + P[3]) / w, v = (P[4] * X + P[5] * Y + P[6] * Z + P[7]) / w; s += Math.hypot(u - pix[i][0], v - pix[i][1]); } return s / p3.length; }

export function voxelStage(ctx, seeds, plane, refine, progress?) {
  const log = progress || (() => {}), cam = cameraFromPlane(plane.H, ctx.w, ctx.h);
  if (!cam) return { reason: 'plane too fronto-parallel to estimate a camera' };
  let P = cam.P, best = null, prevCount = -1, heights = [0];
  log('3D: initial camera from the ground plane (f=' + cam.f.toFixed(0) + ' px)');
  for (let round = 0; round < 4; round++) {
    const thr = ctx.opts, accept = [], cand = [], R = 30, YR = 12;
    for (let y = -YR; y <= YR; y++) for (let z = -R; z <= R; z++) for (let x = -R; x <= R; x++) {
      if (!camQuadOK(P, x, y, z)) continue;
      const q = camQuad(P, x, y, z); if (!inImage(q, ctx.w, ctx.h) || !isConvex(q) || minSide(q) < 8) continue;
      cand.push({ q, x, y, z });
    }
    const vs = batchClassify(ctx, cand.map(c => c.q));
    cand.forEach((c, i) => { const r = verify(ctx, c.q, refine, thr, vs[i]); if (r && r.score > thr.relaxed && r.margin > thr.acceptMargin && r.votes === 4) accept.push(Object.assign(r, { x: c.x, y: c.y, z: c.z })); });
    accept.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const b of accept) { // one voxel per screen position (a camera ray can graze several)
      const c = b.quad.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4], [0, 0]), sd = minSide(b.quad);
      if (!kept.some(k => Math.hypot(k.c[0] - c[0], k.c[1] - c[1]) < 0.5 * Math.min(sd, k.sd))) kept.push(Object.assign(b, { c, sd }));
    }
    kept.forEach(k => { delete k.c; delete k.sd; });
    heights = [...new Set(kept.map(b => b.y))].sort((a, b) => a - b);
    log('3D round ' + (round + 1) + ': ' + cand.length + ' voxel faces tested, ' + kept.length + ' accepted over ' + heights.length + ' height level(s)');
    if (!best || kept.length > best.blocks.length) best = { blocks: kept, P, heights };
    if (kept.length <= prevCount || heights.length < 2 || kept.length < 6) break;
    prevCount = kept.length;
    const p3 = [], pix = []; // self-calibration: refit full camera from every verified face
    for (const b of kept) [[0, 0], [1, 0], [1, 1], [0, 1]].forEach((u, i) => { p3.push([b.x + u[0], b.y, b.z + u[1]]); pix.push(b.quad[i]); });
    const P2 = dltCamera(p3, pix); if (!P2) break;
    const err = reprojErr(P2, p3, pix); log('3D: camera refit mean reprojection ' + err.toFixed(2) + ' px');
    if (!(err < 3)) break; P = P2;
  }
  return best ? { blocks: best.blocks, P: best.P, heights: best.heights } : { reason: 'no voxels verified' };
}

/* ---------- fallback: near-top-down repeating lattice (no features needed) ---------- */
function topDownLattice(ctx) {
  const { w, h, g } = ctx;
  const sc = 1, periods = [];
  for (let axis = 0; axis < 2; axis++) {
    const n = axis === 0 ? w : h, other = axis === 0 ? h : w, e = new Float64Array(n - 1);
    for (let j = 0; j < other; j += 3) for (let i = 0; i < n - 1; i++) { const p = axis === 0 ? j * w + i : i * w + j; e[i] += Math.abs(g[p + (axis === 0 ? 1 : w)] - g[p]); }
    const mean = e.reduce((a, b) => a + b, 0) / e.length, pk = [];
    for (let k = Math.ceil(n / 24); k < n / 2; k++) { let re = 0, im = 0; const an = 2 * Math.PI * k / e.length; for (let i = 0; i < e.length; i++) { const v = e[i] - mean; re += v * Math.cos(an * i); im += v * Math.sin(an * i); } pk.push({ p: e.length / k * 16, power: re * re + im * im }); }
    pk.sort((a, b) => b.power - a.power); periods.push(...pk.slice(0, 2).map(p => p.p));
  }
  const sizes = [...new Set(periods.flatMap(p => [-1, 0, 1].map(d => Math.round(p) + d)))].filter(p => p >= 48 && p < Math.min(w, h) * .6);
  let best = null;
  for (const n of sizes) { const st = Math.max(2, Math.floor(n / 32)); for (let y = 0; y < h - n; y += st) for (let x = 0; x < w - n; x += st) {
    const q = [[x, y], [x + n, y], [x + n, y + n], [x, y + n]], r = classify(ctx, q); if (r && r.score > .72 && (!best || r.score > best.r.score)) best = { q, r };
  } }
  return best ? [Object.assign({ quad: best.q }, best.r)] : [];
}

/* ---------- top level ---------- */
export function runDetection(cv, img, ref, opts, progress?, hooks?) {
  const log = progress || (() => {}), ctx = makeContext(img, ref, opts, hooks), notes = [];
  let refine = null; if (cv) refine = makeRefiner(cv, ctx);
  try {
    let rawSeeds = opts && opts.seeds && opts.seeds.length ? opts.seeds.map(canonicalQuad) : [];
    const manual = rawSeeds.length > 0;
    if (!manual) {
      if (!cv) return { blocks: [], reason: 'OpenCV.js unavailable', notes };
      log('finding grass tiles with feature matching');
      rawSeeds = seedsFromFeatures(cv, ctx, log);
    }
    log('verifying ' + rawSeeds.length + ' candidate tile(s)');
    let seeds = rawSeeds.map(q => verify(ctx, q, refine, ctx.opts)).filter(r => r && r.score >= ctx.opts.accept && r.margin >= ctx.opts.acceptMargin && r.votes === 4);
    notes.push(seeds.length + ' verified seed tile(s)' + (manual ? ' from your manual quad(s)' : ' from feature matching'));
    if (!seeds.length && !manual) { log('no feature seeds, trying top-down lattice scan'); seeds = topDownLattice(ctx).filter(r => r.score >= ctx.opts.accept); if (seeds.length) notes.push('top-down texture scan found a seed'); }
    if (!seeds.length) return { blocks: [], seeds: [], reason: manual ? 'your quad did not verify as a grass top (correlation too low). Make the corners tighter.' : 'no reliable grass tile found automatically. Draw one tile quad by hand.', notes };
    const plane = planeFromSeeds(seeds);
    if (!plane) return { blocks: [], seeds, reason: 'seeds do not share a plane lattice', notes };
    log('growing ' + plane.ids.length + ' coplanar seed(s) across the ground plane');
    let blocks = growPlane(ctx, plane.H, refine, 28), mode = 'planar lattice (all y=0)', camera = null;
    notes.push('plane lattice: ' + plane.ids.length + ' coplanar seeds, ' + blocks.length + ' faces');
    {
      log('trying 3D voxel lattice for other heights');
      const v = voxelStage(ctx, seeds, plane, refine, m => { notes.push(m); log(m); });
      if (v.blocks && v.blocks.length >= blocks.length) { blocks = v.blocks; mode = '3D voxel lattice (heights ' + v.heights.join(',') + ')'; camera = v.P; }
      else notes.push('3D stage: ' + (v.reason || 'no gain over planar'));
    }
    blocks.sort((a, b) => b.score - a.score);
    return { blocks, seeds, mode, camera, planeH: plane.H, notes, reason: blocks.length ? '' : 'seed found but no neighbours verified' };
  } finally { if (refine && refine.free) refine.free(); }
}

