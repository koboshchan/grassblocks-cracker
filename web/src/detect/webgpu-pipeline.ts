import { WebGpuDetector } from './webgpu';

export interface Block {
  rotation: number;
  score: number;
  margin: number;
  votes: number;
  quad: number[][];
  x: number;
  y: number;
  z: number;
}

export interface Verdict {
  valid: boolean;
  rotation: number;
  score: number;
  margin: number;
  votes: number;
}

export interface DetectionResult {
  blocks: Block[];
  seeds: any[];
  camera: number[] | null;
  planeH: number[] | null;
  reason: string;
  mode: string;
  seconds: number;
  backend: string;
}

/* ---------- Linear Algebra & Geometry ---------- */

function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = A.map((r, i) => [...r, b[i]]);
  for (let j = 0; j < n; j++) {
    let k = j;
    for (let i = j + 1; i < n; i++) {
      if (Math.abs(m[i][j]) > Math.abs(m[k][j])) k = i;
    }
    [m[j], m[k]] = [m[k], m[j]];
    if (Math.abs(m[j][j]) < 1e-12) return null;
    const d = m[j][j];
    for (let t = j; t <= n; t++) m[j][t] /= d;
    for (let i = 0; i < n; i++) {
      if (i !== j) {
        const f = m[i][j];
        if (f) {
          for (let t = j; t <= n; t++) m[i][t] -= f * m[j][t];
        }
      }
    }
  }
  return m.map(r => r[n]);
}

export function homography(src: number[][], dst: number[][]): number[] | null {
  const N = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const r = new Array(8).fill(0);
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    const rows = [
      [x, y, 1, 0, 0, 0, -u * x, -u * y, u],
      [0, 0, 0, x, y, 1, -v * x, -v * y, v],
    ];
    for (const q of rows) {
      for (let a = 0; a < 8; a++) {
        r[a] += q[a] * q[8];
        for (let b = 0; b < 8; b++) {
          N[a][b] += q[a] * q[b];
        }
      }
    }
  }
  const h = solveLinear(N, r);
  return h ? [...h, 1] : null;
}

export function applyH(H: number[], x: number, y: number): [number, number] {
  const d = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / d, (H[3] * x + H[4] * y + H[5]) / d];
}

export function inv3(m: number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) return null;
  return [
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ];
}

function cross(a: number[], b: number[]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v: number[]): number {
  return Math.hypot(...v);
}

function jacobiEigen(S: number[][], n: number): { vals: number[]; vecs: number[][] } {
  const A = S.map(r => r.slice());
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    }
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-30) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[p][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { vals: A.map((r, i) => r[i]), vecs: V };
}

export function cameraFromPlane(H: number[], w: number, h: number): { P: number[]; f: number } | null {
  const cx = w / 2;
  const cy = h / 2;
  const v1 = [H[0] / H[6], H[3] / H[6]];
  const v2 = [H[1] / H[7], H[4] / H[7]];
  if (!Number.isFinite(v1[0] + v1[1] + v2[0] + v2[1])) return null;
  const f2 = -((v1[0] - cx) * (v2[0] - cx) + (v1[1] - cy) * (v2[1] - cy));
  if (!(f2 > 100)) return null;
  const f = Math.sqrt(f2);
  const Ki = (a: number[]) => [(a[0] - cx * a[2]) / f, (a[1] - cy * a[2]) / f, a[2]];
  const a = Ki([H[0], H[3], H[6]]);
  const c = Ki([H[1], H[4], H[7]]);
  let b = cross(c, a);
  const s = (norm(a) + norm(c)) / 2 / norm(b);
  b = [b[0] * s, b[1] * s, b[2] * s];
  const Kb = [f * b[0] + cx * b[2], f * b[1] + cy * b[2], b[2]];
  const P = [H[0], Kb[0], H[1], H[2], H[3], Kb[1], H[4], H[5], H[6], Kb[2], H[7], H[8]];
  // Orient +y up on screen
  const o = [P[3], P[7], P[11]];
  const u = [P[3] + P[1], P[7] + P[5], P[11] + P[9]];
  if (u[1] / u[2] > o[1] / o[2]) {
    for (const i of [1, 5, 9]) P[i] = -P[i];
  }
  return { P, f };
}

export function dltCamera(pts3: number[][], pix: number[][]): number[] | null {
  const n = pts3.length;
  const c3 = [0, 0, 0];
  const c2 = [0, 0];
  pts3.forEach(p => p.forEach((v, i) => (c3[i] += v / n)));
  pix.forEach(p => p.forEach((v, i) => (c2[i] += v / n)));
  let d3 = 0, d2 = 0;
  pts3.forEach(p => (d3 += Math.hypot(p[0] - c3[0], p[1] - c3[1], p[2] - c3[2]) / n));
  pix.forEach(p => (d2 += Math.hypot(p[0] - c2[0], p[1] - c2[1]) / n));
  const s3 = Math.sqrt(3) / (d3 || 1);
  const s2 = Math.SQRT2 / (d2 || 1);
  const S = Array.from({ length: 12 }, () => new Array(12).fill(0));
  for (let i = 0; i < n; i++) {
    const X = pts3[i].map((v, k) => (v - c3[k]) * s3);
    const u = (pix[i][0] - c2[0]) * s2;
    const v = (pix[i][1] - c2[1]) * s2;
    const V = [...X, 1];
    const r1 = [...V, 0, 0, 0, 0, ...V.map(t => -u * t)];
    const r2 = [0, 0, 0, 0, ...V, ...V.map(t => -v * t)];
    for (const r of [r1, r2]) {
      for (let a = 0; a < 12; a++) {
        for (let b = 0; b < 12; b++) {
          S[a][b] += r[a] * r[b];
        }
      }
    }
  }
  const { vals, vecs } = jacobiEigen(S, 12);
  let mi = 0;
  vals.forEach((v, i) => {
    if (v < vals[mi]) mi = i;
  });
  const Pn = vecs.map(r => r[mi]);
  const T3 = [
    [s3, 0, 0, -s3 * c3[0]],
    [0, s3, 0, -s3 * c3[1]],
    [0, 0, s3, -s3 * c3[2]],
    [0, 0, 0, 1],
  ];
  const T2i = [
    [1 / s2, 0, c2[0]],
    [0, 1 / s2, c2[1]],
    [0, 0, 1],
  ];
  const out = new Array(12).fill(0);
  const tmp = [0, 1, 2].map(i =>
    [0, 1, 2, 3].map(j => {
      let s = 0;
      for (let k = 0; k < 4; k++) s += Pn[i * 4 + k] * T3[k][j];
      return s;
    })
  );
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += T2i[i][k] * tmp[k][j];
      out[i * 4 + j] = s;
    }
  }
  const sc = out[11];
  return Math.abs(sc) < 1e-12 ? null : out.map(v => v / sc);
}

function reprojErr(P: number[], p3: number[][], pix: number[][]): number {
  let s = 0;
  for (let i = 0; i < p3.length; i++) {
    const [X, Y, Z] = p3[i];
    const w = P[8] * X + P[9] * Y + P[10] * Z + P[11];
    const u = (P[0] * X + P[1] * Y + P[2] * Z + P[3]) / w;
    const v = (P[4] * X + P[5] * Y + P[6] * Z + P[7]) / w;
    s += Math.hypot(u - pix[i][0], v - pix[i][1]);
  }
  return s / p3.length;
}

function isConvex(q: number[][]): boolean {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i];
    const b = q[(i + 1) % 4];
    const c = q[(i + 2) % 4];
    const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (!z) return false;
    if (!s) s = Math.sign(z);
    else if (Math.sign(z) !== s) return false;
  }
  return true;
}

function minSide(q: number[][]): number {
  return Math.min(
    ...q.map((p, i) => Math.hypot(p[0] - q[(i + 1) % 4][0], p[1] - q[(i + 1) % 4][1]))
  );
}

function inImage(q: number[][], w: number, h: number): boolean {
  return q.every(
    p => Number.isFinite(p[0]) && p[0] >= 0 && p[1] >= 0 && p[0] < w - 1 && p[1] < h - 1
  );
}

function gridQuad(H: number[], x: number, z: number): number[][] {
  return [
    applyH(H, x, z),
    applyH(H, x + 1, z),
    applyH(H, x + 1, z + 1),
    applyH(H, x, z + 1),
  ];
}

function projectCamQuad(
  P: number[],
  x: number,
  y: number,
  z: number,
  w: number,
  h: number
): number[][] | null {
  const corners = [
    [x, y, z],
    [x + 1, y, z],
    [x + 1, y, z + 1],
    [x, y, z + 1],
  ];
  const quad: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [X, Y, Z] = corners[i];
    const d = P[8] * X + P[9] * Y + P[10] * Z + P[11];
    if (d <= 0) return null;
    const u = (P[0] * X + P[1] * Y + P[2] * Z + P[3]) / d;
    if (!(u >= 0 && u < w - 1)) return null;
    const v = (P[4] * X + P[5] * Y + P[6] * Z + P[7]) / d;
    if (!(v >= 0 && v < h - 1)) return null;
    quad.push([u, v]);
  }
  if (!isConvex(quad)) return null;
  for (let i = 0; i < 4; i++) {
    const dx = quad[i][0] - quad[(i + 1) % 4][0];
    const dy = quad[i][1] - quad[(i + 1) % 4][1];
    if (dx * dx + dy * dy < 64) return null;
  }
  return quad;
}

export function canonicalQuad(q: number[][]): number[][] {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    if (q[i][0] + q[i][1] < q[s][0] + q[s][1]) s = i;
  }
  const out = [0, 1, 2, 3].map(i => q[(s + i) % 4]);
  let area = 0;
  for (let i = 0; i < 4; i++) {
    area += out[i][0] * out[(i + 1) % 4][1] - out[(i + 1) % 4][0] * out[i][1];
  }
  return area < 0 ? [out[0], out[3], out[2], out[1]] : out;
}

/* ---------- WebGPU Batch Dispatch Helper ---------- */

async function classifyQuadsOnGpu(
  detector: WebGpuDetector,
  quads: number[][][]
): Promise<(Verdict | null)[]> {
  const n = quads.length;
  if (n === 0) return [];

  const flat = new Float32Array(n * 8);
  let idx = 0;
  for (let i = 0; i < n; i++) {
    const q = quads[i];
    for (let k = 0; k < 4; k++) {
      flat[idx++] = q[k][0];
      flat[idx++] = q[k][1];
    }
  }

  const CHUNK_SIZE = 16384;
  const results: (Verdict | null)[] = new Array(n);

  for (let offset = 0; offset < n; offset += CHUNK_SIZE) {
    const count = Math.min(CHUNK_SIZE, n - offset);
    const sub = flat.subarray(offset * 8, (offset + count) * 8);
    const rawOut = await detector.classifyBatch(sub);
    for (let i = 0; i < count; i++) {
      const valid = rawOut[i * 5 + 0] > 0.5;
      if (valid) {
        results[offset + i] = {
          valid: true,
          rotation: Math.round(rawOut[i * 5 + 1]),
          score: rawOut[i * 5 + 2],
          margin: rawOut[i * 5 + 3],
          votes: Math.round(rawOut[i * 5 + 4]),
        };
      } else {
        results[offset + i] = null;
      }
    }
  }

  return results;
}

/* ---------- Pipeline Stages ---------- */

export function planeFromSeeds(
  seeds: { quad: number[][] }[]
): { H: number[]; ids: number[] } | null {
  let best: { H: number[]; ids: number[] } | null = null;
  const unit = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  for (const s of seeds) {
    const H0 = homography(unit, s.quad);
    const I = H0 && inv3(H0);
    if (!I) continue;
    const ids: number[] = [];
    const src: number[][] = [];
    const dst: number[][] = [];
    for (let k = 0; k < seeds.length; k++) {
      const loc = seeds[k].quad.map(p => applyH(I, p[0], p[1]));
      const off = [Math.round(loc[0][0]), Math.round(loc[0][1])];
      const err = Math.max(
        ...loc.map((p, i) =>
          Math.hypot(p[0] - (unit[i][0] + off[0]), p[1] - (unit[i][1] + off[1]))
        )
      );
      if (err < 0.2) {
        ids.push(k);
        unit.forEach((u, i) => {
          src.push([u[0] + off[0], u[1] + off[1]]);
          dst.push(seeds[k].quad[i]);
        });
      }
    }
    const H = src.length >= 4 ? homography(src, dst) : null;
    if (H && (!best || ids.length > best.ids.length)) {
      best = { H, ids };
    }
  }
  return best;
}

function generateCandidateGridQuads(w: number, h: number): number[][][] {
  const quads: number[][][] = [];
  const minDim = Math.min(w, h);
  const sizes = [24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320].filter(
    s => s < minDim * 0.7
  );

  for (const s of sizes) {
    const step = Math.max(6, Math.floor(s / 4));
    for (let y = 0; y <= h - s; y += step) {
      for (let x = 0; x <= w - s; x += step) {
        quads.push([
          [x, y],
          [x + s, y],
          [x + s, y + s],
          [x, y + s],
        ]);
        const halfS = s / 2;
        const hS = Math.round(s * 0.28);
        if (y + hS * 2 < h) {
          quads.push([
            [x + halfS, y],
            [x + s, y + hS],
            [x + halfS, y + hS * 2],
            [x, y + hS],
          ]);
        }
      }
    }
  }
  return quads;
}

export async function growPlane(
  detector: WebGpuDetector,
  img: { width: number; height: number },
  H: number[],
  range: number,
  accept: number,
  margin: number
): Promise<Block[]> {
  const cand: { q: number[][]; x: number; z: number }[] = [];
  for (let z = -range; z <= range; z++) {
    for (let x = -range; x <= range; x++) {
      const q = gridQuad(H, x, z);
      if (!inImage(q, img.width, img.height) || !isConvex(q) || minSide(q) < 12) continue;
      cand.push({ q, x, z });
    }
  }

  if (cand.length === 0) return [];

  const vs = await classifyQuadsOnGpu(
    detector,
    cand.map(c => c.q)
  );
  const out: Block[] = [];
  for (let i = 0; i < cand.length; i++) {
    const v = vs[i];
    if (v && v.valid && v.score >= accept && v.margin >= margin && v.votes === 4) {
      out.push({
        rotation: v.rotation,
        score: v.score,
        margin: v.margin,
        votes: v.votes,
        quad: cand[i].q,
        x: cand[i].x,
        y: 0,
        z: cand[i].z,
      });
    }
  }
  return out;
}

export async function voxelStage(
  detector: WebGpuDetector,
  img: { width: number; height: number },
  planeH: number[],
  opts: { relaxed: number; margin: number; refitFlat: boolean },
  progress: (t: string) => void
): Promise<{ blocks: Block[]; P: number[]; heights: number[] } | { reason: string }> {
  const cam = cameraFromPlane(planeH, img.width, img.height);
  if (!cam) return { reason: 'plane too fronto-parallel to estimate a camera' };

  let P = cam.P;
  let best: { blocks: Block[]; P: number[]; heights: number[] } | null = null;
  let prevCount = -1;
  let heights: number[] = [0];

  progress(`3D: initial camera from the ground plane (f=${cam.f.toFixed(0)} px)`);

  for (let round = 0; round < 4; round++) {
    const cand: { q: number[][]; x: number; y: number; z: number }[] = [];
    const R = 30;
    const YR = 12;

    for (let y = -YR; y <= YR; y++) {
      for (let z = -R; z <= R; z++) {
        for (let x = -R; x <= R; x++) {
          const q = projectCamQuad(P, x, y, z, img.width, img.height);
          if (!q) continue;
          cand.push({ q, x, y, z });
        }
      }
    }

    if (cand.length === 0) break;

    const vs = await classifyQuadsOnGpu(
      detector,
      cand.map(c => c.q)
    );
    const accept: Block[] = [];
    for (let i = 0; i < cand.length; i++) {
      const v = vs[i];
      if (v && v.valid && v.score > opts.relaxed && v.margin > opts.margin && v.votes === 4) {
        accept.push({
          rotation: v.rotation,
          score: v.score,
          margin: v.margin,
          votes: v.votes,
          quad: cand[i].q,
          x: cand[i].x,
          y: cand[i].y,
          z: cand[i].z,
        });
      }
    }

    accept.sort((a, b) => b.score - a.score);

    // One voxel per screen position (suppress rays grazing multiple heights)
    const kept: Block[] = [];
    const centers: [number, number][] = [];
    const sides: number[] = [];

    for (const b of accept) {
      const cx = (b.quad[0][0] + b.quad[1][0] + b.quad[2][0] + b.quad[3][0]) / 4;
      const cy = (b.quad[0][1] + b.quad[1][1] + b.quad[2][1] + b.quad[3][1]) / 4;
      const sd = minSide(b.quad);
      let overlap = false;
      for (let j = 0; j < kept.length; j++) {
        if (Math.hypot(centers[j][0] - cx, centers[j][1] - cy) < 0.5 * Math.min(sd, sides[j])) {
          overlap = true;
          break;
        }
      }
      if (!overlap) {
        kept.push(b);
        centers.push([cx, cy]);
        sides.push(sd);
      }
    }

    heights = Array.from(new Set(kept.map(b => b.y))).sort((a, b) => a - b);
    progress(
      `3D round ${round + 1}: ${cand.length} voxel faces tested, ${kept.length} accepted over ${heights.length} height level(s)`
    );

    if (!best || kept.length > best.blocks.length) {
      best = { blocks: kept, P, heights };
    }

    if (kept.length <= prevCount || kept.length < (opts.refitFlat && heights.length === 1 ? 3 : 6)) {
      break;
    }
    prevCount = kept.length;

    if (heights.length === 1) {
      if (!opts.refitFlat) break;
      const src: number[][] = [];
      const dst: number[][] = [];
      for (const b of kept) {
        [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(u => {
          src.push([b.x + u[0], b.z + u[1]]);
        });
        b.quad.forEach(p => dst.push(p));
      }
      const H = homography(src, dst);
      const flat = H && cameraFromPlane(H, img.width, img.height);
      if (!flat) break;
      P = flat.P;
      for (const i of [0, 4, 8]) {
        P[i + 3] -= P[i + 1] * heights[0];
      }
      progress('3D: flat-height camera refit');
      continue;
    }

    // Camera refit from 3D-2D correspondences via DLT
    const p3: number[][] = [];
    const pix: number[][] = [];
    for (const b of kept) {
      [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(u => {
        p3.push([b.x + u[0], b.y, b.z + u[1]]);
      });
      b.quad.forEach(p => pix.push(p));
    }
    const P2 = dltCamera(p3, pix);
    if (!P2) break;
    const err = reprojErr(P2, p3, pix);
    progress(`3D: camera refit mean reprojection ${err.toFixed(2)} px`);
    if (!(err < 3)) break;
    P = P2;
  }

  return best ? { blocks: best.blocks, P: best.P, heights: best.heights } : { reason: 'no voxels verified' };
}

/* ---------- Main Pipeline Function ---------- */

export async function runWebGpuDetection(
  detector: WebGpuDetector,
  img: { width: number; height: number; data: Uint8Array },
  ref: number[][],
  opts: any = {},
  progress: (t: string) => void = () => {}
): Promise<DetectionResult> {
  const startTime = performance.now();

  const accept = opts?.accept ?? 0.88;
  const margin = opts?.margin ?? opts?.acceptMargin ?? 0.35;
  const relaxed = opts?.relaxed ?? 0.78;
  const refitPlane = !!opts?.refitPlane;
  const refitFlat = !!opts?.refitFlat;

  progress('Initializing WebGPU detector image and reference patterns');
  detector.setImage(img.data, img.width, img.height);
  detector.setRefs(ref);

  const manual = Array.isArray(opts?.seeds) && opts.seeds.length > 0;
  let seeds: { rotation: number; score: number; margin: number; votes: number; quad: number[][] }[] = [];

  if (manual) {
    const rawSeeds = opts.seeds.map(canonicalQuad);
    progress(`Verifying ${rawSeeds.length} manual seed(s) on GPU`);
    const vs = await classifyQuadsOnGpu(detector, rawSeeds);
    for (let i = 0; i < rawSeeds.length; i++) {
      const v = vs[i];
      if (v && v.valid && v.score >= accept && v.margin >= margin && v.votes === 4) {
        seeds.push({
          rotation: v.rotation,
          score: v.score,
          margin: v.margin,
          votes: v.votes,
          quad: rawSeeds[i],
        });
      }
    }
  } else {
    progress('Generating multi-scale grid candidate seeds');
    const candQuads = generateCandidateGridQuads(img.width, img.height);
    progress(`Classifying ${candQuads.length} candidate seeds on GPU`);
    const vs = await classifyQuadsOnGpu(detector, candQuads);
    const verified: typeof seeds = [];
    for (let i = 0; i < candQuads.length; i++) {
      const v = vs[i];
      if (v && v.valid && v.score >= accept && v.margin >= margin && v.votes === 4) {
        verified.push({
          rotation: v.rotation,
          score: v.score,
          margin: v.margin,
          votes: v.votes,
          quad: candQuads[i],
        });
      }
    }
    verified.sort((a, b) => b.score - a.score);

    // Non-maximum suppression over candidate seeds
    for (const s of verified) {
      const cx = (s.quad[0][0] + s.quad[1][0] + s.quad[2][0] + s.quad[3][0]) / 4;
      const cy = (s.quad[0][1] + s.quad[1][1] + s.quad[2][1] + s.quad[3][1]) / 4;
      const sd = minSide(s.quad);
      if (
        !seeds.some(k => {
          const kcx = (k.quad[0][0] + k.quad[1][0] + k.quad[2][0] + k.quad[3][0]) / 4;
          const kcy = (k.quad[0][1] + k.quad[1][1] + k.quad[2][1] + k.quad[3][1]) / 4;
          const ksd = minSide(k.quad);
          return Math.hypot(cx - kcx, cy - kcy) < 0.5 * Math.min(sd, ksd);
        })
      ) {
        seeds.push(s);
      }
    }
  }

  if (seeds.length === 0) {
    const reason = manual
      ? 'your quad did not verify as a grass top (correlation too low). Make the corners tighter.'
      : 'no reliable grass tile found automatically. Draw one tile quad by hand.';
    return {
      blocks: [],
      seeds: [],
      camera: null,
      planeH: null,
      reason,
      mode: '',
      seconds: (performance.now() - startTime) / 1000,
      backend: 'WebGPU detector',
    };
  }

  progress(`Fitting 2D plane homography from ${seeds.length} verified seed(s)`);
  const plane = planeFromSeeds(seeds);
  if (!plane) {
    return {
      blocks: [],
      seeds,
      camera: null,
      planeH: null,
      reason: 'seeds do not share a plane lattice',
      mode: '',
      seconds: (performance.now() - startTime) / 1000,
      backend: 'WebGPU detector',
    };
  }

  progress(`Growing ${plane.ids.length} coplanar seed(s) across the ground plane`);
  let blocks = await growPlane(detector, img, plane.H, 28, accept, margin);
  let mode = 'planar lattice (all y=0)';
  let camera: number[] | null = null;

  if (refitPlane) {
    for (let round = 0; round < 3 && blocks.length >= 3; round++) {
      const src: number[][] = [];
      const dst: number[][] = [];
      for (const b of blocks) {
        [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(u => {
          src.push([b.x + u[0], b.z + u[1]]);
        });
        b.quad.forEach(p => dst.push(p));
      }
      const H = homography(src, dst);
      if (!H) break;
      const next = await growPlane(detector, img, H, 28, accept, margin);
      if (next.length <= blocks.length) break;
      blocks = next;
      plane.H = H;
      progress(`plane refit ${round + 1}: ${blocks.length} verified faces`);
    }
  }

  progress('Trying 3D voxel lattice for other heights');
  const v = await voxelStage(detector, img, plane.H, { relaxed, margin, refitFlat }, progress);
  if ('blocks' in v && v.blocks && v.blocks.length >= blocks.length) {
    blocks = v.blocks;
    mode = `3D voxel lattice (heights ${v.heights.join(',')})`;
    camera = v.P;
  }

  blocks.sort((a, b) => b.score - a.score);
  const totalSeconds = (performance.now() - startTime) / 1000;

  return {
    blocks,
    seeds,
    camera,
    planeH: plane.H,
    reason: blocks.length ? '' : 'seed found but no neighbours verified',
    mode,
    seconds: totalSeconds,
    backend: 'WebGPU detector',
  };
}
