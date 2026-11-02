export interface FBlock { x: number; y: number; z: number; rotation: number; is_side: number; include: boolean; score?: number; quad?: number[][]; }

import { validate } from '../hash';
const NB = { xmin: 0, xmax: 0, ymin: 0, ymax: 0, zmin: 0, zmax: 0, north: [0] };

function check(blocks: FBlock[]) { // reuse the solver's own validator
  validate(blocks.map(k => ({ x: k.x, y: k.y, z: k.z, rotation: k.rotation, is_side: k.is_side })), NB as any);
}

/** Strict parse: integers only, |x,y,z| <= 1e6, rotation 0..3 (0..1 if side), side 0/1. 4-field legacy lines mean is_side 0. */
export function parseFormation(text: string): FBlock[] {
  const rows = text.split(/\r?\n/).map(l => l.replace(/(#|\/\/).*$/, '').trim()).filter(Boolean);
  if (!rows.length) throw new Error('empty file');
  if (!/^\d+$/.test(rows[0])) throw new Error('first line must be the block count');
  const count = Number(rows[0]);
  const body = rows.slice(1);
  if (body.length !== count) throw new Error('count says ' + count + ' but found ' + body.length + ' lines');
  const out: FBlock[] = body.map((r, i) => {
    const t = r.split(/\s+/);
    if ((t.length !== 4 && t.length !== 5) || t.some(s => !/^-?\d+$/.test(s))) throw new Error('line ' + (i + 2) + ' must be 4 or 5 integers: ' + r);
    const v = t.map(Number);
    return { x: v[0], y: v[1], z: v[2], rotation: v[3], is_side: t.length === 5 ? v[4] : 0, include: true };
  });
  check(out);
  return out;
}

/** Base (north 0) formation text; optional comments go after the count line. */
export function formatFormation(blocks: FBlock[], comments: string[] = []): string {
  const b = blocks.filter(k => k.include); check(b);
  return [String(b.length), ...comments.map(c => '# ' + c), ...b.map(k => [k.x, k.y, k.z, k.rotation, k.is_side].join(' '))].join('\n') + '\n';
}

/** Same transform the solver applies for north n: (x,z) -> (-z,x) applied n times, rotation + n. */
export function applyNorth(blocks: FBlock[], n: number): FBlock[] {
  return blocks.filter(k => k.include).map(k => {
    let x = k.x, z = k.z;
    for (let i = 0; i < n; i++) { const t = x; x = -z; z = t; }
    return { ...k, x, z, rotation: (k.rotation + n) % (k.is_side ? 2 : 4) };
  });
}
