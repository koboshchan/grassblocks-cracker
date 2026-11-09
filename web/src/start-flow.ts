import type {FBlock} from './formation';

export const RANGES = [2000, 10000, 100000] as const;
export const DEFAULT_RANGE = 10000;
export const DETECT_BACKEND = 'auto';

export type Bounds = {xmin: number; xmax: number; ymin: number; ymax: number; zmin: number; zmax: number};
export type StartResult = {status: 'done' | 'cancelled' | 'nofaces' | 'noquad' | 'noblocks' | 'error'; stage: 'detect' | 'search' | 'idle'; blocks: FBlock[]; result?: any; message?: string};
export type StartDeps = {
  detect(req: {seeds?: number[][][]; backend: string}): Promise<{blocks: FBlock[]; reason?: string}>;
  search(blocks: FBlock[], bounds: Bounds): Promise<any>;
  onStage?(stage: 'detect' | 'search'): void;
};

export function normalizeRange(value: unknown): number {
  const n = Number(value);
  return (RANGES as readonly number[]).includes(n) ? n : DEFAULT_RANGE;
}
/** Symmetric positive x/z range; y comes from the caller unchanged. */
export function rangeBounds(range: number, y: {ymin: number; ymax: number}): Bounds {
  const r = normalizeRange(range);
  return {xmin: -r, xmax: r, ymin: y.ymin, ymax: y.ymax, zmin: -r, zmax: r};
}

/**
 * One START: detect -> solve. Auto mode with an image always detects. Manual mode grows from the user's
 * 4 corners when present (corners are copied, never modified), else solves the rows already in the table.
 */
export async function runStart(o: {
  mode: 'auto' | 'manual'; hasImage: boolean; blocks: FBlock[]; quad: number[][]; range: number; y: {ymin: number; ymax: number};
  deps: StartDeps; isCancelled: () => boolean;
}): Promise<StartResult> {
  const {deps} = o;
  let blocks = o.blocks;
  try {
    let seeds: number[][][] | undefined;
    let needDetect = o.mode === 'auto' && o.hasImage;
    if (o.mode === 'manual') {
      if (o.quad.length === 4 && o.hasImage) { seeds = [o.quad.map(p => p.slice())]; needDetect = true; }
      else if (!o.blocks.some(b => b.include)) return {status: 'noquad', stage: 'idle', blocks, message: 'Draw a tile quad (4 corners) first, or add rows.'};
    }
    if (needDetect) {
      deps.onStage?.('detect');
      const d = await deps.detect({seeds, backend: DETECT_BACKEND});
      if (o.isCancelled()) return {status: 'cancelled', stage: 'detect', blocks};
      if (!d.blocks || !d.blocks.length) return {status: 'nofaces', stage: 'detect', blocks, message: d.reason || 'No grass faces found.'};
      blocks = d.blocks;
    }
    if (!blocks.some(b => b.include)) return {status: 'noblocks', stage: 'idle', blocks, message: 'Nothing to search. Open a screenshot first.'};
    deps.onStage?.('search');
    const result = await deps.search(blocks, rangeBounds(o.range, o.y));
    return {status: o.isCancelled() ? 'cancelled' : 'done', stage: 'search', blocks, result};
  } catch (e: any) {
    return {status: o.isCancelled() ? 'cancelled' : 'error', stage: 'idle', blocks, message: String(e?.message || e)};
  }
}
