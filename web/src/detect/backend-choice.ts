export type DetectorBackend = 'auto' | 'wasm' | 'webgpu';
/**
 * Normalize detector backend choice. WASM is the main detection backend;
 * 'webgpu' and 'auto' (which tries deprecated WebGPU/WGSL detection before falling back to WASM)
 * are deprecated and kept for compatibility/testing only.
 */
export function normalizeBackend(value: unknown): DetectorBackend { return value === 'webgpu' ? 'webgpu' : value === 'auto' ? 'auto' : 'wasm'; }
