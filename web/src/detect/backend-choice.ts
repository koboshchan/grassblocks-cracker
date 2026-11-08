export type DetectorBackend = 'wasm' | 'webgpu';
export const BACKEND_STORAGE_KEY = 'grass.detector.backend';
export function normalizeBackend(value: unknown): DetectorBackend { return value === 'webgpu' ? 'webgpu' : 'wasm'; }
type StorageAccess = () => Pick<Storage, 'getItem' | 'setItem'>;
export function readBackend(storage: StorageAccess): DetectorBackend {
  try { return normalizeBackend(storage().getItem(BACKEND_STORAGE_KEY)); } catch { return 'wasm'; }
}
export function saveBackend(storage: StorageAccess, value: unknown): DetectorBackend {
  const backend = normalizeBackend(value);
  try { storage().setItem(BACKEND_STORAGE_KEY, backend); return backend; } catch { return 'wasm'; }
}
