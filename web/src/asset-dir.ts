// Versioned asset directory (assets/<content-hash>/), injected by scripts/build.ts.
// Loader JS, its WASM and shaders always live together in one immutable directory.
declare const __ASSET_DIR__: string | undefined;
export const ASSET_DIR: string = typeof __ASSET_DIR__ === 'string' ? __ASSET_DIR__ : 'vendor/';
export const assetUrl = (name: string, base: string | URL): string => new URL(ASSET_DIR + name, base).href;
