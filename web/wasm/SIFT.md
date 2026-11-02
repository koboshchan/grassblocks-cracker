VLFeat SIFT browser seed detection

The local vendor/grass-sift.js and grass-sift.wasm are built from wasm/sift.c and wasm/vlfeat/vl. Run `sh wasm/build-sift.sh` with Emscripten installed. The module is single-threaded, feature output is bounded at 20,000 descriptors and memory at 512MB. No runtime CDN, external service, or scene-specific seed coordinates are used. Screenshot and texture descriptors are extracted in WASM. Ratio matching and proposal bookkeeping are TypeScript; OpenCV WASM fits RANSAC homographies and refines ECC. Dense face sampling/correlation uses detect-wasm.wasm.

Upstream VLFeat https://github.com/vlfeat/vlfeat uses BSD-2-Clause. See COPYING, also distributed in public/vendor/LICENSE-VLFeat.txt. generic.c adds __EMSCRIPTEN__ to the constructor/destructor attribute guard so allocator/thread-local state initializes on WASM. This is required; an unpatched build traps on allocation. SIMD is disabled for portable upstream SIFT; the separate dense correlation module uses SIMD.

The previous OpenCV asset does not expose SIFT. Its ORB/AKAZE attempts found zero useful hill seeds. The generic VLFeat experiment finds 710 ratio matches from 466 template / 11,008 screenshot features. All proposed faces still require the existing correlation, margin, convexity and quadrant-vote checks. This is not proof of robustness on arbitrary screenshots; low-resolution, occluded, heavily shaded or changed textures may still require manual quads.

Vendored upstream revision 1b9075fc42fe54b42f0e937f8b9a230d8e2c7701.
