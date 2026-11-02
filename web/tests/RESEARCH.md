WGSL integer implementation research, November 1 2026

The current W3C WGSL spec exposes i32/u32, not a portable concrete i64/u64 type. Native wgpu integer64 features do not make shader64 portable to browsers.
https://www.w3.org/TR/WGSL/
https://github.com/gpuweb/gpuweb/issues/273

A known crypto-field implementation uses vec2<u32>(low,high) with four 16-bit limb products for 32x32 high multiplication.
https://cyberia.blog/warriors/trisha/wgpu/shaders/goldilocks.wgsl/

Chosen approach uses scalar u32 pairs, not general-purpose u64 operators. Compute unsigned high32 using 16-bit limbs, specialize the square to hi32(lo,lo)+2*lo*hi, and specialize constant multiplications. This avoids unnecessary full four-product64 multiplications and generic shifts. WGSL does not offer a portable mulExtended/mulhi intrinsic. Floating-point tricks risk bit errors and are not used.

Only polynomial bits16..63 are needed. The final Java48 LCG only needs bits46..47. Thus no arithmetic sign extension is needed after the polynomial logical right shift; the discarded sign bits never affect the low48 LCG. The seed stage still genuinely needs upper32 product information, so a blanket32-bit implementation is incorrect. x*3129871 must wrap in signed32 BEFORE extending to64; z*116129781 must not.

These are correctness and operation-count choices, not an unmeasured claim that this is fastest across every browser driver. The actual end-to-end GPU rate includes dispatch/readback and is recorded separately. Metal-native3B/s is historical, not a browser benchmark.

Independent exact tests compare pair32 JS and WASM scalar/SIMD with arbitrary-precision Java-semantics reference on100000 random signed32 coords and800000 SIMD lanes; GPU fixture uses65536 random signed32 coords on the actual device. Native CUDA/Metal tests use independent Python oracle including5043 dense matches and signed-wrap cases.
