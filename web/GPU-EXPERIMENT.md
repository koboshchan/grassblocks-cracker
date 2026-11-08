Experimental GPU ECC integration

The explicit GPU option uses GPU classification with centered compensated correlation and GPU ECC, including already-accepted eligible quads. Automatic proposals still use native SIFT. This is disclosed in diagnostics and is not a pure-GPU automatic detector. The default native WASM pool is unchanged. GPU failures stop rather than silently fall back. Nine configurations run in fixed order, with anchored union; top-down spectral proposals use GPU scoring.

34 integration tests, TypeScript checks and the production build pass. Actual integration shader execution and four-shot parity are pending browser/Mac QA. Earlier isolated stable correlation passed 17/18 frozen growth decisions, not full detector parity. No software-float emulation is included.

Precision reference review only

Mesa source was read at https://raw.githubusercontent.com/mirror/mesa/e24dc5bd1e7fe6101bdc866fb16a15a8fcae1aae/src/compiler/glsl/float64.glsl . This file has the Berkeley SoftFloat 3-clause BSD license, not the repository-default MIT license. No Mesa code is included in this integration. Its uint64 GLSL entry points have substantial normalization/rounding dependencies and cannot be copied directly into WGSL. See also https://github.com/gpuweb/gpuweb/issues/2805 for hardware and reassociation limitations. A future precision component would be isolated to frozen patch reductions in small staged kernels, with runtime bit-pattern probes on real hardware, not a repeated monolithic software-division classifier. No exactness or hardware compilation claim is made for that unimplemented component.
