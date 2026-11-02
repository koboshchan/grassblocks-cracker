Measured hardware and correctness

November 1, 2026. Rates are candidate origins per second. Each north is a separate candidate interpretation. Measurements use early rejection against the22-observation hill formation, so these rates are not comparable to exhaustive full-hash benchmarks or arbitrary weak formations. Browser search wall time includes chunk dispatch, readback, progress and event-loop yielding. CUDA times are search-loop wall time. Do not compare microbenchmark GPU event rates to either.

M1 WebGPU, Chrome154, Apple Metal3. Inclusive x/z -5000..5000, y -64..319, all four north choices.153630721536 origins in120.7982 seconds,1.2718 billion origins/s. Sole match3538 90 -454 north0. Original launch131.57 seconds,1.168B/s. JSON evidence M1-WebGPU.json; final launch-variant regression is recorded separately when complete.

RTX4070Super CUDA. Inclusive x/z -5000..5000, y60..128.6901380069 origins per north. Initial flag-buffer loop3.90925 seconds,1.7654B/s. Sparse append plus tuned launch0.224159 seconds,30.7879B/s. All four norths27605520276 origins in0.840253 seconds on a cache-hit run,32.8538B/s. Exact baseline/optimized results agree, sole3538 90 -454 north0. See ../../cuda/BENCHMARK.txt. The M1 browser run searches a different y interval and has different dispatch overhead; this is not a hardware-only apples-to-apples comparison.

M1 native Metal. Autotune correctness, invalidation and cache corruption tests actually passed. Throughput run awaits explicit release of the GPU by the browser benchmark owner, so no new native speed is claimed here. Historical~3B/s is not a measurement of this version.

M2. SSH initially identified an AppleM2, but connectivity then failed after the owned Chrome launch. No completed benchmark is claimed. A saved PID is not proof of a running or stopped process.

A100 and RTXPRO6000. No allocation or benchmark occurred. Colab preparation was source-pinned and validated offline, but the active Lumine controller requires zero provider assignments before continuing. A concurrent benchmark could block it, so these runs require a genuine nonoverlap window. Availability/quota not established. RunPod was not used. See COLAB-STATUS.md.

SIMD detection kernel. Sandbox Node24 V8,7401 projected hill patches,4384 valid. JS922.0ms, WASM74.9ms,12.31x faster. All validity, rotation, quadrant-vote and acceptance decisions agree; maximum score difference2.96e-8 due toFloat32 results. This is projection/correlation kernel timing, not full image geometry detection or a Mac measurement.

Hash correctness. JS pair arithmetic and scalar WASM100000 random signed32 coordinates; SIMD800000 lanes. GPU65536 signed32 coordinates compared to independent BigInt oracle on actualM1. CUDA100000 signed32 coordinates exact; native solvers5043 dense candidate cases plus12 mixed-height/side/negative/wrapped tests against independent Python oracle. Nine WebGPU launch variants and dense overflow readback are tested separately. No floating-point arithmetic is used for the coordinate hash.
