Colab native CUDA benchmarks remain blocked by the current Lumine chain. Neither A100 nor RTX PRO 6000 was allocated or benchmarked. Availability and quota were not established; there are no measured speeds for either chip. This preparation is for native CUDA, not browser WebGPU.

Benchmark authority is already granted for short runs after local kernel validation. No additional confirmation is requested. The owner may resume after Lumine completes and an actual nonoverlap window is established.

Before any allocation, confirm no other runtime would be affected by the benchmark session.

The preparation bundle contains a benchmark harness, a plan, and a final payload holding the exact six-file source/test/formation allowlist, pinned by final-manifest.json. Its total size is 17686 bytes. No other files belong in the source payload.

The final grass_solver.cu SHA256 was checked against FINAL_KERNEL_READY. The complete pinned allowlist follows.

grass_solver.cu  6d65dca0dff7b7f11025c99a69a6a473a34f395616193a1b2c2fa86b971a31c9
solver_common.h  805c0ca757e1f110c801e1d4b1eee13b2107af90755d4a1a684508544a4dd6a8
fixtures/hill22.txt  613bad96a64d124362fc5ef16b781ea92642625dc2d7c7bbf3153495cd8ccb7f
test_solver.py  d3df15a328be2757f492fe64049e8e29ebef72383ed8e0624d6c0ecf8ff3067e
hash_oracle.cu  e3ce4e2fc4ad041f2f669949264b4f45cc43dd1ab2f5d8684be3d761b9a57014
test_hash.py  fb6495143a7faaca66ee8a6f14a1a939f3f230bf77f621a926e57a90a8dd3210

Harness syntax/import and positive/negative manifest checks passed offline. GPU execution has not occurred. Allocation/shutdown orchestration is not implemented by the harness itself. Before any later allocation, arm an independent controller watchdog and runtime-local hard stop, enforce the total runtime deadline, then finally shut down only the owned benchmark UUID and verify fresh provider GET/LIST exclude it while preserving all other sessions. Do not use RunPod or disconnect-all. Revalidate source pins, actual GPU architecture, compiler support, account isolation and nonoverlap immediately before execution.
