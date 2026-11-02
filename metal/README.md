Native grass texture solver

Keep grass_solver source and solver_common.h together. Both solvers accept browser formation.txt downloads unchanged. A file begins with a positive observation count followed by that many rows of x y z rotation is_side. Coordinates are relative signed int32 offsets, including individual heights. Rotation is 0..3 for tops, 0..1 for sides; is_side is 0 or 1. Blank lines and # comments are accepted anywhere. Extra fields after the five row values, such as confidence=0.95, are ignored. Missing files, empty formations, truncated rows, extra rows, invalid numbers and unsupported rotations fail explicitly. There is no fallback formation.

Run with --formation FILE and --north 0|1|2|3|all, followed optionally by six inclusive positional bounds xmin xmax ymin ymax zmin zmax. Defaults are -6000 6000 60 256 -6000 6000. Bounds and every offset addition are checked before GPU work. Volumes that overflow uint64 are rejected.

The north convention here transforms relative x,z as n0 (x,z), n1 (-z,x), n2 (-x,-z), n3 (z,-x), and adds n to rotation modulo 4 (modulo 2 for sides). --north all searches these four interpretations separately. Do not apply a north rotation twice. Old auto-formation-20261101/auto.html exported already-transformed world-relative coordinates and uses the inverse direction for its north selector. Use --north 0 for its already oriented exports. New browser downloads must use raw observations for this CLI north convention; explicit previously world-oriented exports also use --north 0.

Stdout contains only x y z north, one candidate per line. Stderr reports progress, measured candidates/second, match count and final total tested. Redirect stdout to a file for sparse or very large result sets. All matches are emitted, with no 100-result cap. Each dispatch evaluates at most 262144 candidate origins and writes one byte per origin, eliminating append-buffer overflow or atomic result loss. Ctrl+C stops between dispatches, prints partial totals and exits 130. A GPU command already in flight must finish first. Check exit status before treating output as a complete search.

Each origin uses the modern vanilla Java block-coordinate hash: signed int32 x*3129871 wrap before long extension, signed int64 z*116129781, xor with y, wrapping 64-bit l*l*42317861+l*11, then Java Random's first nextInt(4), modulo 2 for sides. Arithmetic uses fixed-width types on CUDA, including Windows where long is only 32 bits. Distinct north labels are distinct interpretations; no deduplication across north labels is attempted.

Build on macOS with Xcode command-line tools using clang++ -std=c++17 -O3 -fobjc-arc grass_solver.mm -framework Foundation -framework Metal -o grass_solver.

Run ./grass_solver --formation hill22.txt --north all 3535 3541 88 92 -457 -451. Expected sole output is 3538 90 -454 0.

Run python3 test_solver.py ./grass_solver. Tests compare exact candidate sets with an independent arbitrary-precision Python oracle, including negative and large wrapped coordinates, mixed heights, side textures, every north, 5043 dense results and malformed/overflow inputs. Only tiny GPU workloads are used.

Actual Apple M1 source and runtime Metal shader compilation and all regressions passed November 1, 2026, using Xcode-beta clang++. Test details are in TEST_LOG.txt. Remote test files are isolated in ~/grassblocks/native-solvers-20261101. Original TextureRotations sources were not modified.
