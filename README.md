Grassblocks cracker

Local-only monorepo. web contains the plain TypeScript browser app, SIMD WASM detection and CPU cracker, and WebGPU shader. cuda and metal contain native command-line solvers accepting the same formation.txt.

No screenshot or search data is sent to a compute server. A static host only supplies the application and WASM files. Browser WebGPU uses the platform graphics driver, Metal on macOS and typically D3D12 on Windows, not CUDA. The downloadable CUDA solver is a separate native program.

Formation files remain backwards compatible. The first non-comment line is the observation count; each following row has x y z rotation is_side. Heights are per-observation y offsets. Extra confidence fields are comments or ignored trailing metadata. New raw observations use the solver's north interpretation n0(x,z), n1(-z,x), n2(-x,-z), n3(z,-x), rotation+r north modulo4, modulo2 for sides. Already-oriented legacy exports must be searched with north0.

Native build instructions and exact regression logs are in cuda/README.md and metal/README.md. Both were actually compiled and run on the M1 Mac and RTX4070Super, respectively. Other GPUs are supported targets, not claimed hardware benchmarks.

The hill regression has22 observations and the candidate3538,90,-454,north0. A match is evidence only inside the explicitly searched inclusive bounds, not a unique world-wide location proof. The hill example includes previously calibrated geometry and is labeled as such. Automatic geometry on arbitrary perspective terrain can require a manual quad and height corrections. Review observations before an expensive search.

See web/tests/RESEARCH.md for portable64-bit WGSL arithmetic decisions and test coverage. A32-bit-only hash is not correct for modern vanilla rotations. The polynomial's bits16..63 are relevant; its high word cannot be dropped. Floating-point fast-math cannot improve this integer hash.
