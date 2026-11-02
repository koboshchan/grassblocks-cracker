# Build from the monorepo root: docker build -f web/Dockerfile .
FROM debian:bookworm-slim AS wasm-toolchain
RUN apt-get update && apt-get install -y --no-install-recommends clang lld ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /build
FROM wasm-toolchain AS wasm-detection
COPY web/detect-wasm.c .
RUN mkdir /out && clang --target=wasm32 -O3 -msimd128 -ffp-contract=off -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=262144 -Wl,--max-memory=536870912 -o /out/detect-wasm.wasm detect-wasm.c
FROM wasm-toolchain AS wasm-cpu-cracker
COPY web/solver.c .
RUN mkdir /out && clang --target=wasm32 -O3 -msimd128 -fwrapv -fno-builtin -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--initial-memory=262144 -o /out/solver.wasm solver.c

FROM rust:1.91-slim-bookworm AS shader-validator
RUN cargo install naga-cli --version 30.0.1 --locked --root /opt/naga

FROM node:24-bookworm-slim AS node-dependencies
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node-dependencies AS webgpu-cracker
COPY --from=shader-validator /opt/naga/bin/naga /usr/local/bin/naga
COPY web/kernel.wgsl web/solver.ts web/hash.ts web/solver-worker.ts ./
COPY web/scripts/build-webgpu.ts scripts/build-webgpu.ts
RUN npm run build:webgpu

FROM node-dependencies AS typescript-app
COPY web/ ./
COPY --from=wasm-detection /out/detect-wasm.wasm artifacts/detect-wasm.wasm
COPY --from=wasm-cpu-cracker /out/solver.wasm artifacts/solver.wasm
COPY --from=webgpu-cracker /build/artifacts/kernel.wgsl artifacts/kernel.wgsl
RUN PREBUILT_WASM=1 npm run build && rm -f dist/build-meta.json
COPY cuda/ dist/native/cuda/
COPY metal/ dist/native/metal/

FROM nginx:alpine AS runner
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=typescript-app /build/dist/ /usr/share/nginx/html/
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
