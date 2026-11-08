# Build from the monorepo root: docker build -f web/Dockerfile .
FROM debian:bookworm-slim AS wasm-toolchain
RUN apt-get update && apt-get install -y --no-install-recommends clang lld ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /build
# OpenCV and the complete detector share one Emscripten instance and memory.
FROM emscripten/emsdk:3.1.64 AS wasm-pipeline
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /build
ENV GRASS_NATIVE_BUILD=/opt/grass-native-build
# Cache the pinned native libraries independently of detector changes.
COPY web/wasm/build-pipeline.sh wasm/build-pipeline.sh
RUN sh wasm/build-pipeline.sh opencv
COPY web/wasm/ wasm/
COPY web/detect-wasm.c ./
RUN sh wasm/build-pipeline.sh link

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
COPY --from=wasm-cpu-cracker /out/solver.wasm artifacts/solver.wasm
COPY --from=webgpu-cracker /build/artifacts/kernel.wgsl artifacts/kernel.wgsl
COPY --from=wasm-pipeline /build/public/vendor/grass-pipeline.js public/vendor/grass-pipeline.js
COPY --from=wasm-pipeline /build/public/vendor/grass-pipeline.wasm public/vendor/grass-pipeline.wasm
RUN PREBUILT_WASM=1 npm run build && rm -f dist/build-meta.json

FROM nginx:alpine AS runner
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=typescript-app /build/dist/ /usr/share/nginx/html/
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
