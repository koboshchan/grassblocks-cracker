#!/bin/sh
# Native OpenCV, VLFeat SIFT and dense SIMD kernel in one bounded-memory module.
set -eu
WEB_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_ROOT=${GRASS_NATIVE_BUILD:-"$WEB_DIR/../../grass-native-build"}
mkdir -p "$BUILD_ROOT"
BUILD_ROOT=$(CDPATH= cd -- "$BUILD_ROOT" && pwd)
OPENCV_VERSION=4.10.0
OPENCV_SHA256=b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9
ARCHIVE="$BUILD_ROOT/opencv-$OPENCV_VERSION.tar.gz"
SOURCE="$BUILD_ROOT/opencv-$OPENCV_VERSION"
BUILD="$BUILD_ROOT/opencv-build"
PREFIX="$BUILD_ROOT/opencv-install"
OBJECTS="$BUILD_ROOT/pipeline-objects"
JOBS=${GRASS_BUILD_JOBS:-2}
MODE=${1:-all}
case "$MODE" in all|opencv|link) ;; *) echo "Usage: $0 [all|opencv|link]" >&2; exit 2;; esac

if [ "$MODE" != link ]; then
  if [ ! -f "$ARCHIVE" ]; then
    curl --fail --location --retry 3 "https://github.com/opencv/opencv/archive/$OPENCV_VERSION/opencv-$OPENCV_VERSION.tar.gz" -o "$ARCHIVE.download"
    mv "$ARCHIVE.download" "$ARCHIVE"
  fi
  printf '%s  %s\n' "$OPENCV_SHA256" "$ARCHIVE" | sha256sum -c -
  if [ ! -d "$SOURCE" ]; then tar -xzf "$ARCHIVE" -C "$BUILD_ROOT"; fi
  # Scalar OpenCV matches the non-SIMD npm OpenCV baseline. Only the dense
  # classifier below uses SIMD, with lane order preserved and FP contraction off.
  emcmake cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_C_FLAGS="-O3 -ffp-contract=off" \
    -DCMAKE_CXX_FLAGS="-O3 -ffp-contract=off -fexceptions" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DBUILD_LIST=core,imgproc,features2d,flann,calib3d,video \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_js=OFF \
    -DBUILD_opencv_dnn=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF -DBUILD_PROTOBUF=OFF -DWITH_PROTOBUF=OFF \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF -DWITH_TBB=OFF -DWITH_PTHREADS_PF=OFF \
    -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_ADE=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_V4L=OFF \
    -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF \
    -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_QUIRC=OFF \
    -DWITH_1394=OFF -DWITH_VTK=OFF -DWITH_OPENGL=OFF -DWITH_OPENVX=OFF \
    -DCV_ENABLE_INTRINSICS=OFF -DCPU_BASELINE= -DCPU_DISPATCH= \
    -DCV_TRACE=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DENABLE_PIC=OFF \
    -DENABLE_FAST_MATH=OFF -DOPENCV_ENABLE_NONFREE=OFF
  cmake --build "$BUILD" --parallel "$JOBS"
  cmake --install "$BUILD"
  emcc --version > "$BUILD_ROOT/toolchain-version.txt"
fi
[ "$MODE" != opencv ] || exit 0

mkdir -p "$OBJECTS" "$WEB_DIR/public/vendor"
# Keep C translation units in C mode, never pass sift.c/detect-wasm.c to em++.
emcc -c -O3 -msimd128 -ffp-contract=off -fno-builtin -Dmemset=grass_dense_memset \
  "$WEB_DIR/detect-wasm.c" -o "$OBJECTS/dense.o"
for unit in sift vlfeat/vl/sift vlfeat/vl/imopv vlfeat/vl/mathop vlfeat/vl/generic vlfeat/vl/host vlfeat/vl/random; do
  name=$(printf '%s' "$unit" | tr / _)
  emcc -c -O3 -ffp-contract=off -DVL_DISABLE_SSE2 -DVL_DISABLE_AVX \
    -DVL_DISABLE_THREADS -I"$WEB_DIR/wasm/vlfeat" \
    "$WEB_DIR/wasm/$unit.c" -o "$OBJECTS/$name.o"
done
for unit in pipeline geometry features; do
  em++ -c -std=c++17 -O3 -msimd128 -ffp-contract=off -fexceptions \
    -I"$PREFIX/include/opencv4" -I"$WEB_DIR/wasm" \
    "$WEB_DIR/wasm/$unit.cpp" -o "$OBJECTS/$unit.o"
done
# New ABI functions must use EMSCRIPTEN_KEEPALIVE or be named in the override.
EXPORTS=${GRASS_PIPELINE_EXPORTS:-'["_malloc","_free","_extract","_get_out","_set_refs","_classify_batch"]'}
em++ -O3 -msimd128 -ffp-contract=off \
  "$OBJECTS/pipeline.o" "$OBJECTS/geometry.o" "$OBJECTS/features.o" "$OBJECTS/dense.o" \
  "$OBJECTS/sift.o" "$OBJECTS/vlfeat_vl_sift.o" "$OBJECTS/vlfeat_vl_imopv.o" \
  "$OBJECTS/vlfeat_vl_mathop.o" "$OBJECTS/vlfeat_vl_generic.o" \
  "$OBJECTS/vlfeat_vl_host.o" "$OBJECTS/vlfeat_vl_random.o" \
  -Wl,--start-group "$PREFIX"/lib/libopencv_*.a \
  "$PREFIX/lib/opencv4/3rdparty/libzlib.a" -Wl,--end-group \
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=33554432 \
  -s MAXIMUM_MEMORY=536870912 -s ABORTING_MALLOC=0 \
  -s DISABLE_EXCEPTION_CATCHING=0 -s FILESYSTEM=0 \
  -s MODULARIZE=1 -s EXPORT_NAME=GrassPipeline \
  -s ENVIRONMENT=web,worker,node -s EXPORTED_FUNCTIONS="$EXPORTS" \
  -s EXPORTED_RUNTIME_METHODS='["UTF8ToString"]' \
  -o "$WEB_DIR/public/vendor/grass-pipeline.js"
