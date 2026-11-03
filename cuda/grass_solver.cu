#include "solver_common.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <iomanip>

__constant__ Block cached_blocks[1024];

static uint32_t tuning(const char* name, uint32_t def, uint32_t lo, uint32_t hi) {
  const char* s = std::getenv(name);
  if (!s) return def;
  auto v = number(s);
  if (v < lo || v > hi) throw std::runtime_error(std::string(name) + " outside supported range");
  return uint32_t(v);
}

static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

__device__ __forceinline__ uint64_t rotation_xz(int32_t x, int32_t z) {
  // Java int multiplication wraps before sign extension to long.
  uint32_t xp = uint32_t(x) * uint32_t(3129871);
  uint64_t sx = uint64_t(int64_t(int32_t(xp)));
  return sx ^ (uint64_t(int64_t(z)) * uint64_t(116129781));
}

__device__ __forceinline__ int rotation_y(uint64_t xz, int32_t y) {
  uint64_t l = xz ^ uint64_t(int64_t(y));
  l = l * l * uint64_t(42317861) + l * uint64_t(11);
  // Arithmetic >>16 sign extension cannot affect the low 48 seed bits.
  uint64_t seed = (l >> 16) ^ uint64_t(0x5DEECE66D);
  return int(((seed * uint64_t(0x5DEECE66D) + 11) >> 46) & 3);
}

__device__ __forceinline__ void rotation_y2(uint64_t xz, int32_t base_y, int& r0, int& r1) {
  uint64_t l0 = xz ^ uint64_t(int64_t(base_y));
  uint64_t l1 = xz ^ uint64_t(int64_t(base_y + 1));

  uint64_t sq0 = l0 * l0;
  uint64_t sq1 = l1 * l1;

  l0 = sq0 * uint64_t(42317861) + l0 * uint64_t(11);
  l1 = sq1 * uint64_t(42317861) + l1 * uint64_t(11);

  uint64_t seed0 = (l0 >> 16) ^ uint64_t(0x5DEECE66D);
  uint64_t seed1 = (l1 >> 16) ^ uint64_t(0x5DEECE66D);

  uint64_t step0 = seed0 * uint64_t(0x5DEECE66D) + 11;
  uint64_t step1 = seed1 * uint64_t(0x5DEECE66D) + 11;

  r0 = int((step0 >> 46) & 3);
  r1 = int((step1 >> 46) & 3);
}

__device__ __forceinline__ void rotation_y4(uint64_t xz, int32_t base_y,
                                            int& r0, int& r1, int& r2, int& r3) {
  uint64_t l0 = xz ^ uint64_t(int64_t(base_y));
  uint64_t l1 = xz ^ uint64_t(int64_t(base_y + 1));
  uint64_t l2 = xz ^ uint64_t(int64_t(base_y + 2));
  uint64_t l3 = xz ^ uint64_t(int64_t(base_y + 3));

  uint64_t sq0 = l0 * l0;
  uint64_t sq1 = l1 * l1;
  uint64_t sq2 = l2 * l2;
  uint64_t sq3 = l3 * l3;

  l0 = sq0 * uint64_t(42317861) + l0 * uint64_t(11);
  l1 = sq1 * uint64_t(42317861) + l1 * uint64_t(11);
  l2 = sq2 * uint64_t(42317861) + l2 * uint64_t(11);
  l3 = sq3 * uint64_t(42317861) + l3 * uint64_t(11);

  uint64_t seed0 = (l0 >> 16) ^ uint64_t(0x5DEECE66D);
  uint64_t seed1 = (l1 >> 16) ^ uint64_t(0x5DEECE66D);
  uint64_t seed2 = (l2 >> 16) ^ uint64_t(0x5DEECE66D);
  uint64_t seed3 = (l3 >> 16) ^ uint64_t(0x5DEECE66D);

  uint64_t step0 = seed0 * uint64_t(0x5DEECE66D) + 11;
  uint64_t step1 = seed1 * uint64_t(0x5DEECE66D) + 11;
  uint64_t step2 = seed2 * uint64_t(0x5DEECE66D) + 11;
  uint64_t step3 = seed3 * uint64_t(0x5DEECE66D) + 11;

  r0 = int((step0 >> 46) & 3);
  r1 = int((step1 >> 46) & 3);
  r2 = int((step2 >> 46) & 3);
  r3 = int((step3 >> 46) & 3);
}

__device__ __forceinline__ int texture(int32_t bx, int32_t by, int32_t bz, int side) {
  int r = rotation_y(rotation_xz(bx, bz), by);
  if (side) r %= 2;
  return r;
}

__device__ __forceinline__ Block load_block(const Block* __restrict__ blocks, uint32_t i, uint32_t nb) {
  if (nb <= 1024) {
    return cached_blocks[i];
  }
  Block b;
  b.x = __ldg(&blocks[i].x);
  b.y = __ldg(&blocks[i].y);
  b.z = __ldg(&blocks[i].z);
  b.r = __ldg(&blocks[i].r);
  b.side = __ldg(&blocks[i].side);
  return b;
}

__global__ void __launch_bounds__(512, 2) search_fast(const Block* __restrict__ blocks, uint32_t nb, uint32_t* results, uint32_t* result_count,
                            uint32_t count, uint32_t items,
                            int32_t ymin, int32_t zmin,
                            uint32_t ny, uint32_t nz,
                            uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase) {
  uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * items;
  for (uint32_t j = 0; j < items; j++) {
    uint32_t task = first + j;
    if (task >= tasks) break;

    uint32_t zid = task + zoff;
    uint32_t zz = zid % nz;
    int32_t z = int32_t(int64_t(zmin) + zz);
    int32_t x = int32_t(int64_t(xbase) + zid / nz);

    Block b0 = load_block(blocks, 0, nb);
    uint64_t xz0 = rotation_xz(x + b0.x, z + b0.z);

    Block b1{};
    uint64_t xz1 = 0;
    if (nb > 1) {
      b1 = load_block(blocks, 1, nb);
      xz1 = rotation_xz(x + b1.x, z + b1.z);
    }

    Block b2{};
    uint64_t xz2 = 0;
    if (nb > 2) {
      b2 = load_block(blocks, 2, nb);
      xz2 = rotation_xz(x + b2.x, z + b2.z);
    }

    Block b3{};
    uint64_t xz3 = 0;
    if (nb > 3) {
      b3 = load_block(blocks, 3, nb);
      xz3 = rotation_xz(x + b3.x, z + b3.z);
    }

    const int mask0 = b0.side ? 1 : 3;
    const int mask1 = b1.side ? 1 : 3;
    const int mask2 = b2.side ? 1 : 3;
    const int mask3 = b3.side ? 1 : 3;

    uint32_t begin = (task == 0) ? yoff : 0u;
    uint32_t rem = count + yoff - task * ny;
    uint32_t end = rem < ny ? rem : ny;

    uint32_t yy = begin;
    for (; yy + 3 < end; yy += 4) {
      int32_t y0 = int32_t(int64_t(ymin) + yy);
      int32_t y1 = int32_t(int64_t(ymin) + (yy + 1));
      int32_t y2 = int32_t(int64_t(ymin) + (yy + 2));
      int32_t y3 = int32_t(int64_t(ymin) + (yy + 3));

      int r0_0, r0_1, r0_2, r0_3;
      rotation_y4(xz0, y0 + b0.y, r0_0, r0_1, r0_2, r0_3);
      bool m0 = ((r0_0 & mask0) == b0.r);
      bool m1 = ((r0_1 & mask0) == b0.r);
      bool m2 = ((r0_2 & mask0) == b0.r);
      bool m3 = ((r0_3 & mask0) == b0.r);
      if (!m0 && !m1 && !m2 && !m3) continue;

      if (nb > 1) {
        int r1_0, r1_1, r1_2, r1_3;
        rotation_y4(xz1, y0 + b1.y, r1_0, r1_1, r1_2, r1_3);
        m0 = m0 && ((r1_0 & mask1) == b1.r);
        m1 = m1 && ((r1_1 & mask1) == b1.r);
        m2 = m2 && ((r1_2 & mask1) == b1.r);
        m3 = m3 && ((r1_3 & mask1) == b1.r);
        if (!m0 && !m1 && !m2 && !m3) continue;
      }

      if (nb > 2) {
        int r2_0, r2_1, r2_2, r2_3;
        rotation_y4(xz2, y0 + b2.y, r2_0, r2_1, r2_2, r2_3);
        m0 = m0 && ((r2_0 & mask2) == b2.r);
        m1 = m1 && ((r2_1 & mask2) == b2.r);
        m2 = m2 && ((r2_2 & mask2) == b2.r);
        m3 = m3 && ((r2_3 & mask2) == b2.r);
        if (!m0 && !m1 && !m2 && !m3) continue;
      }

      if (nb > 3) {
        int r3_0, r3_1, r3_2, r3_3;
        rotation_y4(xz3, y0 + b3.y, r3_0, r3_1, r3_2, r3_3);
        m0 = m0 && ((r3_0 & mask3) == b3.r);
        m1 = m1 && ((r3_1 & mask3) == b3.r);
        m2 = m2 && ((r3_2 & mask3) == b3.r);
        m3 = m3 && ((r3_3 & mask3) == b3.r);
        if (!m0 && !m1 && !m2 && !m3) continue;
      }

      if (m0) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y0 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + yy - yoff;
      }

      if (m1) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y1 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + (yy + 1) - yoff;
      }

      if (m2) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y2 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + (yy + 2) - yoff;
      }

      if (m3) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y3 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + (yy + 3) - yoff;
      }
    }

    for (; yy + 1 < end; yy += 2) {
      int32_t y0 = int32_t(int64_t(ymin) + yy);
      int32_t y1 = int32_t(int64_t(ymin) + (yy + 1));

      int r0_0, r0_1;
      rotation_y2(xz0, y0 + b0.y, r0_0, r0_1);
      bool m0 = ((r0_0 & mask0) == b0.r);
      bool m1 = ((r0_1 & mask0) == b0.r);
      if (!m0 && !m1) continue;

      if (nb > 1) {
        int r1_0, r1_1;
        rotation_y2(xz1, y0 + b1.y, r1_0, r1_1);
        m0 = m0 && ((r1_0 & mask1) == b1.r);
        m1 = m1 && ((r1_1 & mask1) == b1.r);
        if (!m0 && !m1) continue;
      }

      if (nb > 2) {
        int r2_0, r2_1;
        rotation_y2(xz2, y0 + b2.y, r2_0, r2_1);
        m0 = m0 && ((r2_0 & mask2) == b2.r);
        m1 = m1 && ((r2_1 & mask2) == b2.r);
        if (!m0 && !m1) continue;
      }

      if (nb > 3) {
        int r3_0, r3_1;
        rotation_y2(xz3, y0 + b3.y, r3_0, r3_1);
        m0 = m0 && ((r3_0 & mask3) == b3.r);
        m1 = m1 && ((r3_1 & mask3) == b3.r);
        if (!m0 && !m1) continue;
      }

      if (m0) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y0 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + yy - yoff;
      }

      if (m1) {
        uint8_t ok = 1;
        #pragma unroll
        for (uint32_t i = 4; i < nb; i++) {
          Block b = load_block(blocks, i, nb);
          int r = texture(x + b.x, y1 + b.y, z + b.z, b.side);
          if (r != b.r) { ok = 0; break; }
        }
        if (ok) results[atomicAdd(result_count, 1u)] = task * ny + (yy + 1) - yoff;
      }
    }

    for (; yy < end; yy++) {
      int32_t y = int32_t(int64_t(ymin) + yy);
      int r0 = rotation_y(xz0, y + b0.y);
      if ((r0 & mask0) != b0.r) continue;

      if (nb > 1) {
        int r1 = rotation_y(xz1, y + b1.y);
        if ((r1 & mask1) != b1.r) continue;
      }

      if (nb > 2) {
        int r2 = rotation_y(xz2, y + b2.y);
        if ((r2 & mask2) != b2.r) continue;
      }

      if (nb > 3) {
        int r3 = rotation_y(xz3, y + b3.y);
        if ((r3 & mask3) != b3.r) continue;
      }

      uint8_t ok = 1;
      #pragma unroll
      for (uint32_t i = 4; i < nb; i++) {
        Block b = load_block(blocks, i, nb);
        int r = texture(x + b.x, y + b.y, z + b.z, b.side);
        if (r != b.r) { ok = 0; break; }
      }
      if (ok) results[atomicAdd(result_count, 1u)] = task * ny + yy - yoff;
    }
  }
}

__global__ void search_wide(const Block* __restrict__ blocks, uint32_t nb, uint32_t* results, uint32_t* result_count,
                            uint64_t base, uint32_t count, uint32_t items,
                            int32_t xmin, int32_t ymin, int32_t zmin,
                            uint64_t ny, uint64_t nz) {
  uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * items;
  for (uint32_t j = 0; j < items; j++) {
    uint32_t tid = first + j;
    if (tid >= count) break;

    uint64_t id = base + tid;
    int32_t y = int32_t(int64_t(ymin) + int64_t(id % ny)); id /= ny;
    int32_t z = int32_t(int64_t(zmin) + int64_t(id % nz)); id /= nz;
    int32_t x = int32_t(int64_t(xmin) + int64_t(id));

    Block b0 = load_block(blocks, 0, nb);
    int r0 = texture(x + b0.x, y + b0.y, z + b0.z, b0.side);
    if (r0 != b0.r) continue;

    uint8_t ok = 1;
    #pragma unroll
    for (uint32_t i = 1; i < nb; i++) {
      Block b = load_block(blocks, i, nb);
      int r = texture(x + b.x, y + b.y, z + b.z, b.side);
      if (r != b.r) { ok = 0; break; }
    }
    if (ok) results[atomicAdd(result_count, 1u)] = tid;
  }
}

struct Tune { uint32_t threads = 128, items = 1; };

static Tune autotune(const cudaDeviceProp& prop, Block* db, const std::vector<Block>& tuneBlocks,
                     uint32_t* df, uint32_t* dc, cudaStream_t stream,
                     const Options& o, const std::string& mode, bool fast, uint32_t batch) {
  Tune fallback{128, 1};
  bool manual = std::getenv("GRASS_THREADS") || std::getenv("GRASS_ITEMS");
  if (manual) {
    uint32_t t = tuning("GRASS_THREADS", 128, 32, 1024);
    uint32_t it = tuning("GRASS_ITEMS", 1, 1, 8);
    if (t % 32) throw std::runtime_error("GRASS_THREADS must be multiple of 32");
    return {t, it};
  }
  if (mode == "off") {
    std::cerr << "autotune off threads " << fallback.threads << " items " << fallback.items << "\n";
    return fallback;
  }

  std::string key = std::string(prop.name) + "/" + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                    "/columns-v1/" + std::to_string(o.dim[1]) + "/" + std::to_string(o.dim[2]);
  for (auto b : tuneBlocks)
    key += "/" + std::to_string(b.x) + "," + std::to_string(b.y) + "," + std::to_string(b.z) + "," +
           std::to_string(b.r) + "," + std::to_string(b.side);
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : key) { hash ^= c; hash *= 1099511628211ull; }

  const char* root = std::getenv("LOCALAPPDATA");
  if (!root) root = std::getenv("HOME");
  std::filesystem::path cache;
  if (root) {
    cache = std::filesystem::path(root) / ".grass-solvers" / ("cuda-" + std::to_string(hash) + ".txt");
  }

  if (mode == "auto" && !cache.empty()) {
    std::ifstream f(cache);
    std::string saved;
    uint32_t t = 0, it = 0;
    if (std::getline(f, saved) && saved == key && (f >> t >> it) &&
        (t == 64 || t == 128 || t == 256 || t == 512) && (it == 1 || it == 2 || it == 4)) {
      std::cerr << "autotune cache hit " << t << " threads " << it << " items\n";
      return {t, it};
    }
  }

  if (mode == "auto" && o.total < 32768) {
    std::cerr << "autotune skipped tiny search threads " << fallback.threads << " items " << fallback.items << "\n";
    return fallback;
  }

  uint32_t n = uint32_t(std::min<uint64_t>(batch, std::min<uint64_t>(o.total, 4194304)));
  std::vector<uint32_t> reference;
  float best = std::numeric_limits<float>::max();
  Tune selected = fallback;

  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));

  for (uint32_t t : {64u, 128u, 256u, 512u}) {
    for (uint32_t it : {1u, 2u, 4u}) {
      auto launch = [&]() {
        check(cudaMemsetAsync(dc, 0, sizeof(uint32_t), stream));
        if (fast) {
          uint32_t tasks = uint32_t((n + o.dim[1] - 1) / o.dim[1]);
          search_fast<<<(tasks + t * it - 1) / (t * it), t, 0, stream>>>(
              db, uint32_t(tuneBlocks.size()), df, dc,
              n, it, o.lo[1], o.lo[2], uint32_t(o.dim[1]), uint32_t(o.dim[2]),
              tasks, 0, 0, o.lo[0]);
        } else {
          search_wide<<<(n + t * it - 1) / (t * it), t, 0, stream>>>(
              db, uint32_t(tuneBlocks.size()), df, dc,
              0, n, it, o.lo[0], o.lo[1], o.lo[2], o.dim[1], o.dim[2]);
        }
        check(cudaGetLastError());
      };
      launch();
      check(cudaStreamSynchronize(stream));
      check(cudaEventRecord(begin, stream));
      for (int rep = 0; rep < 3; rep++) launch();
      check(cudaEventRecord(end, stream));
      check(cudaEventSynchronize(end));
      float ms = 0;
      check(cudaEventElapsedTime(&ms, begin, end));
      ms /= 3;

      uint32_t found = 0;
      check(cudaMemcpy(&found, dc, sizeof(found), cudaMemcpyDeviceToHost));
      if (found > n) throw std::runtime_error("Autotune count overflow");
      std::vector<uint32_t> ids(found);
      if (found) check(cudaMemcpy(ids.data(), df, size_t(found) * 4, cudaMemcpyDeviceToHost));
      std::sort(ids.begin(), ids.end());
      if (t == 64 && it == 1) reference = ids;
      else if (ids != reference) throw std::runtime_error("Autotune correctness mismatch");

      std::cerr << "autotune trial threads " << t << " items " << it
                << " kernel_ms " << ms << " origins/s " << double(n) / (ms * .001) << "\n";
      if (ms < best) {
        best = ms;
        selected = {t, it};
      }
    }
  }
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  std::cerr << "autotune selected " << selected.threads << " threads " << selected.items
            << " items; exact outputs agree\n";

  if (!cache.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(cache.parent_path(), ec);
    if (!ec) {
      std::ofstream f(cache);
      if (f) f << key << "\n" << selected.threads << " " << selected.items << "\n";
    }
  }
  return selected;
}

int main(int argc, char** argv) {
  try {
    std::string mode = "auto";
    std::vector<char*> args{argv[0]};
    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "--autotune") {
        if (++i == argc) throw std::runtime_error("Missing autotune mode");
        mode = argv[i];
        if (mode != "auto" && mode != "force" && mode != "off")
          throw std::runtime_error("autotune expects auto|force|off");
      } else {
        args.push_back(argv[i]);
      }
    }

    auto o = options(int(args.size()), args.data());
    auto original = load(o.path);
    // Validate every north before creating a device context or printing partial results.
    for (int n : o.north) orient(original, n, o);

    const uint32_t batch = tuning("GRASS_BATCH", 4194304, 256, 16777216);
    bool fast = o.dim[1] <= 4096 && o.dim[2] <= uint64_t(UINT32_MAX - 4194304u) && !std::getenv("GRASS_CUDA_FORCE_WIDE");

    Block *db = nullptr;
    check(cudaMalloc((void**)&db, original.size() * sizeof(Block)));

    cudaStream_t streams[2] = {nullptr, nullptr};
    uint32_t *df[2] = {nullptr, nullptr};
    uint32_t *dc[2] = {nullptr, nullptr};
    uint32_t *host_results[2] = {nullptr, nullptr};
    uint32_t *host_count[2] = {nullptr, nullptr};

    for (int s = 0; s < 2; s++) {
      check(cudaStreamCreate(&streams[s]));
      check(cudaMalloc((void**)&df[s], size_t(batch) * sizeof(uint32_t)));
      check(cudaMalloc((void**)&dc[s], sizeof(uint32_t)));
      check(cudaMallocHost((void**)&host_results[s], size_t(batch) * sizeof(uint32_t)));
      check(cudaMallocHost((void**)&host_count[s], sizeof(uint32_t)));
    }

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    auto tuneBlocks = orient(original, o.north[0], o);
    std::stable_sort(tuneBlocks.begin(), tuneBlocks.end(), [](const Block& a, const Block& b) { return a.side < b.side; });
    check(cudaMemcpy(db, tuneBlocks.data(), tuneBlocks.size() * sizeof(Block), cudaMemcpyHostToDevice));
    if (tuneBlocks.size() <= 1024) check(cudaMemcpyToSymbol(cached_blocks, tuneBlocks.data(), tuneBlocks.size() * sizeof(Block)));

    Tune tune = autotune(prop, db, tuneBlocks, df[0], dc[0], streams[0], o, mode, fast, batch);
    uint32_t threads = tune.threads;
    uint32_t items = tune.items;

    std::cerr << "CUDA " << prop.name << " sm_" << prop.major << prop.minor
              << " threads " << threads << " items " << items << " batch " << batch << "\n";

    uint64_t done = 0, matches = 0;
    auto start = std::chrono::steady_clock::now();
    auto last = start;
    std::signal(SIGINT, stop_search);
    std::signal(SIGTERM, stop_search);

    for (int north : o.north) {
      auto blocks = orient(original, north, o);
      // Tops reject 75% versus sides 50%; stable ordering preserves equal-information rows.
      std::stable_sort(blocks.begin(), blocks.end(), [](const Block& a, const Block& b) { return a.side < b.side; });
      if (blocks.size() <= 1024) check(cudaMemcpyToSymbol(cached_blocks, blocks.data(), blocks.size() * sizeof(Block)));
      check(cudaMemcpy(db, blocks.data(), blocks.size() * sizeof(Block), cudaMemcpyHostToDevice));

      struct BatchInfo {
        uint64_t base = 0;
        uint32_t count = 0;
        bool active = false;
      } slot_info[2];

      auto launch_slot = [&](int s, uint64_t b, uint32_t cnt) {
        slot_info[s].base = b;
        slot_info[s].count = cnt;
        slot_info[s].active = true;
        check(cudaMemsetAsync(dc[s], 0, sizeof(uint32_t), streams[s]));
        if (fast) {
          uint64_t origin = b;
          uint32_t yoff = uint32_t(origin % o.dim[1]); origin /= o.dim[1];
          uint32_t zoff = uint32_t(origin % o.dim[2]); origin /= o.dim[2];
          int32_t xbase = int32_t(int64_t(o.lo[0]) + int64_t(origin));
          uint32_t tasks = uint32_t((cnt + yoff + o.dim[1] - 1) / o.dim[1]);
          uint32_t grid = (tasks + threads * items - 1) / (threads * items);
          search_fast<<<grid, threads, 0, streams[s]>>>(
              db, uint32_t(blocks.size()), df[s], dc[s],
              cnt, items, o.lo[1], o.lo[2], uint32_t(o.dim[1]), uint32_t(o.dim[2]),
              tasks, yoff, zoff, xbase);
        } else {
          uint32_t grid = (cnt + threads * items - 1) / (threads * items);
          search_wide<<<grid, threads, 0, streams[s]>>>(
              db, uint32_t(blocks.size()), df[s], dc[s],
              b, cnt, items, o.lo[0], o.lo[1], o.lo[2], o.dim[1], o.dim[2]);
        }
        check(cudaGetLastError());
        check(cudaMemcpyAsync(host_count[s], dc[s], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[s]));
      };

      auto harvest_slot = [&](int s) {
        if (!slot_info[s].active) return;
        check(cudaStreamSynchronize(streams[s]));
        slot_info[s].active = false;
        uint32_t found = *host_count[s];
        if (found > slot_info[s].count) throw std::runtime_error("Invalid GPU result count");
        if (found > 0) {
          check(cudaMemcpy(host_results[s], df[s], size_t(found) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
          std::sort(host_results[s], host_results[s] + found);
          for (uint32_t i = 0; i < found; i++) {
            int32_t x, y, z;
            decode(slot_info[s].base + host_results[s][i], o, x, y, z);
            std::cout << x << " " << y << " " << z << " " << north << "\n";
            matches++;
          }
          if (!std::cout) throw std::runtime_error("Output write failed");
        }
        done += slot_info[s].count;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() >= 1) {
          double sec = std::chrono::duration<double>(now - start).count();
          std::cerr << "progress " << done << "/" << o.total * o.north.size()
                    << " rate " << double(done) / sec << " candidates/s matches " << matches << "\n";
          last = now;
        }
      };

      uint64_t base = 0;
      int cur = 0;
      while ((base < o.total || slot_info[0].active || slot_info[1].active) && !stopped) {
        if (slot_info[cur].active) {
          harvest_slot(cur);
        }
        if (base < o.total && !stopped) {
          uint32_t count = uint32_t(std::min<uint64_t>(batch, o.total - base));
          launch_slot(cur, base, count);
          base += count;
        }
        cur = 1 - cur;
      }
      for (int s = 0; s < 2; s++) {
        if (slot_info[s].active) {
          if (!stopped) harvest_slot(s);
          else {
            cudaStreamSynchronize(streams[s]);
            slot_info[s].active = false;
          }
        }
      }
      if (stopped) break;
    }

    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout.flush();
    std::cerr << (stopped ? "cancelled" : "complete") << " total " << done << " matches " << matches
              << " seconds " << sec << " rate " << double(done) / std::max(sec, 1e-9) << " candidates/s\n";

    for (int s = 0; s < 2; s++) {
      if (df[s]) cudaFree(df[s]);
      if (dc[s]) cudaFree(dc[s]);
      if (host_results[s]) cudaFreeHost(host_results[s]);
      if (host_count[s]) cudaFreeHost(host_count[s]);
      if (streams[s]) cudaStreamDestroy(streams[s]);
    }
    if (db) cudaFree(db);

    return stopped ? 130 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
