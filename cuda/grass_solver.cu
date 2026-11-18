#include "solver_common.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <iomanip>

struct alignas(16) PackedBlock {
  int32_t x, y, z;
  uint32_t meta; // bits 0..1: expected rotation for this north, bits 8..9: mask (1 for side, 3 for top)
};

__constant__ PackedBlock cached_blocks[4 * 512];

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

__device__ __forceinline__ void rotation_xz_u32(int32_t x, int32_t z, uint32_t& xz_lo, uint32_t& xz_hi) {
  uint32_t xp = uint32_t(x) * 3129871u;
  uint32_t sx_hi = uint32_t(int32_t(xp) >> 31);
  uint32_t zu = uint32_t(z);
  uint32_t zp_lo = zu * 116129781u;
  uint32_t zp_hi = __umulhi(zu, 116129781u) + uint32_t(z >> 31) * 116129781u;
  xz_lo = xp ^ zp_lo;
  xz_hi = sx_hi ^ zp_hi;
}

__device__ __forceinline__ int rotation_y_lh(uint32_t xz_lo, uint32_t K1, uint32_t K0, uint32_t y_lo) {
  uint32_t lx = xz_lo ^ y_lo;
  uint32_t s_lo = lx * lx;
  uint32_t s_hi = __umulhi(lx, lx);
  uint32_t t_hi = K1 * lx + K0;
  t_hi = s_hi * 42317861u + t_hi;
  uint32_t t_lo = lx * 11u;
  t_hi = __umulhi(lx, 11u) + t_hi;

  uint32_t p_lo, p_hi;
  asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
      "madc.hi.u32   %1, %2, %3, %5;"
      : "=r"(p_lo), "=r"(p_hi)
      : "r"(s_lo), "r"(42317861u), "r"(t_lo), "r"(t_hi));

  uint32_t seed_lo = __funnelshift_r(p_lo, p_hi, 16u) ^ 0xDEECE66Du;
  uint32_t seed_hi = (p_hi >> 16u) ^ 5u;
  uint32_t u_hi = seed_lo * 5u;
  u_hi = seed_hi * 0xDEECE66Du + u_hi;

  uint32_t step_hi;
  asm("{ .reg .u32 tmp;\n\t"
      "  mad.lo.cc.u32 tmp, %1, %2, %3;\n\t"
      "  madc.hi.u32   %0,  %1, %2, %4; }"
      : "=r"(step_hi)
      : "r"(seed_lo), "r"(0xDEECE66Du), "r"(11u), "r"(u_hi));

  return int((step_hi >> 14u) & 3u);
}

__device__ __forceinline__ int rotation_y_u32(uint32_t xz_lo, uint32_t xz_hi, int32_t y) {
  uint32_t ly = xz_hi ^ uint32_t(y >> 31);
  return rotation_y_lh(xz_lo, 84635722u * ly, 11u * ly, uint32_t(y));
}

__device__ __forceinline__ void rotation_y4_lh(uint32_t xz_lo, uint32_t K1, uint32_t K0, uint32_t base_y,
                                               int& r0, int& r1, int& r2, int& r3) {
  r0 = rotation_y_lh(xz_lo, K1, K0, base_y);
  r1 = rotation_y_lh(xz_lo, K1, K0, base_y + 1u);
  r2 = rotation_y_lh(xz_lo, K1, K0, base_y + 2u);
  r3 = rotation_y_lh(xz_lo, K1, K0, base_y + 3u);
}

__device__ __forceinline__ void rotation_y2_lh(uint32_t xz_lo, uint32_t K1, uint32_t K0, uint32_t base_y,
                                               int& r0, int& r1) {
  r0 = rotation_y_lh(xz_lo, K1, K0, base_y);
  r1 = rotation_y_lh(xz_lo, K1, K0, base_y + 1u);
}

__device__ __forceinline__ int texture(int32_t bx, int32_t by, int32_t bz, int side) {
  uint32_t xz_lo, xz_hi;
  rotation_xz_u32(bx, bz, xz_lo, xz_hi);
  int r = rotation_y_u32(xz_lo, xz_hi, by);
  return r & (side ? 1 : 3);
}

__device__ __forceinline__ uint32_t add_i32_wrap(int32_t a, int32_t b) {
  return uint32_t(a) + uint32_t(b);
}

__device__ __forceinline__ PackedBlock load_packed_global(const PackedBlock* __restrict__ blocks, int north, uint32_t i, uint32_t nb) {
  if (nb <= 512u) return cached_blocks[north * 512 + i];
  int4 v = __ldg(reinterpret_cast<const int4*>(blocks + uint32_t(north) * nb + i));
  return PackedBlock{v.x, v.y, v.z, uint32_t(v.w)};
}

__device__ __forceinline__ uint64_t rotation_xz_u64(int32_t x, int32_t z) {
  uint32_t xp = uint32_t(x) * 3129871u;
  uint64_t sx = uint64_t(int64_t(int32_t(xp)));
  return sx ^ (uint64_t(int64_t(z)) * 116129781ULL);
}

__device__ __forceinline__ int rotation_y_u64(uint64_t xz, int32_t y) {
  uint64_t l = xz ^ uint64_t(int64_t(y));
  l = l * l * 42317861ULL + l * 11ULL;
  uint64_t seed = (l >> 16) ^ 0x5DEECE66DULL;
  return int(((seed * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
}

__device__ __forceinline__ void rotation_y4_u64(uint64_t xz, int32_t base_y,
                                                int& r0, int& r1, int& r2, int& r3) {
  uint64_t l0 = xz ^ uint64_t(int64_t(base_y));
  uint64_t l1 = xz ^ uint64_t(int64_t(base_y + 1));
  uint64_t l2 = xz ^ uint64_t(int64_t(base_y + 2));
  uint64_t l3 = xz ^ uint64_t(int64_t(base_y + 3));
  l0 = l0 * l0 * 42317861ULL + l0 * 11ULL;
  l1 = l1 * l1 * 42317861ULL + l1 * 11ULL;
  l2 = l2 * l2 * 42317861ULL + l2 * 11ULL;
  l3 = l3 * l3 * 42317861ULL + l3 * 11ULL;
  uint64_t s0 = (l0 >> 16) ^ 0x5DEECE66DULL;
  uint64_t s1 = (l1 >> 16) ^ 0x5DEECE66DULL;
  uint64_t s2 = (l2 >> 16) ^ 0x5DEECE66DULL;
  uint64_t s3 = (l3 >> 16) ^ 0x5DEECE66DULL;
  r0 = int(((s0 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
  r1 = int(((s1 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
  r2 = int(((s2 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
  r3 = int(((s3 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
}

__device__ __forceinline__ void rotation_y2_u64(uint64_t xz, int32_t base_y, int& r0, int& r1) {
  uint64_t l0 = xz ^ uint64_t(int64_t(base_y));
  uint64_t l1 = xz ^ uint64_t(int64_t(base_y + 1));
  l0 = l0 * l0 * 42317861ULL + l0 * 11ULL;
  l1 = l1 * l1 * 42317861ULL + l1 * 11ULL;
  uint64_t s0 = (l0 >> 16) ^ 0x5DEECE66DULL;
  uint64_t s1 = (l1 >> 16) ^ 0x5DEECE66DULL;
  r0 = int(((s0 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
  r1 = int(((s1 * 0x5DEECE66DULL + 11ULL) >> 46) & 3ULL);
}

__device__ __forceinline__ bool check_multi_const(const PackedBlock* __restrict__ blocks,
                                                  uint32_t nb, int north,
                                                  int32_t x, int32_t y, int32_t z, uint64_t xz1) {
  if (nb > 1u) {
    PackedBlock b1 = load_packed_global(blocks, north, 1u, nb);
    uint32_t r1 = uint32_t(rotation_y_u64(xz1, int32_t(add_i32_wrap(y, b1.y))));
    if ((r1 & (b1.meta >> 8u)) != (b1.meta & 3u)) return false;
    #pragma unroll
    for (uint32_t i = 2u; i < nb; i++) {
      PackedBlock b = load_packed_global(blocks, north, i, nb);
      uint32_t r = uint32_t(rotation_y_u64(
          rotation_xz_u64(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z))),
          int32_t(add_i32_wrap(y, b.y))));
      if ((r & (b.meta >> 8u)) != (b.meta & 3u)) return false;
    }
  }
  return true;
}

static constexpr uint32_t SHARED_CAP = 64u;

struct B12RegsSingle {
  uint32_t xz1_lo, xz1_hi;
  int32_t b1_y;
  uint32_t b1_meta;
  uint32_t xz2_lo, xz2_hi;
  int32_t b2_y;
  uint32_t b2_meta;
};

// Unused multi-north shared-mem helpers removed in favor of 40-register 100%-occupancy path.

__device__ __forceinline__ bool check_tail_shared(const PackedBlock (*s_blocks)[SHARED_CAP],
                                                  const PackedBlock* __restrict__ blocks,
                                                  uint32_t start_i, uint32_t nb, int north,
                                                  int32_t x, int32_t y, int32_t z) {
  uint32_t s_end = nb < SHARED_CAP ? nb : SHARED_CAP;
  #pragma unroll 4
  for (uint32_t i = start_i; i < s_end; i++) {
    PackedBlock b = s_blocks[north][i];
    uint32_t xz_lo, xz_hi;
    rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xz_lo, xz_hi);
    uint32_t r = uint32_t(rotation_y_u32(xz_lo, xz_hi, int32_t(add_i32_wrap(y, b.y))));
    if ((r & (b.meta >> 8u)) != (b.meta & 3u)) return false;
  }
  for (uint32_t i = SHARED_CAP; i < nb; i++) {
    PackedBlock b = load_packed_global(blocks, north, i, nb);
    uint32_t xz_lo, xz_hi;
    rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xz_lo, xz_hi);
    uint32_t r = uint32_t(rotation_y_u32(xz_lo, xz_hi, int32_t(add_i32_wrap(y, b.y))));
    if ((r & (b.meta >> 8u)) != (b.meta & 3u)) return false;
  }
  return true;
}

__device__ __forceinline__ bool check_single(const PackedBlock (*s_blocks)[SHARED_CAP],
                                             const PackedBlock* __restrict__ blocks,
                                             uint32_t nb, int single_n,
                                             int32_t x, int32_t y, int32_t z,
                                             const B12RegsSingle& regs) {
  if (nb > 1u) {
    uint32_t r1 = uint32_t(rotation_y_u32(regs.xz1_lo, regs.xz1_hi, int32_t(add_i32_wrap(y, regs.b1_y))));
    if ((r1 & (regs.b1_meta >> 8u)) != (regs.b1_meta & 3u)) return false;
    if (nb > 2u) {
      uint32_t r2 = uint32_t(rotation_y_u32(regs.xz2_lo, regs.xz2_hi, int32_t(add_i32_wrap(y, regs.b2_y))));
      if ((r2 & (regs.b2_meta >> 8u)) != (regs.b2_meta & 3u)) return false;
      if (nb > 3u) {
        return check_tail_shared(s_blocks, blocks, 3u, nb, single_n, x, y, z);
      }
    }
  }
  return true;
}

template <bool AnchorSide>
__device__ __forceinline__ void sweep_segment_single(
    uint32_t seg_begin, uint32_t seg_end,
    uint32_t xz0_lo, uint32_t K1, uint32_t K0,
    int32_t x, int32_t ymin, int32_t z,
    uint32_t task_base_idx, int single_n, int target_r0,
    const PackedBlock (*s_blocks)[SHARED_CAP],
    const PackedBlock* __restrict__ blocks, uint32_t nb,
    const B12RegsSingle& regs,
    uint32_t* results, uint32_t* result_count) {
  uint32_t yy = seg_begin;
  for (; yy + 3u < seg_end; yy += 4u) {
    uint32_t y0_u = uint32_t(ymin) + yy;
    int r0, r1, r2, r3;
    rotation_y4_lh(xz0_lo, K1, K0, y0_u, r0, r1, r2, r3);
    bool m0 = AnchorSide ? ((r0 & 1) == target_r0) : (r0 == target_r0);
    bool m1 = AnchorSide ? ((r1 & 1) == target_r0) : (r1 == target_r0);
    bool m2 = AnchorSide ? ((r2 & 1) == target_r0) : (r2 == target_r0);
    bool m3 = AnchorSide ? ((r3 & 1) == target_r0) : (r3 == target_r0);
    if (!m0 && !m1 && !m2 && !m3) continue;

    if (m0 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy);
    if (m1 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u + 1u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy + 1u);
    if (m2 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u + 2u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy + 2u);
    if (m3 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u + 3u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy + 3u);
  }
  for (; yy + 1u < seg_end; yy += 2u) {
    uint32_t y0_u = uint32_t(ymin) + yy;
    int r0, r1;
    rotation_y2_lh(xz0_lo, K1, K0, y0_u, r0, r1);
    bool m0 = AnchorSide ? ((r0 & 1) == target_r0) : (r0 == target_r0);
    bool m1 = AnchorSide ? ((r1 & 1) == target_r0) : (r1 == target_r0);
    if (!m0 && !m1) continue;
    if (m0 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy);
    if (m1 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u + 1u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy + 1u);
  }
  for (; yy < seg_end; yy++) {
    uint32_t y0_u = uint32_t(ymin) + yy;
    int r0 = rotation_y_lh(xz0_lo, K1, K0, y0_u);
    bool m0 = AnchorSide ? ((r0 & 1) == target_r0) : (r0 == target_r0);
    if (m0 && check_single(s_blocks, blocks, nb, single_n, x, int32_t(y0_u), z, regs))
      results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy);
  }
}

template <int Mode, bool AnchorSide, bool NonNegY, uint32_t FixedItems>
__global__ void __launch_bounds__(512, 2) search_fast_kernel(
    const PackedBlock* __restrict__ blocks, uint32_t nb,
    uint32_t* results, uint32_t* result_count,
    uint32_t count, uint32_t runtime_items,
    int32_t ymin, int32_t zmin,
    uint32_t ny, uint32_t nz, uint32_t nz_inv,
    uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase,
    uint32_t north_mask, int32_t rA, int single_n, uint32_t zero_yy) {
  __shared__ PackedBlock s_blocks[Mode == 0 ? 4 : 1][Mode == 0 ? SHARED_CAP : 1];
  if constexpr (Mode == 0) {
    uint32_t s_nb = nb < SHARED_CAP ? nb : SHARED_CAP;
    for (uint32_t idx = threadIdx.x; idx < SHARED_CAP; idx += blockDim.x) {
      if (idx < s_nb) {
        s_blocks[single_n][idx] = load_packed_global(blocks, single_n, idx, nb);
      }
    }
    __syncthreads();
  }

  const uint32_t items = FixedItems ? FixedItems : runtime_items;
  uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * items;
  if (first >= tasks) return;

  uint32_t zid = first + zoff;
  uint32_t q = (nz == 1u) ? zid : __umulhi(zid, nz_inv);
  uint32_t zz = zid - q * nz;
  if (zz >= nz) {
    zz -= nz;
    q++;
  }
  int32_t z = int32_t(uint32_t(zmin) + zz);
  int32_t x = int32_t(uint32_t(xbase) + q);

  const int target_r0 = AnchorSide ? ((rA + single_n) & 1) : ((rA + single_n) & 3);
  constexpr uint32_t bit_pat = AnchorSide ? 5u : 1u;
  constexpr bool AllFour = (Mode == 2);

  #pragma unroll 1
  for (uint32_t j = 0; j < items; j++) {
    uint32_t task = first + j;
    if (task >= tasks) break;

    uint32_t begin = (task == 0u) ? yoff : 0u;
    uint32_t rem = count + yoff - task * ny;
    uint32_t end = rem < ny ? rem : ny;
    uint32_t task_base_idx = task * ny - yoff;

    if constexpr (Mode == 0) {
      uint32_t xz0_lo, xz0_hi;
      rotation_xz_u32(x, z, xz0_lo, xz0_hi);
      uint32_t neg_end = end < zero_yy ? end : zero_yy;
      uint32_t pos_begin = begin > zero_yy ? begin : zero_yy;

      B12RegsSingle sregs{};
      if (nb > 1u) {
        PackedBlock sb1 = s_blocks[single_n][1];
        sregs.b1_y = sb1.y;
        sregs.b1_meta = sb1.meta;
        rotation_xz_u32(int32_t(add_i32_wrap(x, sb1.x)), int32_t(add_i32_wrap(z, sb1.z)), sregs.xz1_lo, sregs.xz1_hi);
      }
      if (nb > 2u) {
        PackedBlock sb2 = s_blocks[single_n][2];
        sregs.b2_y = sb2.y;
        sregs.b2_meta = sb2.meta;
        rotation_xz_u32(int32_t(add_i32_wrap(x, sb2.x)), int32_t(add_i32_wrap(z, sb2.z)), sregs.xz2_lo, sregs.xz2_hi);
      }

      if constexpr (NonNegY) {
        uint32_t ly_pos = xz0_hi;
        sweep_segment_single<AnchorSide>(
            begin, end, xz0_lo, 84635722u * ly_pos, 11u * ly_pos,
            x, ymin, z, task_base_idx, single_n, target_r0,
            s_blocks, blocks, nb, sregs, results, result_count);
      } else {
        if (begin < neg_end) {
          uint32_t ly_neg = ~xz0_hi;
          sweep_segment_single<AnchorSide>(
              begin, neg_end, xz0_lo, 84635722u * ly_neg, 11u * ly_neg,
              x, ymin, z, task_base_idx, single_n, target_r0,
              s_blocks, blocks, nb, sregs, results, result_count);
        }
        if (pos_begin < end) {
          uint32_t ly_pos = xz0_hi;
          sweep_segment_single<AnchorSide>(
              pos_begin, end, xz0_lo, 84635722u * ly_pos, 11u * ly_pos,
              x, ymin, z, task_base_idx, single_n, target_r0,
              s_blocks, blocks, nb, sregs, results, result_count);
        }
      }
    } else {
      uint64_t xz0 = rotation_xz_u64(x, z);
      uint64_t xz1[4] = {};
      if (nb > 1u) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
          if (AllFour || (north_mask & (1u << n))) {
            PackedBlock b1 = load_packed_global(blocks, n, 1u, nb);
            xz1[n] = rotation_xz_u64(int32_t(add_i32_wrap(x, b1.x)), int32_t(add_i32_wrap(z, b1.z)));
          }
        }
      }

      uint32_t yy = begin;
      for (; yy + 3u < end; yy += 4u) {
        int32_t y0 = int32_t(uint32_t(ymin) + yy);
        int32_t y1 = int32_t(uint32_t(ymin) + yy + 1u);
        int32_t y2 = int32_t(uint32_t(ymin) + yy + 2u);
        int32_t y3 = int32_t(uint32_t(ymin) + yy + 3u);

        int r0, r1, r2, r3;
        rotation_y4_u64(xz0, y0, r0, r1, r2, r3);
        int n0 = AnchorSide ? ((r0 - (rA & 1)) & 1) : ((r0 - rA) & 3);
        int n1 = AnchorSide ? ((r1 - (rA & 1)) & 1) : ((r1 - rA) & 3);
        int n2 = AnchorSide ? ((r2 - (rA & 1)) & 1) : ((r2 - rA) & 3);
        int n3 = AnchorSide ? ((r3 - (rA & 1)) & 1) : ((r3 - rA) & 3);

        if constexpr (!AllFour) {
          bool m0 = (north_mask & (bit_pat << n0)) != 0;
          bool m1 = (north_mask & (bit_pat << n1)) != 0;
          bool m2 = (north_mask & (bit_pat << n2)) != 0;
          bool m3 = (north_mask & (bit_pat << n3)) != 0;
          if (!m0 && !m1 && !m2 && !m3) continue;
        }

        if ((AllFour || (north_mask & (1u << n0))) && check_multi_const(blocks, nb, n0, x, y0, z, xz1[n0]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0) << 30u) | (task_base_idx + yy);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n0 + 2)))) && check_multi_const(blocks, nb, n0 + 2, x, y0, z, xz1[n0 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0 + 2) << 30u) | (task_base_idx + yy);

        if ((AllFour || (north_mask & (1u << n1))) && check_multi_const(blocks, nb, n1, x, y1, z, xz1[n1]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n1) << 30u) | (task_base_idx + yy + 1u);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n1 + 2)))) && check_multi_const(blocks, nb, n1 + 2, x, y1, z, xz1[n1 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n1 + 2) << 30u) | (task_base_idx + yy + 1u);

        if ((AllFour || (north_mask & (1u << n2))) && check_multi_const(blocks, nb, n2, x, y2, z, xz1[n2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n2) << 30u) | (task_base_idx + yy + 2u);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n2 + 2)))) && check_multi_const(blocks, nb, n2 + 2, x, y2, z, xz1[n2 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n2 + 2) << 30u) | (task_base_idx + yy + 2u);

        if ((AllFour || (north_mask & (1u << n3))) && check_multi_const(blocks, nb, n3, x, y3, z, xz1[n3]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n3) << 30u) | (task_base_idx + yy + 3u);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n3 + 2)))) && check_multi_const(blocks, nb, n3 + 2, x, y3, z, xz1[n3 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n3 + 2) << 30u) | (task_base_idx + yy + 3u);
      }
      for (; yy + 1u < end; yy += 2u) {
        int32_t y0 = int32_t(uint32_t(ymin) + yy);
        int32_t y1 = int32_t(uint32_t(ymin) + yy + 1u);
        int r0, r1;
        rotation_y2_u64(xz0, y0, r0, r1);
        int n0 = AnchorSide ? ((r0 - (rA & 1)) & 1) : ((r0 - rA) & 3);
        int n1 = AnchorSide ? ((r1 - (rA & 1)) & 1) : ((r1 - rA) & 3);
        if constexpr (!AllFour) {
          if (((north_mask & (bit_pat << n0)) == 0) && ((north_mask & (bit_pat << n1)) == 0)) continue;
        }
        if ((AllFour || (north_mask & (1u << n0))) && check_multi_const(blocks, nb, n0, x, y0, z, xz1[n0]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0) << 30u) | (task_base_idx + yy);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n0 + 2)))) && check_multi_const(blocks, nb, n0 + 2, x, y0, z, xz1[n0 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0 + 2) << 30u) | (task_base_idx + yy);

        if ((AllFour || (north_mask & (1u << n1))) && check_multi_const(blocks, nb, n1, x, y1, z, xz1[n1]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n1) << 30u) | (task_base_idx + yy + 1u);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n1 + 2)))) && check_multi_const(blocks, nb, n1 + 2, x, y1, z, xz1[n1 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n1 + 2) << 30u) | (task_base_idx + yy + 1u);
      }
      for (; yy < end; yy++) {
        int32_t y0 = int32_t(uint32_t(ymin) + yy);
        int r0 = rotation_y_u64(xz0, y0);
        int n0 = AnchorSide ? ((r0 - (rA & 1)) & 1) : ((r0 - rA) & 3);
        if ((AllFour || (north_mask & (1u << n0))) && check_multi_const(blocks, nb, n0, x, y0, z, xz1[n0]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0) << 30u) | (task_base_idx + yy);
        if (AnchorSide && (AllFour || (north_mask & (1u << (n0 + 2)))) && check_multi_const(blocks, nb, n0 + 2, x, y0, z, xz1[n0 + 2]))
          results[atomicAdd(result_count, 1u)] = (uint32_t(n0 + 2) << 30u) | (task_base_idx + yy);
      }
    }

    zz++;
    z = int32_t(uint32_t(z) + 1u);
    if (zz == nz) {
      zz = 0u;
      z = zmin;
      x = int32_t(uint32_t(x) + 1u);
    }
  }
}

__global__ void search_wide(const PackedBlock* __restrict__ blocks, uint32_t nb,
                            uint32_t* results, uint32_t* result_count,
                            uint64_t base, uint32_t count, uint32_t items,
                            int32_t xmin, int32_t ymin, int32_t zmin,
                            uint64_t ny, uint64_t nz,
                            uint32_t north_mask, int32_t rA, uint32_t anchor_side) {
  uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * items;
  for (uint32_t j = 0; j < items; j++) {
    uint32_t tid = first + j;
    if (tid >= count) break;

    uint64_t id = base + tid;
    int32_t y = int32_t(int64_t(ymin) + int64_t(id % ny)); id /= ny;
    int32_t z = int32_t(int64_t(zmin) + int64_t(id % nz)); id /= nz;
    int32_t x = int32_t(int64_t(xmin) + int64_t(id));

    uint32_t xz0_lo, xz0_hi;
    rotation_xz_u32(x, z, xz0_lo, xz0_hi);
    int r0 = rotation_y_u32(xz0_lo, xz0_hi, y);

    int steps = anchor_side ? 2 : 1;
    int n_base = anchor_side ? ((r0 - (rA & 1)) & 1) : ((r0 - rA) & 3);
    for (int s = 0; s < steps; s++) {
      int n = n_base + s * 2;
      if ((north_mask & (1u << n)) == 0u) continue;
      bool ok = true;
      for (uint32_t i = 1; i < nb; i++) {
        PackedBlock b = load_packed_global(blocks, n, i, nb);
        int r = texture(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(y, b.y)), int32_t(add_i32_wrap(z, b.z)),
                        (b.meta >> 8u) == 1u ? 1 : 0);
        if (uint32_t(r) != (b.meta & 3u)) { ok = false; break; }
      }
      if (ok) results[atomicAdd(result_count, 1u)] = (uint32_t(n) << 30u) | tid;
    }
  }
}

static void dispatch_search_fast(
    uint32_t grid, uint32_t threads, cudaStream_t stream,
    const PackedBlock* db, uint32_t nb, uint32_t* df, uint32_t* dc,
    uint32_t cnt, uint32_t items, int32_t ymin, int32_t zmin,
    uint32_t ny, uint32_t nz, uint32_t nz_inv,
    uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase,
    uint32_t north_mask, int32_t rA, uint32_t anchor_side, int single_n, uint32_t zero_yy) {
  int mode = (single_n >= 0) ? 0 : (north_mask == 0xFu ? 2 : 1);

  #define LAUNCH_KERNEL(M, AS, NNY, FI) \
    search_fast_kernel<M, AS, NNY, FI><<<grid, threads, 0, stream>>>( \
        db, nb, df, dc, cnt, items, ymin, zmin, ny, nz, nz_inv, \
        tasks, yoff, zoff, xbase, north_mask, rA, single_n, zero_yy)

  #define DISPATCH_ITEMS(M, AS, NNY) \
    switch (items) { \
      case 1: LAUNCH_KERNEL(M, AS, NNY, 1u); break; \
      case 2: LAUNCH_KERNEL(M, AS, NNY, 2u); break; \
      case 4: LAUNCH_KERNEL(M, AS, NNY, 4u); break; \
      default: LAUNCH_KERNEL(M, AS, NNY, 0u); break; \
    }

  #define DISPATCH_NNY(M, AS) \
    if (zero_yy == 0u) { DISPATCH_ITEMS(M, AS, true); } \
    else { DISPATCH_ITEMS(M, AS, false); }

  if (!anchor_side) {
    if (mode == 0) { DISPATCH_NNY(0, false); }
    else if (mode == 2) { DISPATCH_NNY(2, false); }
    else { DISPATCH_NNY(1, false); }
  } else {
    if (mode == 0) { DISPATCH_NNY(0, true); }
    else if (mode == 2) { DISPATCH_NNY(2, true); }
    else { DISPATCH_NNY(1, true); }
  }

  #undef DISPATCH_NNY
  #undef DISPATCH_ITEMS
  #undef LAUNCH_KERNEL
}

struct Tune { uint32_t threads = 128, items = 1; };

static Tune autotune(const cudaDeviceProp& prop, const PackedBlock* db, const std::vector<Block>& tuneBlocks,
                     uint32_t* df, uint32_t* dc, cudaStream_t stream,
                     const Options& s_opt, const std::string& mode, bool fast, uint32_t batch,
                     uint32_t north_mask, int32_t rA, uint32_t anchor_side, int single_n,
                     uint32_t nz_inv, uint32_t zero_yy) {
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
                    "/columns-ptx-smem-v7/" + std::to_string(s_opt.dim[1]) + "/" + std::to_string(s_opt.dim[2]);
  key += "/mask=" + std::to_string(north_mask) + "/rA=" + std::to_string(rA);
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

  if (mode == "auto" && s_opt.total < 32768) {
    std::cerr << "autotune skipped tiny search threads " << fallback.threads << " items " << fallback.items << "\n";
    return fallback;
  }

  uint32_t n = uint32_t(std::min<uint64_t>(batch, std::min<uint64_t>(s_opt.total, 4194304)));
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
          uint32_t tasks = uint32_t((n + s_opt.dim[1] - 1) / s_opt.dim[1]);
          uint32_t grid = (tasks + t * it - 1) / (t * it);
          dispatch_search_fast(
              grid, t, stream, db, uint32_t(tuneBlocks.size()), df, dc,
              n, it, s_opt.lo[1], s_opt.lo[2], uint32_t(s_opt.dim[1]), uint32_t(s_opt.dim[2]), nz_inv,
              tasks, 0, 0, s_opt.lo[0], north_mask, rA, anchor_side, single_n, zero_yy);
        } else {
          search_wide<<<(n + t * it - 1) / (t * it), t, 0, stream>>>(
              db, uint32_t(tuneBlocks.size()), df, dc,
              0, n, it, s_opt.lo[0], s_opt.lo[1], s_opt.lo[2], s_opt.dim[1], s_opt.dim[2],
              north_mask, rA, anchor_side);
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
      if (found > 2ull * n) throw std::runtime_error("Autotune count overflow");
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

static void rotate_coords(int32_t x, int32_t z, int n, int32_t& rx, int32_t& rz) {
  int64_t cx = x, cz = z;
  for (int j = 0; j < n; j++) {
    int64_t old = cx;
    cx = -cz;
    cz = old;
  }
  rx = coord(cx);
  rz = coord(cz);
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
    for (int n : o.north) orient(original, n, o);

    size_t anchor_idx = 0;
    while (anchor_idx < original.size() && original[anchor_idx].side != 0) {
      anchor_idx++;
    }
    if (anchor_idx >= original.size()) {
      anchor_idx = 0;
    }

    int32_t xA = original[anchor_idx].x;
    int32_t yA = original[anchor_idx].y;
    int32_t zA = original[anchor_idx].z;
    int rA = original[anchor_idx].r;
    uint32_t anchor_side = uint32_t(original[anchor_idx].side);

    if (anchor_idx != 0) {
      std::swap(original[0], original[anchor_idx]);
    }

    for (auto& b : original) {
      b.x -= xA;
      b.y -= yA;
      b.z -= zA;
    }
    if (original.size() > 1) {
      std::stable_sort(original.begin() + 1, original.end(),
                       [](const Block& a, const Block& b) { return a.side < b.side; });
    }

    int32_t rot_rx[4] = {}, rot_rz[4] = {};
    int64_t min_rx = INT64_MAX, max_rx = INT64_MIN;
    int64_t min_rz = INT64_MAX, max_rz = INT64_MIN;
    uint32_t north_mask = 0;
    for (int n = 0; n < 4; n++) {
      rotate_coords(xA, zA, n, rot_rx[n], rot_rz[n]);
    }
    for (int n : o.north) {
      north_mask |= (1u << n);
      min_rx = std::min<int64_t>(min_rx, rot_rx[n]);
      max_rx = std::max<int64_t>(max_rx, rot_rx[n]);
      min_rz = std::min<int64_t>(min_rz, rot_rz[n]);
      max_rz = std::max<int64_t>(max_rz, rot_rz[n]);
    }
    bool has_anchor_shift = (min_rx != max_rx) || (min_rz != max_rz);
    int single_n = (o.north.size() == 1) ? o.north[0] : -1;

    Options s_opt = o;
    s_opt.lo[0] = coord(int64_t(o.lo[0]) + min_rx);
    s_opt.hi[0] = coord(int64_t(o.hi[0]) + max_rx);
    s_opt.lo[1] = coord(int64_t(o.lo[1]) + yA);
    s_opt.hi[1] = coord(int64_t(o.hi[1]) + yA);
    s_opt.lo[2] = coord(int64_t(o.lo[2]) + min_rz);
    s_opt.hi[2] = coord(int64_t(o.hi[2]) + max_rz);
    s_opt.total = 1;
    for (int i = 0; i < 3; i++) {
      s_opt.dim[i] = uint64_t(int64_t(s_opt.hi[i]) - s_opt.lo[i]) + 1;
      if (s_opt.total > UINT64_MAX / s_opt.dim[i]) throw std::runtime_error("Search volume overflows uint64");
      s_opt.total *= s_opt.dim[i];
    }

    const uint32_t batch = tuning("GRASS_BATCH", 16777216, 256, 16777216);
    bool fast = s_opt.dim[1] <= 4096 && s_opt.dim[2] <= uint64_t(UINT32_MAX - 16777216u) && !std::getenv("GRASS_CUDA_FORCE_WIDE");

    uint32_t nz32 = fast ? uint32_t(s_opt.dim[2]) : 1u;
    uint32_t nz_inv = (nz32 <= 1u) ? 0xFFFFFFFFu : uint32_t(0x100000000ULL / nz32);
    uint32_t ny32 = fast ? uint32_t(s_opt.dim[1]) : 1u;
    uint32_t zero_yy = (s_opt.lo[1] >= 0)
        ? 0u
        : ((int64_t(s_opt.lo[1]) + int64_t(ny32) <= 0) ? ny32 : uint32_t(-int64_t(s_opt.lo[1])));

    size_t slot_blocks = std::max<size_t>(512, original.size());
    PackedBlock* db = nullptr;
    if (original.size() > 512) {
      check(cudaMalloc((void**)&db, 4 * slot_blocks * sizeof(PackedBlock)));
    }

    PackedBlock all_cached[4 * 512] = {};
    std::vector<PackedBlock> host_blocks(original.size() > 512 ? 4 * slot_blocks : 0);
    Options zero_bounds = o;
    for (int i = 0; i < 3; i++) { zero_bounds.lo[i] = 0; zero_bounds.hi[i] = 0; }
    for (int n : o.north) {
      auto oriented = orient(original, n, zero_bounds);
      for (size_t i = 0; i < oriented.size(); i++) {
        uint32_t mask = oriented[i].side ? 1u : 3u;
        PackedBlock pb{oriented[i].x, oriented[i].y, oriented[i].z, uint32_t(oriented[i].r) | (mask << 8u)};
        if (i < 512) all_cached[n * 512 + i] = pb;
        if (db) host_blocks[n * slot_blocks + i] = pb;
      }
    }
    if (db) {
      check(cudaMemcpy(db, host_blocks.data(), host_blocks.size() * sizeof(PackedBlock), cudaMemcpyHostToDevice));
    }
    check(cudaMemcpyToSymbol(cached_blocks, all_cached, sizeof(all_cached)));

    static constexpr int NUM_STREAMS = 3;
    static constexpr size_t PINNED_STAGING_CAP = 65536;
    uint32_t raw_alloc = uint32_t(std::max<uint64_t>(256, std::min<uint64_t>(batch, s_opt.total)));
    uint32_t alloc_batch = (fast && raw_alloc >= ny32) ? ((raw_alloc / ny32) * ny32) : raw_alloc;
    size_t max_results_per_batch = size_t(alloc_batch) * ((anchor_side && o.north.size() > 1) ? 2u : 1u);
    size_t pinned_cap = std::min<size_t>(PINNED_STAGING_CAP, max_results_per_batch);

    cudaStream_t streams[NUM_STREAMS] = {};
    uint32_t* df[NUM_STREAMS] = {};
    uint32_t* dc[NUM_STREAMS] = {};
    uint32_t* host_results[NUM_STREAMS] = {};
    uint32_t* host_count[NUM_STREAMS] = {};
    std::vector<uint32_t> overflow_results;

    for (int s = 0; s < NUM_STREAMS; s++) {
      check(cudaStreamCreate(&streams[s]));
      check(cudaMalloc((void**)&df[s], max_results_per_batch * sizeof(uint32_t)));
      check(cudaMalloc((void**)&dc[s], sizeof(uint32_t)));
      check(cudaMallocHost((void**)&host_results[s], pinned_cap * sizeof(uint32_t)));
      check(cudaMallocHost((void**)&host_count[s], sizeof(uint32_t)));
    }

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    Tune tune = autotune(prop, db, original, df[0], dc[0], streams[0], s_opt, mode, fast, alloc_batch,
                         north_mask, rA, anchor_side, single_n, nz_inv, zero_yy);
    uint32_t threads = tune.threads;
    uint32_t items = tune.items;

    std::cerr << "CUDA " << prop.name << " sm_" << prop.major << prop.minor
              << " threads " << threads << " items " << items << " batch " << batch << "\n";

    uint64_t done = 0, matches = 0;
    auto start = std::chrono::steady_clock::now();
    auto last = start;
    std::signal(SIGINT, stop_search);
    std::signal(SIGTERM, stop_search);

    struct BatchInfo {
      uint64_t base = 0;
      uint32_t count = 0;
      bool active = false;
    } slot_info[NUM_STREAMS];

    auto launch_slot = [&](int s, uint64_t b, uint32_t cnt) {
      slot_info[s].base = b;
      slot_info[s].count = cnt;
      slot_info[s].active = true;
      check(cudaMemsetAsync(dc[s], 0, sizeof(uint32_t), streams[s]));
      if (fast) {
        uint64_t origin = b;
        uint32_t yoff = uint32_t(origin % s_opt.dim[1]); origin /= s_opt.dim[1];
        uint32_t zoff = uint32_t(origin % s_opt.dim[2]); origin /= s_opt.dim[2];
        int32_t xbase = int32_t(int64_t(s_opt.lo[0]) + int64_t(origin));
        uint32_t tasks = uint32_t((cnt + yoff + s_opt.dim[1] - 1) / s_opt.dim[1]);
        uint32_t grid = (tasks + threads * items - 1) / (threads * items);
        dispatch_search_fast(
            grid, threads, streams[s], db, uint32_t(original.size()), df[s], dc[s],
            cnt, items, s_opt.lo[1], s_opt.lo[2], ny32, nz32, nz_inv,
            tasks, yoff, zoff, xbase, north_mask, rA, anchor_side, single_n, zero_yy);
      } else {
        uint32_t grid = (cnt + threads * items - 1) / (threads * items);
        search_wide<<<grid, threads, 0, streams[s]>>>(
            db, uint32_t(original.size()), df[s], dc[s],
            b, cnt, items, s_opt.lo[0], s_opt.lo[1], s_opt.lo[2], s_opt.dim[1], s_opt.dim[2],
            north_mask, rA, anchor_side);
      }
      check(cudaGetLastError());
      check(cudaMemcpyAsync(host_count[s], dc[s], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[s]));
    };

    auto harvest_slot = [&](int s) {
      if (!slot_info[s].active) return;
      check(cudaStreamSynchronize(streams[s]));
      slot_info[s].active = false;
      uint32_t found = *host_count[s];
      uint64_t cap = uint64_t(slot_info[s].count) * ((anchor_side && o.north.size() > 1) ? 2u : 1u);
      if (found > cap) throw std::runtime_error("Invalid GPU result count");
      if (found > 0) {
        uint32_t* res_ptr = host_results[s];
        if (found <= pinned_cap) {
          check(cudaMemcpyAsync(res_ptr, df[s], size_t(found) * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[s]));
          check(cudaStreamSynchronize(streams[s]));
        } else {
          overflow_results.resize(found);
          res_ptr = overflow_results.data();
          check(cudaMemcpy(res_ptr, df[s], size_t(found) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        std::sort(res_ptr, res_ptr + found);
        for (uint32_t i = 0; i < found; i++) {
          uint32_t val = res_ptr[i];
          int north = int(val >> 30);
          uint32_t offset = val & 0x3FFFFFFFu;
          int32_t wx, wy, wz;
          decode(slot_info[s].base + offset, s_opt, wx, wy, wz);
          int64_t ox = int64_t(wx) - rot_rx[north];
          int64_t oy = int64_t(wy) - yA;
          int64_t oz = int64_t(wz) - rot_rz[north];
          if (has_anchor_shift && (ox < o.lo[0] || ox > o.hi[0] || oz < o.lo[2] || oz > o.hi[2])) continue;
          std::cout << int32_t(ox) << " " << int32_t(oy) << " " << int32_t(oz) << " " << north << "\n";
          matches++;
        }
        if (!std::cout) throw std::runtime_error("Output write failed");
      }
      done = std::min<uint64_t>(o.total, done + slot_info[s].count);
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - last).count() >= 1) {
        double sec = std::chrono::duration<double>(now - start).count();
        std::cerr << "progress " << done << "/" << o.total
                  << " rate " << double(done * o.north.size()) / sec << " candidates/s matches " << matches << "\n";
        last = now;
      }
    };

    auto any_active = [&]() {
      for (int s = 0; s < NUM_STREAMS; s++) if (slot_info[s].active) return true;
      return false;
    };

    uint64_t base = 0;
    int cur = 0;
    while ((base < s_opt.total || any_active()) && !stopped) {
      if (slot_info[cur].active) {
        harvest_slot(cur);
      }
      if (base < s_opt.total && !stopped) {
        uint32_t count = uint32_t(std::min<uint64_t>(alloc_batch, s_opt.total - base));
        launch_slot(cur, base, count);
        base += count;
      }
      cur = (cur + 1) % NUM_STREAMS;
    }
    for (int s = 0; s < NUM_STREAMS; s++) {
      int idx = (cur + s) % NUM_STREAMS;
      if (slot_info[idx].active) {
        if (!stopped) harvest_slot(idx);
        else {
          cudaStreamSynchronize(streams[idx]);
          slot_info[idx].active = false;
        }
      }
    }
    if (!stopped && done < o.total) done = o.total;

    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    uint64_t effective_done = done * o.north.size();
    std::cout.flush();
    std::cerr << (stopped ? "cancelled" : "complete") << " total " << effective_done << " matches " << matches
              << " seconds " << sec << " rate " << double(effective_done) / std::max(sec, 1e-9) << " candidates/s\n";

    for (int s = 0; s < NUM_STREAMS; s++) {
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
