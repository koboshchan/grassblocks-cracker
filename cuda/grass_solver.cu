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

// 6-instruction exact XZ hash using signed __mulhi for z * 116129781.
__device__ __forceinline__ void rotation_xz_u32(int32_t x, int32_t z, uint32_t& xz_lo, uint32_t& xz_hi) {
  uint32_t xp = uint32_t(x) * 3129871u;
  uint32_t sx_hi = uint32_t(int32_t(xp) >> 31);
  uint32_t zp_lo = uint32_t(z) * 116129781u;
  uint32_t zp_hi = uint32_t(__mulhi(z, 116129781));
  xz_lo = xp ^ zp_lo;
  xz_hi = sx_hi ^ zp_hi;
}

// Finishes (p_lo, p_hi) -> step_hi with (-expected << 14) folded for free into IMAD.
// Only bits 14..15 of step_hi are read, so seed_hi * 0xE66D is exact mod 2^16.
__device__ __forceinline__ uint32_t finish_p_to_step_hi(
    uint32_t p_lo, uint32_t p_hi, uint32_t neg_shift) {
  uint32_t seed_lo = __funnelshift_r(p_lo, p_hi, 16u) ^ 0xDEECE66Du;
  uint32_t seed_hi = (p_hi >> 16u) ^ 5u;
  uint32_t u_hi = seed_hi * 0xE66Du + neg_shift;
  u_hi = seed_lo * 5u + u_hi;

  uint32_t step_hi;
  asm("{ .reg .u32 tmp;\n\t"
      "  mad.lo.cc.u32 tmp, %1, %2, 11;\n\t"
      "  madc.hi.u32   %0,  %1, %2, %3; }"
      : "=r"(step_hi)
      : "r"(seed_lo), "r"(0xDEECE66Du), "r"(u_hi));
  return step_hi;
}

// Factored 5-IMAD Y hash taking pre-XORed lx = xz_lo ^ y_lo:
//   p = lx * (42317861 * lx + 11 + 2^32 * K1) + 2^32 * K0  (mod 2^64)
__device__ __forceinline__ uint32_t rotation_lx_lh_raw(
    uint32_t lx, uint32_t K1, uint32_t K0, uint32_t neg_shift) {
  uint32_t a_lo, a_hi;
  asm("mad.lo.cc.u32 %0, %2, %3, 11;\n\t"
      "madc.hi.u32   %1, %2, %3, %4;"
      : "=r"(a_lo), "=r"(a_hi)
      : "r"(lx), "r"(42317861u), "r"(K1));
  uint32_t t_hi = a_hi * lx + K0;
  uint32_t p_lo = lx * a_lo;
  uint32_t p_hi = __umulhi(lx, a_lo) + t_hi;
  return finish_p_to_step_hi(p_lo, p_hi, neg_shift);
}

__device__ __forceinline__ uint32_t rotation_y_lh_raw(
    uint32_t xz_lo, uint32_t K1, uint32_t K0, uint32_t y_lo, uint32_t neg_shift) {
  return rotation_lx_lh_raw(xz_lo ^ y_lo, K1, K0, neg_shift);
}

__device__ __forceinline__ uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// 1-instruction predicated bit set for FD hit mask accumulation (saves 4 insns per 8 steps vs SEL+IADD3 tree).
__device__ __forceinline__ void or_if_zero_mask(uint32_t& m, uint32_t val, uint32_t mask, uint32_t bit) {
  asm("{ .reg .u32 t;\n\t"
      "  .reg .pred p;\n\t"
      "  and.b32 t, %1, %2;\n\t"
      "  setp.eq.u32 p, t, 0;\n\t"
      "  @p or.b32 %0, %0, %3; }"
      : "+r"(m)
      : "r"(val), "r"(mask), "r"(bit));
}

// 2-instruction fused ((a ^ b) & mask) == 0 -> m |= bit (1 LOP3.LUT P + 1 predicated OR/IADD3).
__device__ __forceinline__ void or_if_zero_xor_mask(
    uint32_t& m, uint32_t a, uint32_t b, uint32_t mask, uint32_t bit) {
  asm("{ .reg .u32 t;\n\t"
      "  .reg .pred p;\n\t"
      "  lop3.b32 t, %1, %2, %3, 0x28;\n\t"
      "  setp.eq.u32 p, t, 0;\n\t"
      "  @p or.b32 %0, %0, %4; }"
      : "+r"(m)
      : "r"(a), "r"(b), "r"(mask), "r"(bit));
}

// 6-instruction exact 4-way dual register select directly from sh0 bits 14..15 (2 LOP3.LUT P + 4 SEL).
__device__ __forceinline__ void select4_sh0(
    uint32_t sh0,
    uint32_t xl0, uint32_t xl1, uint32_t xl2, uint32_t xl3,
    uint32_t xh0, uint32_t xh1, uint32_t xh2, uint32_t xh3,
    uint32_t& xl, uint32_t& xh) {
  asm("{ .reg .u32 t0, t1, a0, a1, b0, b1;\n\t"
      "  .reg .pred p0, p1;\n\t"
      "  and.b32 t0, %2, 0x4000;\n\t"
      "  and.b32 t1, %2, 0x8000;\n\t"
      "  setp.ne.u32 p0, t0, 0;\n\t"
      "  setp.ne.u32 p1, t1, 0;\n\t"
      "  selp.u32 a0, %4, %3, p0;\n\t"
      "  selp.u32 a1, %6, %5, p0;\n\t"
      "  selp.u32 %0, a1, a0, p1;\n\t"
      "  selp.u32 b0, %8, %7, p0;\n\t"
      "  selp.u32 b1, %10, %9, p0;\n\t"
      "  selp.u32 %1, b1, b0, p1; }"
      : "=r"(xl), "=r"(xh)
      : "r"(sh0),
        "r"(xl0), "r"(xl1), "r"(xl2), "r"(xl3),
        "r"(xh0), "r"(xh1), "r"(xh2), "r"(xh3));
}

__device__ __forceinline__ int rotation_y_lh(uint32_t xz_lo, uint32_t K1, uint32_t K0, uint32_t y_lo) {
  return int((rotation_y_lh_raw(xz_lo, K1, K0, y_lo, 0u) >> 14u) & 3u);
}

__device__ __forceinline__ int rotation_y_u32(uint32_t xz_lo, uint32_t xz_hi, int32_t y) {
  uint32_t ly = xz_hi ^ uint32_t(y >> 31);
  return rotation_y_lh(xz_lo, 84635722u * ly, 11u * ly, uint32_t(y));
}

// 2nd-order finite-difference initialization for consecutive lx = B, B+1, ..., B+(2^k - 1).
// For any 2^k-aligned y_base, {xz_lo ^ (y_base + i)} == {B + j} where B = (xz_lo ^ y_base) & ~(2^k-1),
// so B + j never overflows 32 bits and each step advances p(B+j) with 4 IADD3s and 0 IMADs.
__device__ __forceinline__ void fd_init(
    uint32_t B, uint32_t K1, uint32_t K0,
    uint32_t& p_lo, uint32_t& p_hi, uint32_t& d_lo, uint32_t& d_hi) {
  uint32_t a_lo, a_hi;
  asm("mad.lo.cc.u32 %0, %2, %3, 11;\n\t"
      "madc.hi.u32   %1, %2, %3, %4;"
      : "=r"(a_lo), "=r"(a_hi)
      : "r"(B), "r"(42317861u), "r"(K1));
  uint32_t t_hi = a_hi * B + K0;
  p_lo = B * a_lo;
  p_hi = __umulhi(B, a_lo) + t_hi;
  asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
      "madc.hi.u32   %1, %2, %3, %5;"
      : "=r"(d_lo), "=r"(d_hi)
      : "r"(B), "r"(84635722u), "r"(42317872u), "r"(K1));
}

__device__ __forceinline__ void fd_step_p(uint32_t& p_lo, uint32_t& p_hi, uint32_t d_lo, uint32_t d_hi) {
  asm("add.cc.u32 %0, %0, %2;\n\t"
      "addc.u32   %1, %1, %3;"
      : "+r"(p_lo), "+r"(p_hi)
      : "r"(d_lo), "r"(d_hi));
}

__device__ __forceinline__ void fd_step_d(uint32_t& d_lo, uint32_t& d_hi) {
  asm("add.cc.u32 %0, %0, %2;\n\t"
      "addc.u32   %1, %1, 0;"
      : "+r"(d_lo), "+r"(d_hi)
      : "r"(84635722u));
}

// Warp-wide exclusive prefix sum for small per-lane counts; returns total across all 32 lanes.
__device__ __forceinline__ uint32_t warp_excl_scan(uint32_t val, uint32_t lane, uint32_t& excl) {
  uint32_t x = val;
  #pragma unroll
  for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
    uint32_t up = __shfl_up_sync(0xFFFFFFFFu, x, offset);
    if (lane >= offset) x += up;
  }
  excl = x - val;
  return __shfl_sync(0xFFFFFFFFu, x, 31);
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

static constexpr uint32_t SHARED_CAP = 16u;

// Cold tail verifier for Mode == 0 (uniform north -> constant cache broadcast, zero SMEM).
__device__ __noinline__ bool check_tail_const(const PackedBlock* __restrict__ blocks,
                                              uint32_t start_i, uint32_t nb, int north,
                                              int32_t x, int32_t y, int32_t z) {
  #pragma unroll 1
  for (uint32_t i = start_i; i < nb; i++) {
    PackedBlock b = load_packed_global(blocks, north, i, nb);
    uint32_t xz_lo, xz_hi;
    rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xz_lo, xz_hi);
    uint32_t y_b = add_i32_wrap(y, b.y);
    uint32_t ly = xz_hi ^ uint32_t(int32_t(y_b) >> 31);
    uint32_t neg_s = (uint32_t(-(int32_t(b.meta & 3u))) & 3u) << 14u;
    uint32_t m_s = (b.meta >> 8u) << 14u;
    if ((rotation_y_lh_raw(xz_lo, 84635722u * ly, 11u * ly, y_b, neg_s) & m_s) != 0u) return false;
  }
  return true;
}

// Cold tail verifier for Mode == 1, 2 (per-lane north -> shared memory for first 16 blocks).
__device__ __noinline__ bool check_tail_shared(const PackedBlock* s_row,
                                               const PackedBlock* __restrict__ blocks,
                                               uint32_t start_i, uint32_t nb, int north,
                                               int32_t x, int32_t y, int32_t z) {
  uint32_t s_end = nb < SHARED_CAP ? nb : SHARED_CAP;
  #pragma unroll 1
  for (uint32_t i = start_i; i < s_end; i++) {
    PackedBlock b = s_row[i];
    uint32_t xz_lo, xz_hi;
    rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xz_lo, xz_hi);
    uint32_t y_b = add_i32_wrap(y, b.y);
    uint32_t ly = xz_hi ^ uint32_t(int32_t(y_b) >> 31);
    uint32_t neg_s = (uint32_t(-(int32_t(b.meta & 3u))) & 3u) << 14u;
    uint32_t m_s = (b.meta >> 8u) << 14u;
    if ((rotation_y_lh_raw(xz_lo, 84635722u * ly, 11u * ly, y_b, neg_s) & m_s) != 0u) return false;
  }
  for (uint32_t i = SHARED_CAP; i < nb; i++) {
    PackedBlock b = load_packed_global(blocks, north, i, nb);
    uint32_t xz_lo, xz_hi;
    rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xz_lo, xz_hi);
    uint32_t y_b = add_i32_wrap(y, b.y);
    uint32_t ly = xz_hi ^ uint32_t(int32_t(y_b) >> 31);
    uint32_t neg_s = (uint32_t(-(int32_t(b.meta & 3u))) & 3u) << 14u;
    uint32_t m_s = (b.meta >> 8u) << 14u;
    if ((rotation_y_lh_raw(xz_lo, 84635722u * ly, 11u * ly, y_b, neg_s) & m_s) != 0u) return false;
  }
  return true;
}

template <bool NonNegY>
__device__ __noinline__ void sweep_segment_single_scalar(
    uint32_t seg_begin, uint32_t seg_end, bool anchor_side,
    uint32_t xz0_lo, uint32_t K1, uint32_t K0,
    int32_t x, int32_t ymin, int32_t z,
    uint32_t task_base_idx, int single_n, uint32_t neg_r0_shift,
    const PackedBlock* __restrict__ blocks, uint32_t nb,
    uint32_t xz1_lo, uint32_t xz1_hi, uint32_t k1_1, uint32_t k0_1,
    int32_t b1_y, uint32_t b1_neg_shift, uint32_t b1_mask_shift,
    uint32_t xz2_lo, uint32_t xz2_hi, uint32_t k1_2, uint32_t k0_2,
    int32_t b2_y, uint32_t b2_neg_shift, uint32_t b2_mask_shift,
    uint32_t* results, uint32_t* result_count) {
  uint32_t target_mask = anchor_side ? 0x4000u : 0xC000u;
  for (uint32_t yy = seg_begin; yy < seg_end; yy++) {
    uint32_t y0_u = uint32_t(ymin) + yy;
    if ((rotation_y_lh_raw(xz0_lo, K1, K0, y0_u, neg_r0_shift) & target_mask) != 0u) continue;
    int32_t yk = int32_t(y0_u);
    bool ok = true;
    if (nb > 1u) {
      uint32_t y1 = add_i32_wrap(yk, b1_y);
      uint32_t k1, k0;
      if constexpr (NonNegY) {
        k1 = k1_1; k0 = k0_1;
      } else {
        uint32_t ly1 = xz1_hi ^ uint32_t(int32_t(y1) >> 31);
        k1 = 84635722u * ly1; k0 = 11u * ly1;
      }
      if ((rotation_y_lh_raw(xz1_lo, k1, k0, y1, b1_neg_shift) & b1_mask_shift) != 0u) ok = false;
      else if (nb > 2u) {
        uint32_t y2 = add_i32_wrap(yk, b2_y);
        uint32_t c_k1_2, c_k0_2;
        if constexpr (NonNegY) {
          c_k1_2 = k1_2; c_k0_2 = k0_2;
        } else {
          uint32_t ly2 = xz2_hi ^ uint32_t(int32_t(y2) >> 31);
          c_k1_2 = 84635722u * ly2; c_k0_2 = 11u * ly2;
        }
        if ((rotation_y_lh_raw(xz2_lo, c_k1_2, c_k0_2, y2, b2_neg_shift) & b2_mask_shift) != 0u) ok = false;
        else if (nb > 3u) ok = check_tail_const(blocks, 3u, nb, single_n, x, yk, z);
      }
    }
    if (ok) results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (task_base_idx + yy);
  }
}

template <bool AnchorSide, bool AllFour, bool NonNegY>
__device__ __noinline__ void sweep_segment_multi_scalar(
    uint32_t seg_begin, uint32_t seg_end,
    uint32_t xz0_lo, uint32_t K1, uint32_t K0,
    int32_t x, int32_t ymin, int32_t z,
    uint32_t task_base_idx, uint32_t north_mask, uint32_t neg_rA_shift,
    const PackedBlock* s_blocks_flat,
    const PackedBlock* __restrict__ blocks, uint32_t nb,
    const uint32_t* lane_xz1,
    int32_t b1_y, uint32_t b1_neg_base_shift, uint32_t b1_mask_shift,
    uint32_t* results, uint32_t* result_count) {
  for (uint32_t yy = seg_begin; yy < seg_end; yy++) {
    uint32_t y0_u = uint32_t(ymin) + yy;
    uint32_t sh0 = rotation_y_lh_raw(xz0_lo, K1, K0, y0_u, neg_rA_shift);
    int nk = AnchorSide ? int((sh0 >> 14u) & 1u) : int((sh0 >> 14u) & 3u);
    int32_t yk = int32_t(y0_u);
    for (int step = 0; step < (AnchorSide ? 2 : 1); step++) {
      int n = nk + step * 2;
      if (!AllFour && ((north_mask & (1u << n)) == 0u)) continue;
      bool ok = true;
      if (nb > 1u) {
        uint32_t off = uint32_t(n) << 5u;
        uint32_t xl = lane_xz1[off];
        uint32_t y1 = add_i32_wrap(yk, b1_y);
        uint32_t k1, k0;
        if constexpr (NonNegY) {
          k1 = lane_xz1[off + 128u];
          k0 = lane_xz1[off + 256u];
        } else {
          uint32_t ly1 = lane_xz1[off + 128u] ^ uint32_t(int32_t(y1) >> 31);
          k1 = 84635722u * ly1;
          k0 = 11u * ly1;
        }
        uint32_t sh1 = rotation_y_lh_raw(xl, k1, k0, y1, b1_neg_base_shift);
        if (((sh0 ^ sh1) & b1_mask_shift) != 0u) ok = false;
        else if (nb > 2u) ok = check_tail_shared(s_blocks_flat + uint32_t(n) * SHARED_CAP, blocks, 2u, nb, n, x, yk, z);
      }
      if (ok) results[atomicAdd(result_count, 1u)] = (uint32_t(n) << 30u) | (task_base_idx + yy);
    }
  }
}

static uint32_t kernel_smem_bytes(int mode, uint32_t threads) {
  uint32_t warps = threads >> 5u;
  if (mode == 0) {
    // Mode 0: zero s_blocks; hybrid in-lane/ballot drain needs only s_q1[64]+s_q2[64]+s_q3[64] = 192 u32 = 768 B/warp!
    return warps * 192u * 4u;
  }
  // Mode 1/2: s_blocks[4][16] (1024 B) + per warp: s_xz2[256] + s_q2[160] + s_q3[64] = 480 u32 = 1920 B/warp.
  // At 256 threads (8 warps): 1024 + 15360 = 16384 B -> EXACTLY 4 blocks/SM = 100% occupancy on sm_75!
  return 1024u + warps * 480u * 4u;
}

__device__ __forceinline__ uint32_t select4b_u32(
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, bool b0, bool b1) {
  uint32_t a = b0 ? v1 : v0;
  uint32_t b = b0 ? v3 : v2;
  return b1 ? b : a;
}

__device__ __forceinline__ uint32_t get_lanemask_lt() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

__device__ __forceinline__ void fd_add_const(
    uint32_t q_lo, uint32_t q_hi, uint32_t c, uint32_t& out_lo, uint32_t& out_hi) {
  asm("add.cc.u32 %0, %2, %4;\n\t"
      "addc.u32   %1, %3, 0;"
      : "=r"(out_lo), "=r"(out_hi)
      : "r"(q_lo), "r"(q_hi), "r"(c));
}

template <int Mode, bool AnchorSide, bool NonNegY, bool B1SameY, uint32_t FixedItems>
__global__ void __launch_bounds__(256, 4) search_fast_kernel(
    const PackedBlock* __restrict__ blocks, uint32_t nb,
    uint32_t* results, uint32_t* result_count,
    uint32_t count, uint32_t runtime_items,
    int32_t ymin, int32_t zmin,
    uint32_t ny, uint32_t nz, uint32_t nz_inv,
    uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase,
    uint32_t north_mask, int32_t rA, int single_n, uint32_t zero_yy) {
  extern __shared__ uint32_t dyn_smem[];
  constexpr bool AllFour = (Mode == 2);
  PackedBlock* s_blocks_flat = reinterpret_cast<PackedBlock*>(dyn_smem);
  const uint32_t warp_id = threadIdx.x >> 5u;
  const uint32_t lane = threadIdx.x & 31u;

  if constexpr (Mode != 0) {
    uint32_t s_nb = nb < SHARED_CAP ? nb : SHARED_CAP;
    for (int n = 0; n < 4; n++) {
      if (AllFour || (north_mask & (1u << n))) {
        for (uint32_t idx = threadIdx.x; idx < s_nb; idx += blockDim.x) {
          s_blocks_flat[uint32_t(n) * SHARED_CAP + idx] = load_packed_global(blocks, n, idx, nb);
        }
      }
    }
    __syncthreads();
  }

  const uint32_t items = FixedItems ? FixedItems : runtime_items;
  const uint32_t warp_first = (blockIdx.x * blockDim.x + (threadIdx.x & ~31u)) * items;
  if (warp_first >= tasks) return;

  const uint32_t first = (blockIdx.x * blockDim.x + threadIdx.x) * items;

  uint32_t zid = (first < tasks) ? (first + zoff) : zoff;
  uint32_t q = (nz == 1u) ? zid : __umulhi(zid, nz_inv);
  uint32_t zz = zid - q * nz;
  if (zz >= nz) {
    zz -= nz;
    q++;
  }
  int32_t z = int32_t(uint32_t(zmin) + zz);
  int32_t x = int32_t(uint32_t(xbase) + q);

  const int target_r0 = AnchorSide ? ((rA + single_n) & 1) : ((rA + single_n) & 3);
  const uint32_t neg_r0_shift = (uint32_t(-target_r0) & 3u) << 14u;
  const uint32_t neg_rA_shift = (uint32_t(-rA) & 3u) << 14u;

  uint32_t* warp_mem = (Mode == 0)
      ? (dyn_smem + warp_id * 192u)
      : (dyn_smem + 256u + warp_id * 480u);

  int32_t b1_y = 0, b2_y = 0;
  uint32_t b1_mask_shift = 0u, b1_neg_base_shift = 0u, b1_neg_single = 0u;
  uint32_t b2_mask_shift = 0u, b2_neg_base_shift = 0u, b2_neg_single = 0u;

  if (nb > 1u) {
    if constexpr (Mode == 0) {
      PackedBlock sb1 = load_packed_global(blocks, single_n, 1u, nb);
      b1_y = sb1.y;
      b1_mask_shift = (sb1.meta >> 8u) << 14u;
      b1_neg_single = (uint32_t(-(int32_t(sb1.meta & 3u))) & 3u) << 14u;
      if (nb > 2u) {
        PackedBlock sb2 = load_packed_global(blocks, single_n, 2u, nb);
        b2_y = sb2.y;
        b2_mask_shift = (sb2.meta >> 8u) << 14u;
        b2_neg_single = (uint32_t(-(int32_t(sb2.meta & 3u))) & 3u) << 14u;
      }
    } else {
      int first_n = AllFour ? 0 : (__ffs(int(north_mask)) - 1);
      PackedBlock sb1 = s_blocks_flat[uint32_t(first_n) * SHARED_CAP + 1u];
      b1_y = sb1.y;
      b1_mask_shift = (sb1.meta >> 8u) << 14u;
      uint32_t r1_base = uint32_t(int32_t(sb1.meta & 3u) - first_n) & 3u;
      b1_neg_base_shift = (uint32_t(-int32_t(r1_base)) & 3u) << 14u;
      if (nb > 2u) {
        PackedBlock sb2 = s_blocks_flat[uint32_t(first_n) * SHARED_CAP + 2u];
        b2_y = sb2.y;
        b2_mask_shift = (sb2.meta >> 8u) << 14u;
        uint32_t r2_base = uint32_t(int32_t(sb2.meta & 3u) - first_n) & 3u;
        b2_neg_base_shift = (uint32_t(-int32_t(r2_base)) & 3u) << 14u;
      }
    }
  }

  #pragma unroll 1
  for (uint32_t j = 0; j < items; j++) {
    uint32_t task = first + j;
    bool lane_valid = (task < tasks);
    if (!__any_sync(0xFFFFFFFFu, lane_valid)) break;

    uint32_t begin = 0u, end = 0u, task_base_idx = 0u;
    if (lane_valid) {
      begin = (task == 0u) ? yoff : 0u;
      uint32_t rem = count + yoff - task * ny;
      end = rem < ny ? rem : ny;
      task_base_idx = task * ny - yoff;
    }

    uint32_t xz0_lo = 0u, xz0_hi = 0u;
    if (lane_valid) {
      rotation_xz_u32(x, z, xz0_lo, xz0_hi);
    }

    bool full_warp = (nb >= 4u) && __all_sync(0xFFFFFFFFu, lane_valid && begin == 0u && end == ny);

    if constexpr (Mode == 0) {
      uint32_t* s_q1 = warp_mem;          // 64 u32 (31 + 32 max)
      uint32_t* s_q2 = warp_mem + 64u;    // 64 u32
      uint32_t* s_q3 = warp_mem + 128u;   // 64 u32

      // Keep Block 1 & Block 2 hashes strictly in scalar registers (0-byte stack frame, 0 LDL!).
      uint32_t xz1_lo = 0u, xz1_hi = 0u, k1_1 = 0u, k0_1 = 0u;
      uint32_t xz2_lo = 0u, xz2_hi = 0u, k1_2 = 0u, k0_2 = 0u;

      if (lane_valid && nb > 1u) {
        PackedBlock sb1 = load_packed_global(blocks, single_n, 1u, nb);
        if (sb1.x == 0 && sb1.z == 0) {
          xz1_lo = xz0_lo;
          xz1_hi = xz0_hi;
        } else {
          rotation_xz_u32(int32_t(add_i32_wrap(x, sb1.x)), int32_t(add_i32_wrap(z, sb1.z)), xz1_lo, xz1_hi);
        }
        if constexpr (NonNegY) {
          k1_1 = 84635722u * xz1_hi;
          k0_1 = 11u * xz1_hi;
        }
      }
      if (lane_valid && nb > 2u) {
        PackedBlock sb2 = load_packed_global(blocks, single_n, 2u, nb);
        if (sb2.x == 0 && sb2.z == 0) {
          xz2_lo = xz0_lo;
          xz2_hi = xz0_hi;
        } else {
          rotation_xz_u32(int32_t(add_i32_wrap(x, sb2.x)), int32_t(add_i32_wrap(z, sb2.z)), xz2_lo, xz2_hi);
        }
        if constexpr (NonNegY) {
          k1_2 = 84635722u * xz2_hi;
          k0_2 = 11u * xz2_hi;
        }
      }

      if (full_warp) {
        uint32_t q1_cnt = 0u, q2_cnt = 0u, q3_cnt = 0u;
        constexpr uint32_t target_mask = AnchorSide ? 0x4000u : 0xC000u;

        const uint32_t y1_origin = uint32_t(ymin) + uint32_t(b1_y);

        auto flush_q3 = [&](uint32_t take) {
          __syncwarp();
          q3_cnt -= take;
          bool active = (lane < take);
          uint32_t pkt = active ? s_q3[q3_cnt + lane] : 0u;
          uint32_t src = (pkt >> 12u) & 31u;
          uint32_t pyy = pkt & 0xFFFu;
          int32_t cx = __shfl_sync(0xFFFFFFFFu, x, src);
          int32_t cz = __shfl_sync(0xFFFFFFFFu, z, src);
          uint32_t cbase = __shfl_sync(0xFFFFFFFFu, task_base_idx, src);
          if (active && check_tail_const(blocks, 3u, nb, single_n, cx, int32_t(uint32_t(ymin) + pyy), cz)) {
            results[atomicAdd(result_count, 1u)] = (uint32_t(single_n) << 30u) | (cbase + pyy);
          }
        };

        auto flush_q2 = [&](uint32_t take) {
          __syncwarp();
          q2_cnt -= take;
          bool active = (lane < take);
          uint32_t pkt = active ? s_q2[q2_cnt + lane] : 0u;
          uint32_t src = (pkt >> 12u) & 31u;
          uint32_t pyy = pkt & 0xFFFu;
          uint32_t c_lo = __shfl_sync(0xFFFFFFFFu, xz2_lo, src);
          uint32_t c_k1, c_k0;
          uint32_t y2 = uint32_t(ymin) + pyy + uint32_t(b2_y);
          if constexpr (NonNegY) {
            c_k1 = __shfl_sync(0xFFFFFFFFu, k1_2, src);
            c_k0 = __shfl_sync(0xFFFFFFFFu, k0_2, src);
          } else {
            uint32_t c_hi = __shfl_sync(0xFFFFFFFFu, xz2_hi, src);
            uint32_t ly2 = c_hi ^ uint32_t(int32_t(y2) >> 31);
            c_k1 = 84635722u * ly2;
            c_k0 = 11u * ly2;
          }
          bool pass2 = active && ((rotation_y_lh_raw(c_lo, c_k1, c_k0, y2, b2_neg_single) & b2_mask_shift) == 0u);
          uint32_t m2 = __ballot_sync(0xFFFFFFFFu, pass2);
          if (m2) {
            uint32_t rank = __popc(m2 & get_lanemask_lt());
            if (pass2) s_q3[q3_cnt + rank] = pkt;
            q3_cnt += __popc(m2);
            if (q3_cnt >= 32u) flush_q3(32u);
          }
        };

        auto flush_q1 = [&](uint32_t take) {
          __syncwarp();
          q1_cnt -= take;
          bool active = (lane < take);
          uint32_t pkt = active ? s_q1[q1_cnt + lane] : 0u;
          uint32_t src = (pkt >> 12u) & 31u;
          uint32_t pyy = pkt & 0xFFFu;
          uint32_t c_lo = __shfl_sync(0xFFFFFFFFu, xz1_lo, src);
          uint32_t c_k1, c_k0;
          uint32_t y1 = y1_origin + pyy;
          if constexpr (NonNegY) {
            c_k1 = __shfl_sync(0xFFFFFFFFu, k1_1, src);
            c_k0 = __shfl_sync(0xFFFFFFFFu, k0_1, src);
          } else {
            uint32_t c_hi = __shfl_sync(0xFFFFFFFFu, xz1_hi, src);
            uint32_t ly1 = c_hi ^ uint32_t(int32_t(y1) >> 31);
            c_k1 = 84635722u * ly1;
            c_k0 = 11u * ly1;
          }
          bool pass1 = active && ((rotation_y_lh_raw(c_lo, c_k1, c_k0, y1, b1_neg_single) & b1_mask_shift) == 0u);
          uint32_t m1 = __ballot_sync(0xFFFFFFFFu, pass1);
          if (m1) {
            uint32_t rank = __popc(m1 & get_lanemask_lt());
            if (pass1) s_q2[q2_cnt + rank] = pkt;
            q2_cnt += __popc(m1);
            if (q2_cnt >= 32u) flush_q2(32u);
          }
        };

        auto push_b0_single = [&](bool hit, uint32_t yy_hit) {
          uint32_t bmask = __ballot_sync(0xFFFFFFFFu, hit);
          if (bmask) {
            uint32_t rank = __popc(bmask & get_lanemask_lt());
            if (hit) s_q1[q1_cnt + rank] = yy_hit | (lane << 12u);
            q1_cnt += __popc(bmask);
            if (q1_cnt >= 32u) flush_q1(32u);
          }
        };

        constexpr uint32_t FD_T[8] = {
          0u, 0u, 84635722u, 253907166u, 507814332u, 846357220u, 1269535830u, 1777350162u
        };

        // Single seg loop prevents doubling NonNegY=false code size!
        const int seg_start = NonNegY ? 1 : 0;
        #pragma unroll 1
        for (int seg = seg_start; seg < 2; seg++) {
          uint32_t seg_begin = (seg == 0) ? 0u : (NonNegY ? 0u : zero_yy);
          uint32_t seg_end = (seg == 0) ? (ny < zero_yy ? ny : zero_yy) : ny;
          if (seg_begin >= seg_end) continue;
          uint32_t lh = (seg == 0) ? ~xz0_hi : xz0_hi;
          uint32_t seg_k1_1 = k1_1, seg_k0_1 = k0_1;
          if constexpr (!NonNegY && B1SameY) {
            uint32_t lh1 = (seg == 0) ? ~xz1_hi : xz1_hi;
            seg_k1_1 = 84635722u * lh1;
            seg_k0_1 = 11u * lh1;
          }

          uint32_t yy = seg_begin;
          uint32_t y_first = uint32_t(ymin) + yy;
          uint32_t mis = y_first & 63u;
          bool do_head64 = (mis != 0u) && (mis <= 32u) && (yy + (64u - mis) <= seg_end);
          if (!do_head64) {
            uint32_t K1 = 84635722u * lh, K0 = 11u * lh;
            #pragma unroll 1
            while (yy < seg_end && (((uint32_t(ymin) + yy) & 31u) != 0u)) {
              uint32_t sh = rotation_y_lh_raw(xz0_lo, K1, K0, uint32_t(ymin) + yy, neg_r0_shift);
              push_b0_single((sh & target_mask) == 0u, yy);
              yy++;
            }
          }
          // Unified masked-head + 64/32-step FD (8-step unroll, register-slimmed!) + hybrid drain!
          #pragma unroll 1
          while (do_head64 || (yy + 32u <= seg_end)) {
            uint32_t min_dy = do_head64 ? mis : 0u;
            uint32_t yy_blk = yy - min_dy;
            uint32_t y_aligned = uint32_t(ymin) + yy_blk;
            bool do64 = do_head64 || (((y_aligned & 63u) == 0u) && (yy + 64u <= seg_end));
            uint32_t n_halves = do64 ? 2u : 1u;
            do_head64 = false;

            uint32_t B = (xz0_lo ^ y_aligned) & (do64 ? ~63u : ~31u);
            uint32_t perm = xz0_lo & (do64 ? 63u : 31u);
            uint32_t p_lo, p_hi, d_lo, d_hi;
            fd_init(B, 84635722u * lh, 11u * lh, p_lo, p_hi, d_lo, d_hi);

            #pragma unroll 1
            for (uint32_t half = 0u; half < n_halves; half++) {
              uint32_t hm = 0u;
              #pragma unroll 1
              for (uint32_t c = 0u; c < 4u; c++) {
                uint32_t q_lo = p_lo, q_hi = p_hi;
                uint32_t m8 = 0u;
                #pragma unroll
                for (uint32_t s = 0u; s < 8u; s++) {
                  uint32_t cur_lo, cur_hi;
                  if (s == 0u) {
                    cur_lo = q_lo; cur_hi = q_hi;
                  } else {
                    fd_step_p(q_lo, q_hi, d_lo, d_hi);
                    if (s == 1u) { cur_lo = q_lo; cur_hi = q_hi; }
                    else fd_add_const(q_lo, q_hi, FD_T[s], cur_lo, cur_hi);
                  }
                  uint32_t sh = finish_p_to_step_hi(cur_lo, cur_hi, neg_r0_shift);
                  or_if_zero_mask(m8, sh, target_mask, 1u << s);
                }
                hm = __funnelshift_r(hm, m8, 8u);
                if (c < 3u || half + 1u < n_halves) {
                  fd_step_p(q_lo, q_hi, d_lo, d_hi);
                  fd_add_const(q_lo, q_hi, 2369800216u, p_lo, p_hi);
                  fd_add_const(d_lo, d_hi, 677085776u, d_lo, d_hi);
                }
              }

              uint32_t perm_h = perm ^ (half << 5u);
              if (min_dy > 0u && hm != 0u) {
                uint32_t tmp = hm;
                #pragma unroll 1
                while (tmp) {
                  uint32_t st = uint32_t(__ffs(int(tmp)) - 1);
                  tmp &= tmp - 1u;
                  if ((st ^ perm_h) < min_dy) hm &= ~(1u << st);
                }
              }

              // Hybrid in-lane/ballot drain: direct register verify when >= 22 lanes active, else ballot-compact to s_q1[64].
              uint32_t y1_base = y1_origin + yy_blk;
              #pragma unroll 1
              while (true) {
                bool has = (hm != 0u);
                uint32_t m = __ballot_sync(0xFFFFFFFFu, has);
                if (!m) break;
                uint32_t st = uint32_t(__ffs(int(hm)) - 1);
                if (has) hm &= hm - 1u;
                uint32_t dy = st ^ perm_h;
                uint32_t active_cnt = __popc(m);
                if (active_cnt >= 22u) {
                  bool pass1 = false;
                  if (has) {
                    uint32_t y1 = y1_base + dy;
                    uint32_t k1, k0;
                    if constexpr (NonNegY || B1SameY) {
                      k1 = seg_k1_1; k0 = seg_k0_1;
                    } else {
                      uint32_t ly1 = xz1_hi ^ uint32_t(int32_t(y1) >> 31);
                      k1 = 84635722u * ly1; k0 = 11u * ly1;
                    }
                    pass1 = (rotation_y_lh_raw(xz1_lo, k1, k0, y1, b1_neg_single) & b1_mask_shift) == 0u;
                  }
                  uint32_t m1 = __ballot_sync(0xFFFFFFFFu, pass1);
                  if (m1) {
                    uint32_t rank = __popc(m1 & get_lanemask_lt());
                    if (pass1) s_q2[q2_cnt + rank] = (yy_blk + dy) | (lane << 12u);
                    q2_cnt += __popc(m1);
                    if (q2_cnt >= 32u) flush_q2(32u);
                  }
                } else {
                  uint32_t rank = __popc(m & get_lanemask_lt());
                  if (has) s_q1[q1_cnt + rank] = (yy_blk + dy) | (lane << 12u);
                  q1_cnt += active_cnt;
                  if (q1_cnt >= 32u) flush_q1(32u);
                }
              }
            }
            yy = yy_blk + (n_halves << 5u);
          }
          if (yy < seg_end) {
            uint32_t K1 = 84635722u * lh, K0 = 11u * lh;
            #pragma unroll 1
            for (; yy < seg_end; yy++) {
              uint32_t sh = rotation_y_lh_raw(xz0_lo, K1, K0, uint32_t(ymin) + yy, neg_r0_shift);
              push_b0_single((sh & target_mask) == 0u, yy);
            }
          }
        }

        if (q1_cnt > 0u) flush_q1(q1_cnt);
        if (q2_cnt > 0u) flush_q2(q2_cnt);
        if (q3_cnt > 0u) flush_q3(q3_cnt);
      } else if (lane_valid) {
        uint32_t neg_end = end < zero_yy ? end : zero_yy;
        uint32_t pos_begin = begin > zero_yy ? begin : zero_yy;
        if constexpr (NonNegY) {
          sweep_segment_single_scalar<true>(
              begin, end, AnchorSide, xz0_lo, 84635722u * xz0_hi, 11u * xz0_hi,
              x, ymin, z, task_base_idx, single_n, neg_r0_shift,
              blocks, nb, xz1_lo, xz1_hi, k1_1, k0_1, b1_y, b1_neg_single, b1_mask_shift,
              xz2_lo, xz2_hi, k1_2, k0_2, b2_y, b2_neg_single, b2_mask_shift, results, result_count);
        } else {
          if (begin < neg_end) {
            uint32_t ly_neg = ~xz0_hi;
            sweep_segment_single_scalar<false>(
                begin, neg_end, AnchorSide, xz0_lo, 84635722u * ly_neg, 11u * ly_neg,
                x, ymin, z, task_base_idx, single_n, neg_r0_shift,
                blocks, nb, xz1_lo, xz1_hi, k1_1, k0_1, b1_y, b1_neg_single, b1_mask_shift,
                xz2_lo, xz2_hi, k1_2, k0_2, b2_y, b2_neg_single, b2_mask_shift, results, result_count);
          }
          if (pos_begin < end) {
            sweep_segment_single_scalar<false>(
                pos_begin, end, AnchorSide, xz0_lo, 84635722u * xz0_hi, 11u * xz0_hi,
                x, ymin, z, task_base_idx, single_n, neg_r0_shift,
                blocks, nb, xz1_lo, xz1_hi, k1_1, k0_1, b1_y, b1_neg_single, b1_mask_shift,
                xz2_lo, xz2_hi, k1_2, k0_2, b2_y, b2_neg_single, b2_mask_shift, results, result_count);
          }
        }
      }
    } else {
      // Mode == 1 or Mode == 2 (multi-north).
      // Per-warp slice (480 u32 = 1920 B): s_xz2 [0..255], s_q2[160] [256..415], s_q3[64] [416..479].
      // At 256 threads: 1024 + 8*1920 = 16384 B -> EXACTLY 4 blocks/SM = 100% occupancy on sm_75!
      uint32_t* s_xz2 = warp_mem;
      uint32_t* s_q2 = warp_mem + 256u;
      uint32_t* s_q3 = warp_mem + 416u;

      uint32_t xl1_0 = 0u, xl1_1 = 0u, xl1_2 = 0u, xl1_3 = 0u;
      uint32_t xh1_0 = 0u, xh1_1 = 0u, xh1_2 = 0u, xh1_3 = 0u;

      if (lane_valid && nb > 1u) {
        auto load_b1_n = [&](int n, uint32_t& xl, uint32_t& xh) {
          if (AllFour || (north_mask & (1u << n))) {
            PackedBlock b = s_blocks_flat[uint32_t(n) * SHARED_CAP + 1u];
            if (b.x == 0 && b.z == 0) { xl = xz0_lo; xh = xz0_hi; }
            else rotation_xz_u32(int32_t(add_i32_wrap(x, b.x)), int32_t(add_i32_wrap(z, b.z)), xl, xh);
          }
        };
        load_b1_n(0, xl1_0, xh1_0);
        load_b1_n(1, xl1_1, xh1_1);
        load_b1_n(2, xl1_2, xh1_2);
        load_b1_n(3, xl1_3, xh1_3);
      }

      if (full_warp) {
        // Precompute Block 2 (xl2, xh2) once per column into SMEM -> eliminates 50x rotation_xz_u32 in Block 2!
        #pragma unroll
        for (int n = 0; n < 4; n++) {
          if (AllFour || (north_mask & (1u << n))) {
            PackedBlock b2 = s_blocks_flat[uint32_t(n) * SHARED_CAP + 2u];
            uint32_t xl2, xh2;
            if (b2.x == 0 && b2.z == 0) { xl2 = xz0_lo; xh2 = xz0_hi; }
            else rotation_xz_u32(int32_t(add_i32_wrap(x, b2.x)), int32_t(add_i32_wrap(z, b2.z)), xl2, xh2);
            uint32_t off = (uint32_t(n) << 5u) | lane;
            s_xz2[off] = xl2;
            s_xz2[128u + off] = xh2;
          }
        }
        __syncwarp();

        uint32_t q2_cnt = 0u, q3_cnt = 0u;
        const uint32_t y1_origin = uint32_t(ymin) + uint32_t(b1_y);

        auto flush_q3 = [&](uint32_t take) {
          __syncwarp();
          q3_cnt -= take;
          bool active = (lane < take);
          uint32_t pkt = active ? s_q3[q3_cnt + lane] : 0u;
          uint32_t src = (pkt >> 12u) & 31u;
          uint32_t pyy = pkt & 0xFFFu;
          int pn = int((pkt >> 17u) & 3u);
          int32_t cx = __shfl_sync(0xFFFFFFFFu, x, src);
          int32_t cz = __shfl_sync(0xFFFFFFFFu, z, src);
          uint32_t cbase = __shfl_sync(0xFFFFFFFFu, task_base_idx, src);
          if (active && check_tail_shared(s_blocks_flat + uint32_t(pn) * SHARED_CAP, blocks, 3u, nb, pn,
                                          cx, int32_t(uint32_t(ymin) + pyy), cz)) {
            results[atomicAdd(result_count, 1u)] = (uint32_t(pn) << 30u) | (cbase + pyy);
          }
        };

        auto flush_q2 = [&](uint32_t take) {
          __syncwarp();
          q2_cnt -= take;
          bool active = (lane < take);
          uint32_t pkt = active ? s_q2[q2_cnt + lane] : 0u;
          bool pass2 = false;
          if (active) {
            uint32_t off = pkt >> 12u; // bits 12..16 = src (0..31), bits 17..18 = pn (0..3) -> (pn << 5) | src!
            uint32_t pyy = pkt & 0xFFFu;
            uint32_t xl2 = s_xz2[off];
            uint32_t xh2 = s_xz2[128u + off];
            uint32_t y2 = uint32_t(ymin) + pyy + uint32_t(b2_y);
            uint32_t ly2 = NonNegY ? xh2 : (xh2 ^ uint32_t(int32_t(y2) >> 31));
            uint32_t sh2 = rotation_y_lh_raw(xl2, 84635722u * ly2, 11u * ly2, y2, b2_neg_base_shift);
            pass2 = ((sh2 ^ (pkt >> 3u)) & b2_mask_shift) == 0u;
          }
          uint32_t mask2 = __ballot_sync(0xFFFFFFFFu, pass2);
          if (mask2) {
            uint32_t rank = __popc(mask2 & get_lanemask_lt());
            if (pass2) s_q3[q3_cnt + rank] = pkt;
            q3_cnt += __popc(mask2);
            if (q3_cnt >= 32u) flush_q3(32u);
          }
        };

        auto push_b1_single = [&](bool pass1, uint32_t yy_hit, uint32_t n_u) {
          uint32_t bmask = __ballot_sync(0xFFFFFFFFu, pass1);
          if (bmask) {
            uint32_t rank = __popc(bmask & get_lanemask_lt());
            if (pass1) s_q2[q2_cnt + rank] = yy_hit | (lane << 12u) | (n_u << 17u);
            q2_cnt += __popc(bmask);
            if (q2_cnt >= 32u) flush_q2(32u);
          }
        };

        // Zero-LDS register-only Block 1 test via direct sh0 bit-tested SEL tree!
        auto test_b1_bits = [&](uint32_t sh0, bool b0, bool b1, uint32_t y1) -> bool {
          if constexpr (!AllFour) {
            uint32_t n_u = (b0 ? 1u : 0u) | (b1 ? 2u : 0u);
            if ((north_mask & (1u << n_u)) == 0u) return false;
          }
          uint32_t xl = select4b_u32(xl1_0, xl1_1, xl1_2, xl1_3, b0, b1);
          uint32_t xh = select4b_u32(xh1_0, xh1_1, xh1_2, xh1_3, b0, b1);
          uint32_t ly1 = NonNegY ? xh : (xh ^ uint32_t(int32_t(y1) >> 31));
          uint32_t sh1 = rotation_y_lh_raw(xl, 84635722u * ly1, 11u * ly1, y1, b1_neg_base_shift);
          return ((sh0 ^ sh1) & b1_mask_shift) == 0u;
        };

        // One-shot warp-prefix-sum scatter into compact s_q2[160] (100% occupancy at t=256, overflow-proof 4+4 split!).
        auto commit_mask8_multi = [&](uint32_t hm, uint32_t nk_pack, uint32_t yy_blk, uint32_t step_off, uint32_t perm) {
          uint32_t my_n = __popc(hm);
          uint32_t my_off = 0u;
          uint32_t total = warp_excl_scan(my_n, lane, my_off);
          if (total == 0u) return;
          uint32_t perm_c = perm ^ step_off;
          uint32_t lane_tag = lane << 12u;
          if (q2_cnt + total <= 160u) {
            uint32_t dst = q2_cnt + my_off;
            #pragma unroll 1
            while (hm) {
              uint32_t st = uint32_t(__ffs(int(hm)) - 1);
              hm &= hm - 1u;
              uint32_t nk = (nk_pack >> (st << 1u)) & 3u;
              s_q2[dst++] = (yy_blk + (st ^ perm_c)) | lane_tag | (nk << 17u);
            }
            q2_cnt += total;
            #pragma unroll 1
            while (q2_cnt >= 32u) flush_q2(32u);
          } else {
            #pragma unroll 1
            for (uint32_t h = 0u; h < 2u; h++) {
              uint32_t sub = (hm >> (h * 4u)) & 0xFu;
              uint32_t sn = __popc(sub);
              uint32_t soff = 0u;
              uint32_t stot = warp_excl_scan(sn, lane, soff);
              if (stot) {
                uint32_t dst = q2_cnt + soff;
                #pragma unroll 1
                while (sub) {
                  uint32_t st = uint32_t(__ffs(int(sub)) - 1);
                  sub &= sub - 1u;
                  uint32_t full_st = h * 4u + st;
                  uint32_t nk = (nk_pack >> (full_st << 1u)) & 3u;
                  s_q2[dst++] = (yy_blk + (full_st ^ perm_c)) | lane_tag | (nk << 17u);
                }
                q2_cnt += stot;
                #pragma unroll 1
                while (q2_cnt >= 32u) flush_q2(32u);
              }
            }
          }
        };

        constexpr uint32_t FD_T[8] = {
          0u, 0u, 84635722u, 253907166u, 507814332u, 846357220u, 1269535830u, 1777350162u
        };

        // Single seg loop eliminates code duplication & register spills in NonNegY=false!
        const int seg_start = NonNegY ? 1 : 0;
        #pragma unroll 1
        for (int seg = seg_start; seg < 2; seg++) {
          uint32_t seg_begin = (seg == 0) ? 0u : (NonNegY ? 0u : zero_yy);
          uint32_t seg_end = (seg == 0) ? (ny < zero_yy ? ny : zero_yy) : ny;
          if (seg_begin >= seg_end) continue;
          uint32_t lh = (seg == 0) ? ~xz0_hi : xz0_hi;

          uint32_t yy = seg_begin;
          uint32_t y_first = uint32_t(ymin) + yy;
          uint32_t mis = y_first & 63u;
          bool do_head64 = (mis != 0u) && (mis <= 32u) && (yy + (64u - mis) <= seg_end);
          if (!do_head64) {
            uint32_t K1 = 84635722u * lh, K0 = 11u * lh;
            #pragma unroll 1
            while (yy < seg_end && (((uint32_t(ymin) + yy) & 31u) != 0u)) {
              uint32_t sh0 = rotation_y_lh_raw(xz0_lo, K1, K0, uint32_t(ymin) + yy, neg_rA_shift);
              uint32_t nk = AnchorSide ? ((sh0 >> 14u) & 1u) : ((sh0 >> 14u) & 3u);
              uint32_t y1 = y1_origin + yy;
              push_b1_single(test_b1_bits(sh0, (nk & 1u) != 0u, (nk & 2u) != 0u, y1), yy, nk);
              if constexpr (AnchorSide) push_b1_single(test_b1_bits(sh0, (nk & 1u) != 0u, true, y1), yy, nk + 2u);
              yy++;
            }
          }

          if constexpr (!NonNegY && B1SameY && !AnchorSide) {
            if (seg == 0) {
              xh1_0 = ~xh1_0; xh1_1 = ~xh1_1; xh1_2 = ~xh1_2; xh1_3 = ~xh1_3;
            }
          }

          // Unified masked-head + 64/32-step FD (8-step unroll, register-slimmed!) + 8-step one-shot commit!
          #pragma unroll 1
          while (do_head64 || (yy + 32u <= seg_end)) {
            uint32_t min_dy = do_head64 ? mis : 0u;
            uint32_t yy_blk = yy - min_dy;
            uint32_t y_aligned = uint32_t(ymin) + yy_blk;
            uint32_t y1_base = 0u;
            if constexpr (!B1SameY || AnchorSide) y1_base = y1_origin + yy_blk;
            bool do64 = do_head64 || (((y_aligned & 63u) == 0u) && (yy + 64u <= seg_end));
            uint32_t n_steps = do64 ? 64u : 32u;
            do_head64 = false;

            uint32_t B = (xz0_lo ^ y_aligned) & (do64 ? ~63u : ~31u);
            uint32_t perm = xz0_lo & (do64 ? 63u : 31u);
            uint32_t p_lo, p_hi, d_lo, d_hi;
            fd_init(B, 84635722u * lh, 11u * lh, p_lo, p_hi, d_lo, d_hi);
            if constexpr (!AnchorSide) {
              uint32_t xlp_mask = 0u;
              if constexpr (B1SameY) {
                xlp_mask = y_aligned ^ perm;
                xl1_0 ^= xlp_mask; xl1_1 ^= xlp_mask; xl1_2 ^= xlp_mask; xl1_3 ^= xlp_mask;
              }
              #pragma unroll 1
              for (uint32_t chunk = 0u; chunk < n_steps; chunk += 8u) {
                uint32_t q_lo = p_lo, q_hi = p_hi;
                uint32_t hm = 0u;
                uint32_t nk_pack = 0u;
                #pragma unroll
                for (uint32_t s = 0u; s < 8u; s++) {
                  uint32_t cur_lo, cur_hi;
                  if (s == 0u) {
                    cur_lo = q_lo; cur_hi = q_hi;
                  } else {
                    fd_step_p(q_lo, q_hi, d_lo, d_hi);
                    if (s == 1u) { cur_lo = q_lo; cur_hi = q_hi; }
                    else fd_add_const(q_lo, q_hi, FD_T[s], cur_lo, cur_hi);
                  }
                  uint32_t sh0 = finish_p_to_step_hi(cur_lo, cur_hi, neg_rA_shift);
                  uint32_t xl, xh;
                  select4_sh0(sh0, xl1_0, xl1_1, xl1_2, xl1_3, xh1_0, xh1_1, xh1_2, xh1_3, xl, xh);
                  nk_pack = (s == 0u) ? (sh0 & 0xC000u) : ((nk_pack >> 2u) | (sh0 & 0xC000u));
                  uint32_t sh1;
                  if constexpr (B1SameY) {
                    uint32_t lx1 = (s == 0u) ? (xl ^ chunk) : lop3_xor3(xl, chunk, s);
                    sh1 = rotation_lx_lh_raw(lx1, 84635722u * xh, 11u * xh, b1_neg_base_shift);
                  } else {
                    uint32_t y1 = y1_base + ((chunk + s) ^ perm);
                    uint32_t ly1 = NonNegY ? xh : (xh ^ uint32_t(int32_t(y1) >> 31));
                    sh1 = rotation_y_lh_raw(xl, 84635722u * ly1, 11u * ly1, y1, b1_neg_base_shift);
                  }
                  if constexpr (AllFour) {
                    or_if_zero_xor_mask(hm, nk_pack, sh1, b1_mask_shift, 1u << s);
                  } else {
                    if ((((nk_pack ^ sh1) & b1_mask_shift) == 0u) &&
                        ((north_mask & (1u << ((nk_pack >> 14u) & 3u))) != 0u)) {
                      hm |= (1u << s);
                    }
                  }
                }
                if (chunk + 8u < n_steps) {
                  fd_step_p(q_lo, q_hi, d_lo, d_hi);
                  fd_add_const(q_lo, q_hi, 2369800216u, p_lo, p_hi);
                  fd_add_const(d_lo, d_hi, 677085776u, d_lo, d_hi);
                }
                if (min_dy > 0u && hm != 0u) {
                  uint32_t tmp = hm;
                  #pragma unroll 1
                  while (tmp) {
                    uint32_t st = uint32_t(__ffs(int(tmp)) - 1);
                    tmp &= tmp - 1u;
                    if (((chunk + st) ^ perm) < min_dy) hm &= ~(1u << st);
                  }
                }
                commit_mask8_multi(hm, nk_pack, yy_blk, chunk, perm);
              }
              if constexpr (B1SameY) {
                xl1_0 ^= xlp_mask; xl1_1 ^= xlp_mask; xl1_2 ^= xlp_mask; xl1_3 ^= xlp_mask;
              }
            } else {
              #pragma unroll 1
              for (uint32_t step = 0u; step < n_steps; step++) {
                uint32_t sh0 = finish_p_to_step_hi(p_lo, p_hi, neg_rA_shift);
                fd_step_p(p_lo, p_hi, d_lo, d_hi);
                fd_step_d(d_lo, d_hi);
                uint32_t nk = (sh0 >> 14u) & 1u;
                uint32_t off_y = step ^ perm;
                uint32_t y1 = y1_base + off_y;
                bool ok_dy = (off_y >= min_dy);
                push_b1_single(ok_dy && test_b1_bits(sh0, nk != 0u, false, y1), yy_blk + off_y, nk);
                push_b1_single(ok_dy && test_b1_bits(sh0, nk != 0u, true, y1), yy_blk + off_y, nk + 2u);
              }
            }
            yy = yy_blk + n_steps;
          }

          if constexpr (!NonNegY && B1SameY && !AnchorSide) {
            if (seg == 0) {
              xh1_0 = ~xh1_0; xh1_1 = ~xh1_1; xh1_2 = ~xh1_2; xh1_3 = ~xh1_3;
            }
          }

          if (yy < seg_end) {
            uint32_t K1 = 84635722u * lh, K0 = 11u * lh;
            #pragma unroll 1
            for (; yy < seg_end; yy++) {
              uint32_t sh0 = rotation_y_lh_raw(xz0_lo, K1, K0, uint32_t(ymin) + yy, neg_rA_shift);
              uint32_t nk = AnchorSide ? ((sh0 >> 14u) & 1u) : ((sh0 >> 14u) & 3u);
              uint32_t y1 = y1_origin + yy;
              push_b1_single(test_b1_bits(sh0, (nk & 1u) != 0u, (nk & 2u) != 0u, y1), yy, nk);
              if constexpr (AnchorSide) push_b1_single(test_b1_bits(sh0, (nk & 1u) != 0u, true, y1), yy, nk + 2u);
            }
          }
        }

        if (q2_cnt > 0u) flush_q2(q2_cnt);
        if (q3_cnt > 0u) flush_q3(q3_cnt);
      } else if (lane_valid) {
        uint32_t* lane_xz1 = s_xz2 + lane;
        lane_xz1[0u] = xl1_0; lane_xz1[32u] = xl1_1; lane_xz1[64u] = xl1_2; lane_xz1[96u] = xl1_3;
        lane_xz1[128u] = xh1_0; lane_xz1[160u] = xh1_1; lane_xz1[192u] = xh1_2; lane_xz1[224u] = xh1_3;
        uint32_t neg_end = end < zero_yy ? end : zero_yy;
        uint32_t pos_begin = begin > zero_yy ? begin : zero_yy;
        if constexpr (NonNegY) {
          sweep_segment_multi_scalar<AnchorSide, AllFour, false>(
              begin, end, xz0_lo, 84635722u * xz0_hi, 11u * xz0_hi,
              x, ymin, z, task_base_idx, north_mask, neg_rA_shift,
              s_blocks_flat, blocks, nb, lane_xz1,
              b1_y, b1_neg_base_shift, b1_mask_shift, results, result_count);
        } else {
          if (begin < neg_end) {
            uint32_t ly_neg = ~xz0_hi;
            sweep_segment_multi_scalar<AnchorSide, AllFour, false>(
                begin, neg_end, xz0_lo, 84635722u * ly_neg, 11u * ly_neg,
                x, ymin, z, task_base_idx, north_mask, neg_rA_shift,
                s_blocks_flat, blocks, nb, lane_xz1,
                b1_y, b1_neg_base_shift, b1_mask_shift, results, result_count);
          }
          if (pos_begin < end) {
            sweep_segment_multi_scalar<AnchorSide, AllFour, false>(
                pos_begin, end, xz0_lo, 84635722u * xz0_hi, 11u * xz0_hi,
                x, ymin, z, task_base_idx, north_mask, neg_rA_shift,
                s_blocks_flat, blocks, nb, lane_xz1,
                b1_y, b1_neg_base_shift, b1_mask_shift, results, result_count);
          }
        }
      }
      __syncwarp();
    }

    if (lane_valid) {
      zz++;
      z = int32_t(uint32_t(z) + 1u);
      if (zz == nz) {
        zz = 0u;
        z = zmin;
        x = int32_t(uint32_t(x) + 1u);
      }
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

template <int Mode, bool AnchorSide, bool NonNegY, bool B1SameY, uint32_t FixedItems>
static void launch_fast_typed(
    uint32_t grid, uint32_t threads, uint32_t smem_bytes, cudaStream_t stream,
    const PackedBlock* db, uint32_t nb, uint32_t* df, uint32_t* dc,
    uint32_t cnt, uint32_t items, int32_t ymin, int32_t zmin,
    uint32_t ny, uint32_t nz, uint32_t nz_inv,
    uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase,
    uint32_t north_mask, int32_t rA, int single_n, uint32_t zero_yy) {
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(search_fast_kernel<Mode, AnchorSide, NonNegY, B1SameY, FixedItems>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    configured = true;
  }
  search_fast_kernel<Mode, AnchorSide, NonNegY, B1SameY, FixedItems><<<grid, threads, smem_bytes, stream>>>(
      db, nb, df, dc, cnt, items, ymin, zmin, ny, nz, nz_inv,
      tasks, yoff, zoff, xbase, north_mask, rA, single_n, zero_yy);
}

static void dispatch_search_fast(
    uint32_t grid, uint32_t threads, cudaStream_t stream,
    const PackedBlock* db, uint32_t nb, uint32_t* df, uint32_t* dc,
    uint32_t cnt, uint32_t items, int32_t ymin, int32_t zmin,
    uint32_t ny, uint32_t nz, uint32_t nz_inv,
    uint32_t tasks, uint32_t yoff, uint32_t zoff, int32_t xbase,
    uint32_t north_mask, int32_t rA, uint32_t anchor_side, int single_n,
    uint32_t zero_yy, bool b012_nonneg, bool b1_same_y) {
  int mode = (single_n >= 0) ? 0 : (north_mask == 0xFu ? 2 : 1);
  uint32_t smem_bytes = kernel_smem_bytes(mode, threads);

  #define LAUNCH_KERNEL(M, AS, NNY, SY, FI) \
    launch_fast_typed<M, AS, NNY, SY, FI>( \
        grid, threads, smem_bytes, stream, \
        db, nb, df, dc, cnt, items, ymin, zmin, ny, nz, nz_inv, \
        tasks, yoff, zoff, xbase, north_mask, rA, single_n, zero_yy)

  #define DISPATCH_ITEMS(M, AS, NNY, SY) \
    if (items == 1u) { LAUNCH_KERNEL(M, AS, NNY, SY, 1u); } \
    else { LAUNCH_KERNEL(M, AS, NNY, SY, 0u); }

  #define DISPATCH_SY(M, AS, NNY) \
    if (b1_same_y) { DISPATCH_ITEMS(M, AS, NNY, true); } \
    else { DISPATCH_ITEMS(M, AS, NNY, false); }

  #define DISPATCH_NNY(M, AS) \
    if (b012_nonneg) { DISPATCH_SY(M, AS, true); } \
    else { DISPATCH_SY(M, AS, false); }

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
  #undef DISPATCH_SY
  #undef DISPATCH_ITEMS
  #undef LAUNCH_KERNEL
}

struct Tune { uint32_t threads = 128, items = 1; };

static Tune autotune(const cudaDeviceProp& prop, const PackedBlock* db, const std::vector<Block>& tuneBlocks,
                     uint32_t* df, uint32_t* dc, cudaStream_t stream,
                     const Options& s_opt, const std::string& mode, bool fast, uint32_t batch,
                     uint32_t north_mask, int32_t rA, uint32_t anchor_side, int single_n,
                     uint32_t nz_inv, uint32_t zero_yy, bool b012_nonneg, bool b1_same_y) {
  Tune fallback{128, 1};
  bool manual = std::getenv("GRASS_THREADS") || std::getenv("GRASS_ITEMS");
  if (manual) {
    uint32_t t = tuning("GRASS_THREADS", 128, 32, 256);
    uint32_t it = tuning("GRASS_ITEMS", 1, 1, 8);
    if (t % 32) throw std::runtime_error("GRASS_THREADS must be multiple of 32");
    return {t, it};
  }
  if (mode == "off") {
    std::cerr << "autotune off threads " << fallback.threads << " items " << fallback.items << "\n";
    return fallback;
  }

  std::string key = std::string(prop.name) + "/" + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                    "/columns-fd-scan-v13/" + std::to_string(s_opt.dim[1]) + "/" + std::to_string(s_opt.dim[2]);
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
        (t == 64 || t == 128 || t == 256) && (it == 1 || it == 2 || it == 4)) {
      std::cerr << "autotune cache hit " << t << " threads " << it << " items\n";
      return {t, it};
    }
  }

  if (mode == "auto" && s_opt.total < 32768) {
    std::cerr << "autotune skipped tiny search threads " << fallback.threads << " items " << fallback.items << "\n";
    return fallback;
  }

  uint32_t sample_cap = (s_opt.dim[1] >= 128u) ? 16777216u : 4194304u;
  uint32_t n = uint32_t(std::min<uint64_t>(batch, std::min<uint64_t>(s_opt.total, sample_cap)));
  std::vector<uint32_t> reference;
  float best = std::numeric_limits<float>::max();
  Tune selected = fallback;

  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));

  for (uint32_t t : {64u, 128u, 256u}) {
    for (uint32_t it : {1u, 2u, 4u}) {
      auto launch = [&]() {
        check(cudaMemsetAsync(dc, 0, sizeof(uint32_t), stream));
        if (fast) {
          uint32_t tasks = uint32_t((n + s_opt.dim[1] - 1) / s_opt.dim[1]);
          uint32_t grid = (tasks + t * it - 1) / (t * it);
          dispatch_search_fast(
              grid, t, stream, db, uint32_t(tuneBlocks.size()), df, dc,
              n, it, s_opt.lo[1], s_opt.lo[2], uint32_t(s_opt.dim[1]), uint32_t(s_opt.dim[2]), nz_inv,
              tasks, 0, 0, s_opt.lo[0], north_mask, rA, anchor_side, single_n, zero_yy, b012_nonneg, b1_same_y);
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
                       [](const Block& a, const Block& b) {
                         if (a.side != b.side) return a.side < b.side;
                         bool a_same_col = (a.x == 0 && a.z == 0);
                         bool b_same_col = (b.x == 0 && b.z == 0);
                         if (a_same_col != b_same_col) return a_same_col;
                         bool a_y0 = (a.y == 0);
                         bool b_y0 = (b.y == 0);
                         if (a_y0 != b_y0) return a_y0;
                         return (a.y >= 0) && (b.y < 0);
                       });
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

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    uint32_t default_batch = 16777216u;
    if (s_opt.dim[1] >= 128u && prop.totalGlobalMem >= (4ULL << 30) && !anchor_side) {
      default_batch = 67108864u;
    } else if (s_opt.dim[1] >= 128u && prop.totalGlobalMem >= (3ULL << 30)) {
      default_batch = 33554432u;
    }
    const uint32_t batch = tuning("GRASS_BATCH", default_batch, 256, 268435456);
    bool fast = s_opt.dim[1] <= 4096 && s_opt.dim[2] <= uint64_t(UINT32_MAX - batch) && !std::getenv("GRASS_CUDA_FORCE_WIDE");

    uint32_t nz32 = fast ? uint32_t(s_opt.dim[2]) : 1u;
    uint32_t nz_inv = (nz32 <= 1u) ? 0xFFFFFFFFu : uint32_t(0x100000000ULL / nz32);
    uint32_t ny32 = fast ? uint32_t(s_opt.dim[1]) : 1u;
    uint32_t zero_yy = (s_opt.lo[1] >= 0)
        ? 0u
        : ((int64_t(s_opt.lo[1]) + int64_t(ny32) <= 0) ? ny32 : uint32_t(-int64_t(s_opt.lo[1])));
    bool b012_nonneg = (s_opt.lo[1] >= 0) &&
        (original.size() <= 1 || int64_t(s_opt.lo[1]) + int64_t(original[1].y) >= 0) &&
        (original.size() <= 2 || int64_t(s_opt.lo[1]) + int64_t(original[2].y) >= 0);
    bool b1_same_y = (original.size() > 1 && original[1].y == 0);

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

    Tune tune = autotune(prop, db, original, df[0], dc[0], streams[0], s_opt, mode, fast, alloc_batch,
                         north_mask, rA, anchor_side, single_n, nz_inv, zero_yy, b012_nonneg, b1_same_y);
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
            tasks, yoff, zoff, xbase, north_mask, rA, anchor_side, single_n, zero_yy, b012_nonneg, b1_same_y);
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
