#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "solver_common.h"
#include <cstring>
#include <cstdlib>
#include <vector>

struct alignas(16) PackedBlock {
  int32_t x, y, z;
  uint32_t meta; // bits 0..1: expected rotation, bits 8..9: mask (1 for side, 3 for top)
};

struct Params {
  uint64_t base, ny, nz;
  uint32_t count, nb;
  int32_t xmin, ymin, zmin;
  uint32_t items, threads, fast, yoff, zoff;
  int32_t xbase;
  uint32_t active_norths;
  int32_t anchor_r, anchor_side;
  uint32_t nz_inv, zero_yy, tiles_per_col, tpc_inv, y_tile;
  uint32_t pad;
};

static const char* source = R"MSL(#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;
typedef int int32_t;
typedef ulong uint64_t;
typedef long int64_t;

constant int FC_SINGLE_NORTH [[function_constant(0)]];
constant bool FC_ALL_FOUR    [[function_constant(1)]];
constant bool FC_ANCHOR_SIDE [[function_constant(2)]];
constant bool FC_NON_NEG_Y   [[function_constant(3)]];
constant bool FC_HAS_B1      [[function_constant(4)]];
constant bool FC_HAS_B2      [[function_constant(5)]];

struct PackedBlock {
  int x, y, z;
  uint meta;
};

struct Params {
  ulong base, ny, nz;
  uint count, nb;
  int xmin, ymin, zmin;
  uint items, threads, fast, yoff, zoff;
  int xbase;
  uint active_norths;
  int anchor_r, anchor_side;
  uint nz_inv, zero_yy, tiles_per_col, tpc_inv, y_tile;
  uint pad;
};

inline void rotation_xz_u32(int x, int z, thread uint& xz_lo, thread uint& xz_hi) {
  uint xp = uint(x) * 3129871u;
  uint sx_hi = uint(as_type<int>(xp) >> 31);
  uint zu = uint(z);
  uint zp_lo = zu * 116129781u;
  uint zp_hi = mulhi(zu, 116129781u) + uint(z >> 31) * 116129781u;
  xz_lo = xp ^ zp_lo;
  xz_hi = sx_hi ^ zp_hi;
}

inline uint rotation_y_lh(uint xz_lo, uint K1, uint K0, uint y_lo) {
  uint lx = xz_lo ^ y_lo;
  uint s_lo = lx * lx;
  uint s_hi = mulhi(lx, lx);
  uint lx11_lo = lx * 11u;
  uint lx11_hi = mulhi(lx, 11u);
  uint px = s_lo * 42317861u;
  uint rx = px + lx11_lo;
  uint py = mulhi(s_lo, 42317861u) + s_hi * 42317861u + K1 * lx + K0 + lx11_hi + (rx < px ? 1u : 0u);
  uint seed_lo = ((rx >> 16u) | (py << 16u)) ^ 0xDEECE66Du;
  uint seed_hi = (py >> 16u) ^ 5u;
  uint prod_lo = seed_lo * 0xDEECE66Du;
  uint step_hi = mulhi(seed_lo, 0xDEECE66Du) + seed_hi * 0xDEECE66Du + seed_lo * 5u + (prod_lo > 0xFFFFFFF4u ? 1u : 0u);
  return (step_hi >> 14u) & 3u;
}

inline uint rotation_y_u32(uint xz_lo, uint xz_hi, int y) {
  uint ly = xz_hi ^ uint(y >> 31);
  return rotation_y_lh(xz_lo, 84635722u * ly, 11u * ly, uint(y));
}

inline uint4 rotation_y4_lh(uint xz_lo, uint K1, uint K0, uint base_y) {
  return uint4(
    rotation_y_lh(xz_lo, K1, K0, base_y),
    rotation_y_lh(xz_lo, K1, K0, base_y + 1u),
    rotation_y_lh(xz_lo, K1, K0, base_y + 2u),
    rotation_y_lh(xz_lo, K1, K0, base_y + 3u)
  );
}

inline uint2 rotation_y2_lh(uint xz_lo, uint K1, uint K0, uint base_y) {
  return uint2(
    rotation_y_lh(xz_lo, K1, K0, base_y),
    rotation_y_lh(xz_lo, K1, K0, base_y + 1u)
  );
}

inline uint texture_u32(int bx, int by, int bz, uint mask) {
  uint xz_lo, xz_hi;
  rotation_xz_u32(bx, bz, xz_lo, xz_hi);
  return rotation_y_u32(xz_lo, xz_hi, by) & mask;
}

inline void emit(device uint* results, uint val) {
  uint slot = atomic_fetch_add_explicit((device atomic_uint*)results, 1u, memory_order_relaxed);
  results[slot + 1u] = val;
}

inline int wrap_add(int a, int b) {
  return as_type<int>(uint(a) + uint(b));
}

inline bool check_tail(constant PackedBlock* blocks, uint nb, uint north, uint start_i, int x, int y, int z) {
  constant PackedBlock* b = blocks + north * nb;
  uint bound = min(nb, 8u);
  #pragma unroll
  for (uint i = start_i; i < 8u; i++) {
    if (i >= bound) break;
    PackedBlock bi = b[i];
    uint r = texture_u32(wrap_add(x, bi.x), wrap_add(y, bi.y), wrap_add(z, bi.z), bi.meta >> 8u);
    if (r != (bi.meta & 3u)) return false;
  }
  for (uint i = 8u; i < nb; i++) {
    PackedBlock bi = b[i];
    uint r = texture_u32(wrap_add(x, bi.x), wrap_add(y, bi.y), wrap_add(z, bi.z), bi.meta >> 8u);
    if (r != (bi.meta & 3u)) return false;
  }
  return true;
}

inline bool check_single(constant PackedBlock* blocks, uint nb, uint north,
                         int x, int y, int z,
                         uint xz1_lo, uint xz1_hi, int b1_y, uint b1_meta,
                         uint xz2_lo, uint xz2_hi, int b2_y, uint b2_meta) {
  if (FC_HAS_B1) {
    uint r1 = rotation_y_u32(xz1_lo, xz1_hi, wrap_add(y, b1_y));
    if ((r1 & (b1_meta >> 8u)) != (b1_meta & 3u)) return false;
  }
  if (FC_HAS_B2) {
    uint r2 = rotation_y_u32(xz2_lo, xz2_hi, wrap_add(y, b2_y));
    if ((r2 & (b2_meta >> 8u)) != (b2_meta & 3u)) return false;
  }
  if (nb > 3u) {
    return check_tail(blocks, nb, north, 3u, x, y, z);
  }
  return true;
}

inline uint sel4_u32(uint n, uint v0, uint v1, uint v2, uint v3) {
  uint lo = (n & 1u) ? v1 : v0;
  uint hi = (n & 1u) ? v3 : v2;
  return (n & 2u) ? hi : lo;
}

inline bool check_multi(constant PackedBlock* blocks, uint nb, uint north,
                        int x, int y, int z,
                        uint xz1_lo0, uint xz1_lo1, uint xz1_lo2, uint xz1_lo3,
                        uint xz1_hi0, uint xz1_hi1, uint xz1_hi2, uint xz1_hi3,
                        int b1_y, uint b1_r_pack, uint b1_mask) {
  if (FC_HAS_B1) {
    uint lo = sel4_u32(north, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3);
    uint hi = sel4_u32(north, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3);
    uint r1 = rotation_y_u32(lo, hi, wrap_add(y, b1_y));
    uint exp_r = (b1_r_pack >> (north * 2u)) & 3u;
    if ((r1 & b1_mask) != exp_r) return false;
  }
  if (nb > 2u) {
    return check_tail(blocks, nb, north, 2u, x, y, z);
  }
  return true;
}

kernel void search(constant PackedBlock* blocks [[buffer(0)]],
                   device uint* results [[buffer(1)]],
                   constant Params& p [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]) {
  for (uint item = 0; item < p.items; item++) {
    uint task = tid + item * p.threads;
    if (p.fast) {
      uint ny = uint(p.ny), nz = uint(p.nz);
      uint tpc = p.tiles_per_col;
      uint total_cols = (p.count + p.yoff + ny - 1u) / ny;
      uint total_tasks = total_cols * tpc;
      if (task >= total_tasks) break;

      uint col, tile;
      if (tpc == 1u) {
        col = task;
        tile = 0u;
      } else {
        col = mulhi(task, p.tpc_inv);
        tile = task - col * tpc;
        if (tile >= tpc) {
          tile -= tpc;
          col++;
        }
      }

      uint zid = col + p.zoff;
      uint q = (nz == 1u) ? zid : mulhi(zid, p.nz_inv);
      uint zz = zid - q * nz;
      if (zz >= nz) {
        zz -= nz;
        q++;
      }
      int z = wrap_add(p.zmin, int(zz));
      int x = wrap_add(p.xbase, int(q));

      uint begin, end;
      if (tpc == 1u) {
        begin = (col == 0u) ? p.yoff : 0u;
        end = min(ny, p.count + p.yoff - col * ny);
      } else {
        begin = tile * p.y_tile;
        end = min(ny, begin + p.y_tile);
      }
      uint col_base_idx = col * ny - p.yoff;

      uint xz0_lo, xz0_hi;
      rotation_xz_u32(x, z, xz0_lo, xz0_hi);

      if (FC_SINGLE_NORTH >= 0) {
        const uint sn = uint(FC_SINGLE_NORTH);
        const uint target_r0 = FC_ANCHOR_SIDE ? ((uint(p.anchor_r) + sn) & 1u) : ((uint(p.anchor_r) + sn) & 3u);
        uint xz1_lo = 0u, xz1_hi = 0u, b1_meta = 0u;
        int b1_y = 0;
        if (FC_HAS_B1) {
          PackedBlock b1 = blocks[sn * p.nb + 1u];
          b1_y = b1.y; b1_meta = b1.meta;
          rotation_xz_u32(wrap_add(x, b1.x), wrap_add(z, b1.z), xz1_lo, xz1_hi);
        }
        uint xz2_lo = 0u, xz2_hi = 0u, b2_meta = 0u;
        int b2_y = 0;
        if (FC_HAS_B2) {
          PackedBlock b2 = blocks[sn * p.nb + 2u];
          b2_y = b2.y; b2_meta = b2.meta;
          rotation_xz_u32(wrap_add(x, b2.x), wrap_add(z, b2.z), xz2_lo, xz2_hi);
        }

        for (int seg = 0; seg < 2; seg++) {
          if (FC_NON_NEG_Y && seg == 0) continue;
          uint sb = (seg == 0) ? begin : max(begin, p.zero_yy);
          uint se = (seg == 0) ? min(end, p.zero_yy) : end;
          if (sb >= se) continue;
          uint ly = (seg == 0) ? ~xz0_hi : xz0_hi;
          uint K1 = 84635722u * ly;
          uint K0 = 11u * ly;
          uint yy = sb;
          for (; yy + 3u < se; yy += 4u) {
            uint y0_u = uint(p.ymin) + yy;
            uint4 r0 = rotation_y4_lh(xz0_lo, K1, K0, y0_u);
            bool m0 = FC_ANCHOR_SIDE ? ((r0.x & 1u) == target_r0) : (r0.x == target_r0);
            bool m1 = FC_ANCHOR_SIDE ? ((r0.y & 1u) == target_r0) : (r0.y == target_r0);
            bool m2 = FC_ANCHOR_SIDE ? ((r0.z & 1u) == target_r0) : (r0.z == target_r0);
            bool m3 = FC_ANCHOR_SIDE ? ((r0.w & 1u) == target_r0) : (r0.w == target_r0);
            if (!m0 && !m1 && !m2 && !m3) continue;
            if (m0 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy) << 2u) | sn);
            if (m1 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u + 1u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | sn);
            if (m2 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u + 2u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy + 2u) << 2u) | sn);
            if (m3 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u + 3u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy + 3u) << 2u) | sn);
          }
          for (; yy + 1u < se; yy += 2u) {
            uint y0_u = uint(p.ymin) + yy;
            uint2 r0 = rotation_y2_lh(xz0_lo, K1, K0, y0_u);
            bool m0 = FC_ANCHOR_SIDE ? ((r0.x & 1u) == target_r0) : (r0.x == target_r0);
            bool m1 = FC_ANCHOR_SIDE ? ((r0.y & 1u) == target_r0) : (r0.y == target_r0);
            if (!m0 && !m1) continue;
            if (m0 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy) << 2u) | sn);
            if (m1 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u + 1u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | sn);
          }
          for (; yy < se; yy++) {
            uint y0_u = uint(p.ymin) + yy;
            uint r0 = rotation_y_lh(xz0_lo, K1, K0, y0_u);
            bool m0 = FC_ANCHOR_SIDE ? ((r0 & 1u) == target_r0) : (r0 == target_r0);
            if (m0 && check_single(blocks, p.nb, sn, x, as_type<int>(y0_u), z, xz1_lo, xz1_hi, b1_y, b1_meta, xz2_lo, xz2_hi, b2_y, b2_meta))
              emit(results, ((col_base_idx + yy) << 2u) | sn);
          }
        }
      } else {
        uint xz1_lo0 = 0u, xz1_lo1 = 0u, xz1_lo2 = 0u, xz1_lo3 = 0u;
        uint xz1_hi0 = 0u, xz1_hi1 = 0u, xz1_hi2 = 0u, xz1_hi3 = 0u;
        int b1_y = 0;
        uint b1_r_pack = 0u, b1_mask = 0u;
        if (FC_HAS_B1) {
          PackedBlock b0 = blocks[1u];
          b1_y = b0.y;
          b1_mask = b0.meta >> 8u;
          if (FC_ALL_FOUR || (p.active_norths & 1u)) {
            b1_r_pack |= (b0.meta & 3u);
            rotation_xz_u32(wrap_add(x, b0.x), wrap_add(z, b0.z), xz1_lo0, xz1_hi0);
          }
          if (FC_ALL_FOUR || (p.active_norths & 2u)) {
            PackedBlock b1 = blocks[p.nb + 1u];
            b1_r_pack |= ((b1.meta & 3u) << 2u);
            rotation_xz_u32(wrap_add(x, b1.x), wrap_add(z, b1.z), xz1_lo1, xz1_hi1);
          }
          if (FC_ALL_FOUR || (p.active_norths & 4u)) {
            PackedBlock b2 = blocks[2u * p.nb + 1u];
            b1_r_pack |= ((b2.meta & 3u) << 4u);
            rotation_xz_u32(wrap_add(x, b2.x), wrap_add(z, b2.z), xz1_lo2, xz1_hi2);
          }
          if (FC_ALL_FOUR || (p.active_norths & 8u)) {
            PackedBlock b3 = blocks[3u * p.nb + 1u];
            b1_r_pack |= ((b3.meta & 3u) << 6u);
            rotation_xz_u32(wrap_add(x, b3.x), wrap_add(z, b3.z), xz1_lo3, xz1_hi3);
          }
        }
        const uint bit_pat = FC_ANCHOR_SIDE ? 5u : 1u;

        for (int seg = 0; seg < 2; seg++) {
          if (FC_NON_NEG_Y && seg == 0) continue;
          uint sb = (seg == 0) ? begin : max(begin, p.zero_yy);
          uint se = (seg == 0) ? min(end, p.zero_yy) : end;
          if (sb >= se) continue;
          uint ly = (seg == 0) ? ~xz0_hi : xz0_hi;
          uint K1 = 84635722u * ly;
          uint K0 = 11u * ly;
          uint yy = sb;
          for (; yy + 3u < se; yy += 4u) {
            uint y0_u = uint(p.ymin) + yy;
            uint4 r0 = rotation_y4_lh(xz0_lo, K1, K0, y0_u);
            uint n0 = FC_ANCHOR_SIDE ? ((r0.x - uint(p.anchor_r)) & 1u) : ((r0.x - uint(p.anchor_r)) & 3u);
            uint n1 = FC_ANCHOR_SIDE ? ((r0.y - uint(p.anchor_r)) & 1u) : ((r0.y - uint(p.anchor_r)) & 3u);
            uint n2 = FC_ANCHOR_SIDE ? ((r0.z - uint(p.anchor_r)) & 1u) : ((r0.z - uint(p.anchor_r)) & 3u);
            uint n3 = FC_ANCHOR_SIDE ? ((r0.w - uint(p.anchor_r)) & 1u) : ((r0.w - uint(p.anchor_r)) & 3u);

            if (!FC_ALL_FOUR) {
              bool m0 = (p.active_norths & (bit_pat << n0)) != 0u;
              bool m1 = (p.active_norths & (bit_pat << n1)) != 0u;
              bool m2 = (p.active_norths & (bit_pat << n2)) != 0u;
              bool m3 = (p.active_norths & (bit_pat << n3)) != 0u;
              if (!m0 && !m1 && !m2 && !m3) continue;
            }

            if ((FC_ALL_FOUR || (p.active_norths & (1u << n0))) &&
                check_multi(blocks, p.nb, n0, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | n0);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n0 + 2u)))) &&
                check_multi(blocks, p.nb, n0 + 2u, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | (n0 + 2u));

            if ((FC_ALL_FOUR || (p.active_norths & (1u << n1))) &&
                check_multi(blocks, p.nb, n1, x, as_type<int>(y0_u + 1u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | n1);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n1 + 2u)))) &&
                check_multi(blocks, p.nb, n1 + 2u, x, as_type<int>(y0_u + 1u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | (n1 + 2u));

            if ((FC_ALL_FOUR || (p.active_norths & (1u << n2))) &&
                check_multi(blocks, p.nb, n2, x, as_type<int>(y0_u + 2u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 2u) << 2u) | n2);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n2 + 2u)))) &&
                check_multi(blocks, p.nb, n2 + 2u, x, as_type<int>(y0_u + 2u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 2u) << 2u) | (n2 + 2u));

            if ((FC_ALL_FOUR || (p.active_norths & (1u << n3))) &&
                check_multi(blocks, p.nb, n3, x, as_type<int>(y0_u + 3u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 3u) << 2u) | n3);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n3 + 2u)))) &&
                check_multi(blocks, p.nb, n3 + 2u, x, as_type<int>(y0_u + 3u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 3u) << 2u) | (n3 + 2u));
          }
          for (; yy + 1u < se; yy += 2u) {
            uint y0_u = uint(p.ymin) + yy;
            uint2 r0 = rotation_y2_lh(xz0_lo, K1, K0, y0_u);
            uint n0 = FC_ANCHOR_SIDE ? ((r0.x - uint(p.anchor_r)) & 1u) : ((r0.x - uint(p.anchor_r)) & 3u);
            uint n1 = FC_ANCHOR_SIDE ? ((r0.y - uint(p.anchor_r)) & 1u) : ((r0.y - uint(p.anchor_r)) & 3u);
            if (!FC_ALL_FOUR && ((p.active_norths & (bit_pat << n0)) == 0u) && ((p.active_norths & (bit_pat << n1)) == 0u)) continue;
            if ((FC_ALL_FOUR || (p.active_norths & (1u << n0))) &&
                check_multi(blocks, p.nb, n0, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | n0);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n0 + 2u)))) &&
                check_multi(blocks, p.nb, n0 + 2u, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | (n0 + 2u));

            if ((FC_ALL_FOUR || (p.active_norths & (1u << n1))) &&
                check_multi(blocks, p.nb, n1, x, as_type<int>(y0_u + 1u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | n1);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n1 + 2u)))) &&
                check_multi(blocks, p.nb, n1 + 2u, x, as_type<int>(y0_u + 1u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy + 1u) << 2u) | (n1 + 2u));
          }
          for (; yy < se; yy++) {
            uint y0_u = uint(p.ymin) + yy;
            uint r0 = rotation_y_lh(xz0_lo, K1, K0, y0_u);
            uint n0 = FC_ANCHOR_SIDE ? ((r0 - uint(p.anchor_r)) & 1u) : ((r0 - uint(p.anchor_r)) & 3u);
            if ((FC_ALL_FOUR || (p.active_norths & (1u << n0))) &&
                check_multi(blocks, p.nb, n0, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | n0);
            if (FC_ANCHOR_SIDE && (FC_ALL_FOUR || (p.active_norths & (1u << (n0 + 2u)))) &&
                check_multi(blocks, p.nb, n0 + 2u, x, as_type<int>(y0_u), z, xz1_lo0, xz1_lo1, xz1_lo2, xz1_lo3, xz1_hi0, xz1_hi1, xz1_hi2, xz1_hi3, b1_y, b1_r_pack, b1_mask))
              emit(results, ((col_base_idx + yy) << 2u) | (n0 + 2u));
          }
        }
      }
    } else {
      uint index = task;
      if (index >= p.count) break;
      ulong id = p.base + index;
      int y = wrap_add(p.ymin, int(id % p.ny)); id /= p.ny;
      int z = wrap_add(p.zmin, int(id % p.nz)); id /= p.nz;
      int x = wrap_add(p.xmin, int(id));
      uint r0 = texture_u32(x, y, z, 3u);
      if (FC_ANCHOR_SIDE) {
        uint n0 = (r0 - uint(p.anchor_r)) & 1u;
        if ((p.active_norths & (1u << n0)) && check_tail(blocks, p.nb, n0, 1u, x, y, z))
          emit(results, (index << 2u) | n0);
        uint n1 = n0 + 2u;
        if ((p.active_norths & (1u << n1)) && check_tail(blocks, p.nb, n1, 1u, x, y, z))
          emit(results, (index << 2u) | n1);
      } else {
        uint n = (r0 - uint(p.anchor_r)) & 3u;
        if ((p.active_norths & (1u << n)) && check_tail(blocks, p.nb, n, 1u, x, y, z))
          emit(results, (index << 2u) | n);
      }
    }
  }
}
)MSL";

struct Tune { uint32_t width = 128, items = 1; };

static uint64_t hashBytes(uint64_t h, const void* p, size_t n) {
  const auto* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void compute_tile_params(uint32_t ny, uint32_t yoff, uint32_t cnt,
                                uint32_t& tiles_per_col, uint32_t& tpc_inv, uint32_t& y_tile, uint32_t& tasks) {
  // Keep each thread processing an entire y column (up to ny <= 4096) so rotation_xz(x,z)
  // and shifted b1/b2 column hashes are computed once rather than re-hashed across y tiles.
  y_tile = ny;
  tiles_per_col = 1u;
  tpc_inv = 0xFFFFFFFFu;
  tasks = uint32_t((uint64_t(cnt) + yoff + ny - 1u) / ny);
}

static id<MTLCommandBuffer> launch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipe,
                                   id<MTLBuffer> bb, id<MTLBuffer> bf, Params p, Tune t) {
  p.items = t.items;
  p.fast = p.ny <= 4096 && p.nz <= UINT32_MAX - 16777216u && !std::getenv("GRASS_METAL_FORCE_WIDE");
  uint64_t origin = p.base;
  p.yoff = uint32_t(origin % p.ny); origin /= p.ny;
  p.zoff = uint32_t(origin % p.nz); origin /= p.nz;
  p.xbase = int32_t(int64_t(p.xmin) + int64_t(origin));
  uint32_t tasks = p.count;
  if (p.fast) {
    uint32_t ny32 = uint32_t(p.ny);
    compute_tile_params(ny32, p.yoff, p.count, p.tiles_per_col, p.tpc_inv, p.y_tile, tasks);
  } else {
    p.tiles_per_col = 1u;
    p.tpc_inv = 0xFFFFFFFFu;
    p.y_tile = uint32_t(p.ny);
  }
  p.threads = (tasks + t.items - 1u) / t.items;
  *static_cast<uint32_t*>(bf.contents) = 0;
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) throw std::runtime_error("Metal encoder allocation failed");
  [enc setComputePipelineState:pipe];
  [enc setBuffer:bb offset:0 atIndex:0];
  [enc setBuffer:bf offset:0 atIndex:1];
  [enc setBytes:&p length:sizeof(p) atIndex:2];
  [enc dispatchThreads:MTLSizeMake(p.threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(t.width, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  return cmd;
}

static double dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipe,
                       id<MTLBuffer> bb, id<MTLBuffer> bf, Params p, Tune t) {
  @autoreleasepool {
    auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cmd = launch(queue, pipe, bb, bf, p, t);
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(cmd.error ? [[cmd.error localizedDescription] UTF8String] : "Metal dispatch failed");
    double gpu = cmd.GPUEndTime - cmd.GPUStartTime;
    return gpu > 0 ? gpu : std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
}

static std::vector<uint32_t> resultIds(id<MTLBuffer> bf, uint32_t count, uint32_t max_entries) {
  auto* r = static_cast<uint32_t*>(bf.contents);
  if (r[0] > max_entries) throw std::runtime_error("GPU result capacity exceeded");
  std::vector<uint32_t> ids(r + 1, r + 1 + r[0]);
  std::sort(ids.begin(), ids.end());
  for (size_t i = 0; i < ids.size(); i++)
    if ((ids[i] >> 2) >= count || (i && ids[i] == ids[i - 1]))
      throw std::runtime_error("Invalid/duplicate GPU result index");
  return ids;
}

static NSString* cachePath() {
  const char* env = std::getenv("GRASS_METAL_AUTOTUNE_CACHE");
  if (env && *env) return [NSString stringWithUTF8String:env];
  return [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject]
      stringByAppendingPathComponent:@"grass-solver/metal-autotune-v2.plist"];
}

static Tune autotune(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipe,
                     id<MTLBuffer> bb, id<MTLBuffer> bf, const std::vector<PackedBlock>& blocks, const Options& s_opt,
                     const std::string& mode, NSString* path, uint32_t active_norths, int32_t anchor_r,
                     int32_t anchor_side, uint32_t nb, uint32_t nz_inv, uint32_t zero_yy, uint32_t north_count) {
  Tune fallback{uint32_t(std::min<NSUInteger>(128, pipe.maxTotalThreadsPerThreadgroup)), 1};
  if (mode == "off") {
    std::cerr << "autotune off width " << fallback.width << " items 1\n";
    return fallback;
  }
  uint64_t h = hashBytes(14695981039346656037ULL, source, std::strlen(source));
  h = hashBytes(h, blocks.data(), blocks.size() * sizeof(PackedBlock));
  h = hashBytes(h, s_opt.dim, sizeof(s_opt.dim));
  h = hashBytes(h, s_opt.lo, sizeof(s_opt.lo));
  bool wide = std::getenv("GRASS_METAL_FORCE_WIDE") != nullptr;
  h = hashBytes(h, &wide, sizeof(wide));
  h = hashBytes(h, &active_norths, sizeof(active_norths));
  h = hashBytes(h, &anchor_r, sizeof(anchor_r));
  h = hashBytes(h, &anchor_side, sizeof(anchor_side));
  NSString* key = [NSString stringWithFormat:@"columns-u32-v4|%@|%@|%@|%lu|%lu|%016llx",
                                             device.name, device.architecture.name,
                                             NSProcessInfo.processInfo.operatingSystemVersionString,
                                             (unsigned long)pipe.threadExecutionWidth,
                                             (unsigned long)pipe.maxTotalThreadsPerThreadgroup,
                                             (unsigned long long)h];
  NSMutableDictionary* cache = [[NSDictionary dictionaryWithContentsOfFile:path] mutableCopy];
  if (!cache) cache = [NSMutableDictionary dictionary];
  NSDictionary* record = cache[key];
  if (mode == "auto" && [record isKindOfClass:[NSDictionary class]]) {
    id w = record[@"width"], i = record[@"items"], ms = record[@"milliseconds"];
    if ([w isKindOfClass:[NSNumber class]] && [i isKindOfClass:[NSNumber class]] && [ms isKindOfClass:[NSNumber class]]) {
      uint32_t width = [w unsignedIntValue], items = [i unsignedIntValue];
      if ((width == 64 || width == 128 || width == 256 || width == 512) &&
          width <= pipe.maxTotalThreadsPerThreadgroup &&
          width % pipe.threadExecutionWidth == 0 &&
          (items == 1 || items == 2 || items == 4 || items == 8) &&
          [ms doubleValue] > 0) {
        std::cerr << "autotune cache hit width " << width << " items " << items
                  << " measured_ms " << [ms doubleValue] << " cache " << path.UTF8String << "\n";
        return {width, items};
      }
    }
  }
  if (mode == "auto" && s_opt.total < 32768) {
    std::cerr << "autotune skipped tiny search width " << fallback.width << " items 1\n";
    return fallback;
  }
  auto start = std::chrono::steady_clock::now();
  uint32_t raw_count = uint32_t(std::min<uint64_t>(262144, s_opt.total));
  uint32_t ny32 = uint32_t(s_opt.dim[1]);
  uint32_t count = (s_opt.dim[1] <= 4096 && raw_count >= ny32) ? ((raw_count / ny32) * ny32) : raw_count;
  uint32_t max_entries = count * ((anchor_side && north_count > 1u) ? 2u : 1u);
  std::vector<uint64_t> bases{0};
  if (s_opt.total > count) {
    bases.push_back((s_opt.total - count) / 2);
    bases.push_back(s_opt.total - count);
  }
  std::vector<std::vector<uint32_t>> expected;
  auto params = [&](uint64_t base) {
    return Params{base, s_opt.dim[1], s_opt.dim[2], count, nb,
                  s_opt.lo[0], s_opt.lo[1], s_opt.lo[2],
                  1, count, 0, 0, 0, 0,
                  active_norths, anchor_r, anchor_side,
                  nz_inv, zero_yy, 1u, 0xFFFFFFFFu, ny32, 0u};
  };
  for (auto base : bases) {
    dispatch(queue, pipe, bb, bf, params(base), fallback);
    expected.push_back(resultIds(bf, count, max_entries));
  }
  Tune best = fallback;
  double bestTime = std::numeric_limits<double>::max();
  int tested = 0;
  std::vector<Tune> variants;
  for (uint32_t width : {64u, 128u, 256u, 512u})
    for (uint32_t items : {1u, 2u, 4u, 8u})
      if (width <= pipe.maxTotalThreadsPerThreadgroup && width % pipe.threadExecutionWidth == 0)
        variants.push_back({width, items});
  std::vector<std::vector<double>> times(variants.size());
  for (auto candidate : variants) dispatch(queue, pipe, bb, bf, params(bases[0]), candidate);
  for (size_t round = 0; round < 3; round++)
    for (size_t k = 0; k < variants.size(); k++) {
      size_t v = (k * 5 + round * 7) % variants.size();
      double seconds = 0;
      for (size_t j = 0; j < bases.size(); j++) {
        std::memset(bf.contents, 0xff, (size_t(max_entries) + 1) * 4);
        seconds += dispatch(queue, pipe, bb, bf, params(bases[j]), variants[v]);
        if (resultIds(bf, count, max_entries) != expected[j])
          throw std::runtime_error("Autotune variant correctness mismatch");
      }
      times[v].push_back(seconds / bases.size());
    }
  for (size_t v = 0; v < variants.size(); v++) {
    std::sort(times[v].begin(), times[v].end());
    double seconds = times[v][1];
    tested++;
    std::cerr << "autotune sample width " << variants[v].width << " items " << variants[v].items
              << " gpu_ms " << seconds * 1000 << "\n";
    if (seconds < bestTime) {
      bestTime = seconds;
      best = variants[v];
    }
  }
  if (!tested) throw std::runtime_error("No valid Metal autotune variants");
  cache[key] = @{@"width": @(best.width), @"items": @(best.items), @"milliseconds": @(bestTime * 1000)};
  if (cache.count > 64) {
    [cache removeAllObjects];
    cache[key] = @{@"width": @(best.width), @"items": @(best.items), @"milliseconds": @(bestTime * 1000)};
  }
  NSError* err = nil;
  BOOL dir = [[NSFileManager defaultManager] createDirectoryAtPath:path.stringByDeletingLastPathComponent
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:&err];
  BOOL saved = dir && [cache writeToFile:path atomically:YES];
  std::cerr << "autotune selected width " << best.width << " items " << best.items
            << " measured_ms " << bestTime * 1000
            << " variants " << tested << " verified_samples " << tested * 3 * bases.size()
            << " tune_seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << " cache " << path.UTF8String << (saved ? " saved" : " not-saved") << "\n";
  return best;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      std::string mode = "auto";
      NSString* path = cachePath();
      bool clear = false;
      std::vector<char*> args{argv[0]};
      for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--autotune") {
          if (++i == argc) throw std::runtime_error("Missing autotune mode");
          mode = argv[i];
        } else if (a == "--autotune-cache") {
          if (++i == argc) throw std::runtime_error("Missing cache path");
          path = [NSString stringWithUTF8String:argv[i]];
        } else if (a == "--clear-autotune-cache") {
          clear = true;
        } else {
          if (a == "--help" || a == "-h")
            std::cout << "Metal options: --autotune auto|force|off (default auto), --autotune-cache FILE, --clear-autotune-cache\n";
          args.push_back(argv[i]);
        }
      }
      if (mode != "auto" && mode != "force" && mode != "off")
        throw std::runtime_error("autotune must be auto, force or off");
      auto o = options(int(args.size()), args.data());
      auto original = load(o.path);
      for (int n : o.north) orient(original, n, o);
      if (clear) {
        NSError* error = nil;
        if ([[NSFileManager defaultManager] fileExistsAtPath:path] &&
            ![[NSFileManager defaultManager] removeItemAtPath:path error:&error])
          throw std::runtime_error("Cannot clear autotune cache");
        std::cerr << "autotune cache cleared " << path.UTF8String << "\n";
      }

      size_t anchor_idx = 0;
      for (size_t i = 0; i < original.size(); i++) {
        if (original[i].side == 0) {
          anchor_idx = i;
          break;
        }
      }
      int32_t xA = original[anchor_idx].x;
      int32_t yA = original[anchor_idx].y;
      int32_t zA = original[anchor_idx].z;
      int32_t rA = original[anchor_idx].r;
      int32_t sideA = original[anchor_idx].side;
      if (anchor_idx != 0) std::swap(original[0], original[anchor_idx]);
      for (auto& b : original) {
        b.x -= xA;
        b.y -= yA;
        b.z -= zA;
      }

      int32_t xA_n[4], zA_n[4];
      int64_t min_rx = INT64_MAX, max_rx = INT64_MIN;
      int64_t min_rz = INT64_MAX, max_rz = INT64_MIN;
      for (int n = 0; n < 4; n++) {
        int64_t x = xA, z = zA;
        for (int j = 0; j < n; j++) {
          int64_t old = x;
          x = -z;
          z = old;
        }
        xA_n[n] = coord(x);
        zA_n[n] = coord(z);
      }

      uint32_t active_norths = 0;
      for (int n : o.north) {
        active_norths |= (1u << n);
        min_rx = std::min<int64_t>(min_rx, xA_n[n]);
        max_rx = std::max<int64_t>(max_rx, xA_n[n]);
        min_rz = std::min<int64_t>(min_rz, zA_n[n]);
        max_rz = std::max<int64_t>(max_rz, zA_n[n]);
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

      uint32_t nb = uint32_t(original.size());
      std::vector<PackedBlock> all_blocks(4 * nb);
      for (int n = 0; n < 4; n++) {
        std::vector<Block> oriented = original;
        for (auto& v : oriented) {
          int64_t x = v.x, z = v.z;
          for (int j = 0; j < n; j++) {
            int64_t old = x;
            x = -z;
            z = old;
          }
          v.x = coord(x);
          v.z = coord(z);
          v.r = (v.r + n) & 3;
          if (v.side) v.r &= 1;
        }
        if (oriented.size() > 1) {
          std::stable_sort(oriented.begin() + 1, oriented.end(),
                           [](const Block& a, const Block& b) { return a.side < b.side; });
        }
        for (size_t i = 0; i < nb; i++) {
          uint32_t mask = oriented[i].side ? 1u : 3u;
          all_blocks[size_t(n) * nb + i] = PackedBlock{oriented[i].x, oriented[i].y, oriented[i].z, uint32_t(oriented[i].r) | (mask << 8u)};
        }
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("No Metal device");
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
      if (!lib) throw std::runtime_error([[error localizedDescription] UTF8String]);

      bool fc_all_four = (active_norths == 0xFu);
      bool fc_anchor_side = (sideA != 0);
      bool fc_non_neg_y = (s_opt.lo[1] >= 0);
      bool fc_has_b1 = (nb > 1u);
      bool fc_has_b2 = (nb > 2u);
      MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
      [fcv setConstantValue:&single_n type:MTLDataTypeInt atIndex:0];
      [fcv setConstantValue:&fc_all_four type:MTLDataTypeBool atIndex:1];
      [fcv setConstantValue:&fc_anchor_side type:MTLDataTypeBool atIndex:2];
      [fcv setConstantValue:&fc_non_neg_y type:MTLDataTypeBool atIndex:3];
      [fcv setConstantValue:&fc_has_b1 type:MTLDataTypeBool atIndex:4];
      [fcv setConstantValue:&fc_has_b2 type:MTLDataTypeBool atIndex:5];
      id<MTLFunction> fn = [lib newFunctionWithName:@"search" constantValues:fcv error:&error];
      if (!fn) throw std::runtime_error([[error localizedDescription] UTF8String]);
      id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];
      if (!pipe) throw std::runtime_error([[error localizedDescription] UTF8String]);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("Metal queue allocation failed");

      const uint32_t ny32 = uint32_t(std::min<uint64_t>(4096u, s_opt.dim[1]));
      const uint32_t batch = 16777216u;
      const uint32_t raw_alloc = uint32_t(std::max<uint64_t>(256, std::min<uint64_t>(batch, s_opt.total)));
      const uint32_t alloc_batch = (s_opt.dim[1] <= 4096 && raw_alloc >= ny32) ? ((raw_alloc / ny32) * ny32) : raw_alloc;
      const uint32_t result_mult = (sideA && o.north.size() > 1u) ? 2u : 1u;
      const size_t max_buf_len = (size_t(alloc_batch) * result_mult + 1u) * 4u;

      static constexpr int NUM_SLOTS = 3;
      id<MTLBuffer> bf[NUM_SLOTS] = {};
      for (int s = 0; s < NUM_SLOTS; s++) {
        bf[s] = [device newBufferWithLength:max_buf_len options:MTLResourceStorageModeShared];
        if (!bf[s]) throw std::runtime_error("Metal allocation failed");
      }

      id<MTLBuffer> bb = [device newBufferWithBytes:all_blocks.data()
                                             length:all_blocks.size() * sizeof(PackedBlock)
                                            options:MTLResourceStorageModeShared];
      if (!bb) throw std::runtime_error("Metal block allocation failed");

      bool fast_shape = s_opt.dim[1] <= 4096 && s_opt.dim[2] <= uint64_t(UINT32_MAX - 16777216u) && !std::getenv("GRASS_METAL_FORCE_WIDE");
      uint32_t nz32 = fast_shape ? uint32_t(s_opt.dim[2]) : 1u;
      uint32_t nz_inv = (nz32 <= 1u) ? 0xFFFFFFFFu : uint32_t(0x100000000ULL / nz32);
      uint32_t zero_yy = (s_opt.lo[1] >= 0)
          ? 0u
          : ((int64_t(s_opt.lo[1]) + int64_t(ny32) <= 0) ? ny32 : uint32_t(-int64_t(s_opt.lo[1])));

      Tune tune = autotune(device, queue, pipe, bb, bf[0], all_blocks, s_opt, mode, path,
                           active_norths, rA, sideA, nb, nz_inv, zero_yy, uint32_t(o.north.size()));

      double searchSeconds = 0;
      uint64_t done = 0, matches = 0;
      auto start = std::chrono::steady_clock::now();
      auto last = start;
      std::signal(SIGINT, stop_search);
      std::signal(SIGTERM, stop_search);

      auto searchStart = std::chrono::steady_clock::now();
      uint64_t base = 0;
      struct SlotState {
        id<MTLCommandBuffer> cmd = nil;
        uint64_t base = 0;
        uint32_t count = 0;
      } slots[NUM_SLOTS];

      auto enqueue_slot = [&](int s) {
        if (base >= s_opt.total || stopped) return false;
        uint32_t cnt = uint32_t(std::min<uint64_t>(alloc_batch, s_opt.total - base));
        Params p{base, s_opt.dim[1], s_opt.dim[2], cnt, nb,
                 s_opt.lo[0], s_opt.lo[1], s_opt.lo[2],
                 tune.items, 0, 0, 0, 0, 0,
                 active_norths, rA, sideA,
                 nz_inv, zero_yy, 1u, 0xFFFFFFFFu, ny32, 0u};
        slots[s].cmd = launch(queue, pipe, bb, bf[s], p, tune);
        slots[s].base = base;
        slots[s].count = cnt;
        base += cnt;
        return true;
      };

      for (int s = 0; s < NUM_SLOTS; s++) enqueue_slot(s);

      int cur = 0;
      while (slots[cur].cmd != nil) {
        id<MTLCommandBuffer> cmd = slots[cur].cmd;
        uint64_t curBase = slots[cur].base;
        uint32_t curCount = slots[cur].count;
        slots[cur].cmd = nil;

        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted)
          throw std::runtime_error(cmd.error ? [[cmd.error localizedDescription] UTF8String] : "Metal dispatch failed");
        auto ids = resultIds(bf[cur], curCount, curCount * result_mult);
        enqueue_slot(cur);

        for (uint32_t val : ids) {
          uint32_t i = val >> 2;
          uint32_t north = val & 3;
          int32_t X, Y, Z;
          decode(curBase + i, s_opt, X, Y, Z);
          int64_t ox = int64_t(X) - xA_n[north];
          int64_t oy = int64_t(Y) - yA;
          int64_t oz = int64_t(Z) - zA_n[north];
          if (has_anchor_shift && (ox < o.lo[0] || ox > o.hi[0] || oz < o.lo[2] || oz > o.hi[2])) continue;
          std::cout << coord(ox) << " " << coord(oy) << " " << coord(oz) << " " << north << "\n";
          matches++;
        }
        if (!std::cout) throw std::runtime_error("Output write failed");
        done = std::min<uint64_t>(o.total, done + curCount);
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() >= 1) {
          double sec = std::chrono::duration<double>(now - start).count();
          std::cerr << "progress " << done << "/" << o.total
                    << " rate " << double(done) / sec << " candidates/s matches " << matches << "\n";
          last = now;
        }
        cur = (cur + 1) % NUM_SLOTS;
      }
      if (!stopped && done < o.total) done = o.total;
      searchSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::cout.flush();
      std::cerr << "search_loop seconds " << searchSeconds
                << " rate " << double(done) / std::max(searchSeconds, 1e-9)
                << " candidates/s tuning_excluded 1\n";
      std::cerr << (stopped ? "cancelled" : "complete")
                << " total " << done << " matches " << matches
                << " seconds " << sec << " rate " << double(done) / std::max(sec, 1e-9) << " candidates/s\n";
      return stopped ? 130 : 0;
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
    }
  }
}
