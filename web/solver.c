#include <stdint.h>
#include <wasm_simd128.h>

typedef struct { int32_t x, y, z, r, side; } Block;
static Block blocks[512];
static Block mblocks[4][512];
static int32_t results[32768];

int blocks_ptr(void) { return (int)(uintptr_t)blocks; }
int mblocks_ptr(void) { return (int)(uintptr_t)mblocks; }
int results_ptr(void) { return (int)(uintptr_t)results; }

static inline uint32_t rot_xz_y(int32_t x, uint64_t zy) {
  uint64_t l = (uint64_t)(int64_t)(int32_t)((uint32_t)x * 3129871u) ^ zy;
  l = l * l * 42317861ull + l * 11ull;
  // Logical shift is enough here: only low 48 bits matter after the Java LCG.
  uint64_t s = (l >> 16) ^ 0x5deece66dull;
  return (uint32_t)(((s * 0x5deece66dull + 11ull) >> 46) & 3);
}

static inline uint32_t rot(int32_t x, int32_t y, int32_t z) {
  return rot_xz_y(x, ((uint64_t)(int64_t)z * 116129781ull) ^ (uint64_t)(int64_t)y);
}

int rotation(int x, int y, int z) { return (int)rot(x, y, z); }

static inline v128_t vr_xp(v128_t xp, v128_t zy_lo4, v128_t zy_hi4) {
  const v128_t c42_32 = wasm_u32x4_splat(42317861u);
  const v128_t c11_32 = wasm_u32x4_splat(11u);
  const v128_t cdeec_32 = wasm_u32x4_splat(0xdeece66du);
  const v128_t c5deec_64 = wasm_u64x2_splat(0x5deece66dull);
  const v128_t c11_64 = wasm_u64x2_splat(11ull);
  const v128_t zero = wasm_i32x4_splat(0);

  v128_t lo4 = wasm_v128_xor(xp, zy_lo4);
  v128_t hi4 = wasm_v128_xor(wasm_i32x4_shr(xp, 31), zy_hi4);

  v128_t sq01 = wasm_u64x2_extmul_low_u32x4(lo4, lo4);
  v128_t sq23 = wasm_u64x2_extmul_high_u32x4(lo4, lo4);
  v128_t cross4 = wasm_i32x4_shl(wasm_i32x4_mul(lo4, hi4), 1);
  v128_t sq_lo4 = wasm_i32x4_shuffle(sq01, sq23, 0, 2, 4, 6);
  v128_t sq_hi4 = wasm_i32x4_add(wasm_i32x4_shuffle(sq01, sq23, 1, 3, 5, 7), cross4);

  v128_t p01 = wasm_i64x2_add(wasm_u64x2_extmul_low_u32x4(sq_lo4, c42_32),
                              wasm_u64x2_extmul_low_u32x4(lo4, c11_32));
  v128_t p23 = wasm_i64x2_add(wasm_u64x2_extmul_high_u32x4(sq_lo4, c42_32),
                              wasm_u64x2_extmul_high_u32x4(lo4, c11_32));
  v128_t hi_term = wasm_i32x4_add(wasm_i32x4_mul(sq_hi4, c42_32), wasm_i32x4_mul(hi4, c11_32));
  p01 = wasm_i64x2_add(p01, wasm_i32x4_shuffle(zero, hi_term, 0, 4, 1, 5));
  p23 = wasm_i64x2_add(p23, wasm_i32x4_shuffle(zero, hi_term, 2, 6, 3, 7));

  v128_t s01 = wasm_v128_xor(wasm_u64x2_shr(p01, 16), c5deec_64);
  v128_t s23 = wasm_v128_xor(wasm_u64x2_shr(p23, 16), c5deec_64);

  v128_t s_lo4 = wasm_i32x4_shuffle(s01, s23, 0, 2, 4, 6);
  v128_t s_hi4 = wasm_i32x4_shuffle(s01, s23, 1, 3, 5, 7);
  v128_t st01 = wasm_i64x2_add(wasm_u64x2_extmul_low_u32x4(s_lo4, cdeec_32), c11_64);
  v128_t st23 = wasm_i64x2_add(wasm_u64x2_extmul_high_u32x4(s_lo4, cdeec_32), c11_64);
  v128_t st_hi4 = wasm_i32x4_shuffle(st01, st23, 1, 3, 5, 7);
  v128_t s_lo5 = wasm_i32x4_add(wasm_i32x4_shl(s_lo4, 2), s_lo4);
  v128_t hh = wasm_i32x4_add(st_hi4, wasm_i32x4_add(wasm_i32x4_mul(s_hi4, cdeec_32), s_lo5));
  return wasm_v128_and(wasm_u32x4_shr(hh, 14), wasm_i32x4_splat(3));
}

int scan(int32_t x, int32_t y, int32_t z, uint32_t count, uint32_t n) {
  uint32_t out = 0;
  Block b0 = blocks[0];
  uint64_t z0 = ((uint64_t)(int64_t)(z + b0.z) * 116129781ull) ^ (uint64_t)(int64_t)(y + b0.y);
  v128_t zy_lo4 = wasm_u32x4_splat((uint32_t)z0);
  v128_t zy_hi4 = wasm_u32x4_splat((uint32_t)(z0 >> 32));
  v128_t mask0_vec = wasm_i32x4_splat(b0.side ? 1 : 3);
  v128_t target0_vec = wasm_i32x4_splat(b0.r);

  uint64_t q_zy[512];
  if (n > 1) {
    for (uint32_t j = 1; j < n; j++) {
      q_zy[j] = ((uint64_t)(int64_t)(z + blocks[j].z) * 116129781ull) ^ (uint64_t)(int64_t)(y + blocks[j].y);
    }
  }

  uint32_t base_xp = ((uint32_t)x + (uint32_t)b0.x) * 3129871u;
  v128_t xp0 = wasm_u32x4_make(base_xp, base_xp + 3129871u, base_xp + 6259742u, base_xp + 9389613u);
  v128_t xp1 = wasm_i32x4_add(xp0, wasm_u32x4_splat(12519484u));
  const v128_t step8 = wasm_u32x4_splat(25038968u);
  const v128_t step4 = wasm_u32x4_splat(12519484u);

  uint32_t i = 0;
  for (; i + 8 <= count; i += 8) {
    v128_t rA = vr_xp(xp0, zy_lo4, zy_hi4);
    v128_t rB = vr_xp(xp1, zy_lo4, zy_hi4);
    xp0 = wasm_i32x4_add(xp0, step8);
    xp1 = wasm_i32x4_add(xp1, step8);

    v128_t cmpA = wasm_i32x4_eq(wasm_v128_and(rA, mask0_vec), target0_vec);
    v128_t cmpB = wasm_i32x4_eq(wasm_v128_and(rB, mask0_vec), target0_vec);
    int mask = wasm_i32x4_bitmask(cmpA) | (wasm_i32x4_bitmask(cmpB) << 4);
    if (!mask) continue;
    for (; mask; mask &= mask - 1) {
      uint32_t lane = (uint32_t)__builtin_ctz(mask);
      int32_t ax = (int32_t)((uint32_t)x + i + lane);
      int ok = 1;
      for (uint32_t j = 1; j < n; j++) {
        Block q = blocks[j];
        if ((rot_xz_y(ax + q.x, q_zy[j]) & (q.side ? 1 : 3)) != (uint32_t)q.r) { ok = 0; break; }
      }
      if (ok) results[out++] = ax;
    }
  }
  for (; i < count; i += 4) {
    v128_t rA = vr_xp(xp0, zy_lo4, zy_hi4);
    xp0 = wasm_i32x4_add(xp0, step4);
    v128_t cmpA = wasm_i32x4_eq(wasm_v128_and(rA, mask0_vec), target0_vec);
    int mask = wasm_i32x4_bitmask(cmpA);
    if (!mask) continue;
    for (; mask; mask &= mask - 1) {
      uint32_t lane = (uint32_t)__builtin_ctz(mask);
      if (i + lane >= count) break;
      int32_t ax = (int32_t)((uint32_t)x + i + lane);
      int ok = 1;
      for (uint32_t j = 1; j < n; j++) {
        Block q = blocks[j];
        if ((rot_xz_y(ax + q.x, q_zy[j]) & (q.side ? 1 : 3)) != (uint32_t)q.r) { ok = 0; break; }
      }
      if (ok) results[out++] = ax;
    }
  }
  return (int)out;
}

int scan_multi(int32_t xmin, uint32_t xcount, int32_t ymin, int32_t ymax,
               int32_t zmin, int32_t zmax, uint32_t nb, uint32_t active_mask,
               int32_t anchor_r, int32_t anchor_side, uint32_t max_matches) {
  if (max_matches > 8192u) max_matches = 8192u;
  uint32_t out = 0;
  const v128_t step8 = wasm_u32x4_splat(25038968u);
  const v128_t step4 = wasm_u32x4_splat(12519484u);
  const v128_t anchor_vec = wasm_i32x4_splat(anchor_r);
  const v128_t mask3 = wasm_i32x4_splat(3);
  const uint32_t base_xp_init = (uint32_t)xmin * 3129871u;
  const v128_t xp0_init = wasm_u32x4_make(base_xp_init, base_xp_init + 3129871u,
                                          base_xp_init + 6259742u, base_xp_init + 9389613u);
  const v128_t xp1_init = wasm_i32x4_add(xp0_init, step4);

  const v128_t lut_top = wasm_i8x16_make(
    (active_mask & 1u) ? -128 : 0, (active_mask & 2u) ? -128 : 0,
    (active_mask & 4u) ? -128 : 0, (active_mask & 8u) ? -128 : 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  );
  const uint32_t side_fold0 = (active_mask & 5u) ? 1u : 0u;
  const uint32_t side_fold1 = (active_mask & 10u) ? 1u : 0u;
  const v128_t lut_side = wasm_i8x16_make(
    side_fold0 ? -128 : 0, side_fold1 ? -128 : 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  );
  const v128_t byte3_mul = wasm_u32x4_splat(0x01000000u);
  const v128_t gather_mul = wasm_u32x4_splat(0x04040404u);
  const v128_t gather_add = wasm_u32x4_splat(0x03020100u);

  v128_t b1_xp_delta = wasm_i32x4_splat(0);
  v128_t b1_target = wasm_i32x4_splat(-1);
  v128_t b1_mask_vec = wasm_i32x4_splat(3);
  if (!anchor_side && nb > 1u) {
    b1_xp_delta = wasm_u32x4_make(
      (uint32_t)mblocks[0][1].x * 3129871u,
      (uint32_t)mblocks[1][1].x * 3129871u,
      (uint32_t)mblocks[2][1].x * 3129871u,
      (uint32_t)mblocks[3][1].x * 3129871u
    );
    b1_target = wasm_i32x4_make(
      (active_mask & 1u) ? mblocks[0][1].r : -1,
      (active_mask & 2u) ? mblocks[1][1].r : -1,
      (active_mask & 4u) ? mblocks[2][1].r : -1,
      (active_mask & 8u) ? mblocks[3][1].r : -1
    );
    b1_mask_vec = wasm_i32x4_splat(mblocks[0][1].side ? 1 : 3);
  }

  uint64_t z_part[4][512];
  for (int32_t z = zmin; z <= zmax; z++) {
    uint64_t zp0 = (uint64_t)(int64_t)z * 116129781ull;
    if (nb > 1u) {
      for (uint32_t dir = 0; dir < 4u; dir++) {
        if (!(active_mask & (1u << dir))) continue;
        for (uint32_t j = 1u; j < nb; j++) {
          z_part[dir][j] = (uint64_t)(int64_t)(z + mblocks[dir][j].z) * 116129781ull;
        }
      }
    }
    for (int32_t y = ymin; y <= ymax; y++) {
      uint64_t z0 = zp0 ^ (uint64_t)(int64_t)y;
      v128_t zy_lo4 = wasm_u32x4_splat((uint32_t)z0);
      v128_t zy_hi4 = wasm_u32x4_splat((uint32_t)(z0 >> 32));
      v128_t xp0 = xp0_init;
      v128_t xp1 = xp1_init;

      v128_t b1_zy_lo_tbl = wasm_i32x4_splat(0);
      v128_t b1_zy_hi_tbl = wasm_i32x4_splat(0);
      if (!anchor_side && nb > 1u) {
        uint64_t zb1_0 = (active_mask & 1u) ? (z_part[0][1] ^ (uint64_t)(int64_t)(y + mblocks[0][1].y)) : 0ull;
        uint64_t zb1_1 = (active_mask & 2u) ? (z_part[1][1] ^ (uint64_t)(int64_t)(y + mblocks[1][1].y)) : 0ull;
        uint64_t zb1_2 = (active_mask & 4u) ? (z_part[2][1] ^ (uint64_t)(int64_t)(y + mblocks[2][1].y)) : 0ull;
        uint64_t zb1_3 = (active_mask & 8u) ? (z_part[3][1] ^ (uint64_t)(int64_t)(y + mblocks[3][1].y)) : 0ull;
        b1_zy_lo_tbl = wasm_u32x4_make((uint32_t)zb1_0, (uint32_t)zb1_1, (uint32_t)zb1_2, (uint32_t)zb1_3);
        b1_zy_hi_tbl = wasm_u32x4_make((uint32_t)(zb1_0 >> 32), (uint32_t)(zb1_1 >> 32),
                                       (uint32_t)(zb1_2 >> 32), (uint32_t)(zb1_3 >> 32));
      }

      uint32_t i = 0;
      if (!anchor_side && nb > 1u) {
        for (; i + 8 <= xcount; i += 8) {
          v128_t cur_xp0 = xp0;
          v128_t cur_xp1 = xp1;
          xp0 = wasm_i32x4_add(xp0, step8);
          xp1 = wasm_i32x4_add(xp1, step8);

          v128_t rA = vr_xp(cur_xp0, zy_lo4, zy_hi4);
          v128_t rB = vr_xp(cur_xp1, zy_lo4, zy_hi4);
          v128_t nA = wasm_v128_and(wasm_i32x4_sub(rA, anchor_vec), mask3);
          v128_t nB = wasm_v128_and(wasm_i32x4_sub(rB, anchor_vec), mask3);

          int mask = 0;
          if (active_mask == 15u) {
            v128_t idxA = wasm_i32x4_add(wasm_i32x4_mul(nA, gather_mul), gather_add);
            v128_t idxB = wasm_i32x4_add(wasm_i32x4_mul(nB, gather_mul), gather_add);
            v128_t r1A = vr_xp(wasm_i32x4_add(cur_xp0, wasm_i8x16_swizzle(b1_xp_delta, idxA)),
                               wasm_i8x16_swizzle(b1_zy_lo_tbl, idxA),
                               wasm_i8x16_swizzle(b1_zy_hi_tbl, idxA));
            v128_t r1B = vr_xp(wasm_i32x4_add(cur_xp1, wasm_i8x16_swizzle(b1_xp_delta, idxB)),
                               wasm_i8x16_swizzle(b1_zy_lo_tbl, idxB),
                               wasm_i8x16_swizzle(b1_zy_hi_tbl, idxB));
            v128_t cmp1A = wasm_i32x4_eq(wasm_v128_and(r1A, b1_mask_vec), wasm_i8x16_swizzle(b1_target, idxA));
            v128_t cmp1B = wasm_i32x4_eq(wasm_v128_and(r1B, b1_mask_vec), wasm_i8x16_swizzle(b1_target, idxB));
            mask = wasm_i32x4_bitmask(cmp1A) | (wasm_i32x4_bitmask(cmp1B) << 4);
          } else {
            int m0A = wasm_i32x4_bitmask(wasm_i8x16_swizzle(lut_top, wasm_i32x4_mul(nA, byte3_mul)));
            int m0B = wasm_i32x4_bitmask(wasm_i8x16_swizzle(lut_top, wasm_i32x4_mul(nB, byte3_mul)));
            if (!(m0A | m0B)) continue;
            if (m0A) {
              v128_t idxA = wasm_i32x4_add(wasm_i32x4_mul(nA, gather_mul), gather_add);
              v128_t r1A = vr_xp(wasm_i32x4_add(cur_xp0, wasm_i8x16_swizzle(b1_xp_delta, idxA)),
                                 wasm_i8x16_swizzle(b1_zy_lo_tbl, idxA),
                                 wasm_i8x16_swizzle(b1_zy_hi_tbl, idxA));
              v128_t cmp1A = wasm_i32x4_eq(wasm_v128_and(r1A, b1_mask_vec), wasm_i8x16_swizzle(b1_target, idxA));
              mask |= wasm_i32x4_bitmask(cmp1A);
            }
            if (m0B) {
              v128_t idxB = wasm_i32x4_add(wasm_i32x4_mul(nB, gather_mul), gather_add);
              v128_t r1B = vr_xp(wasm_i32x4_add(cur_xp1, wasm_i8x16_swizzle(b1_xp_delta, idxB)),
                                 wasm_i8x16_swizzle(b1_zy_lo_tbl, idxB),
                                 wasm_i8x16_swizzle(b1_zy_hi_tbl, idxB));
              v128_t cmp1B = wasm_i32x4_eq(wasm_v128_and(r1B, b1_mask_vec), wasm_i8x16_swizzle(b1_target, idxB));
              mask |= wasm_i32x4_bitmask(cmp1B) << 4;
            }
          }
          if (!mask) continue;

          uint32_t ns[8];
          wasm_v128_store(ns, nA);
          wasm_v128_store(ns + 4, nB);
          for (; mask; mask &= mask - 1) {
            uint32_t lane = (uint32_t)__builtin_ctz(mask);
            uint32_t dir = ns[lane];
            int32_t ax = (int32_t)((uint32_t)xmin + i + lane);
            int ok = 1;
            for (uint32_t j = 2u; j < nb; j++) {
              Block q = mblocks[dir][j];
              uint64_t zy_j = z_part[dir][j] ^ (uint64_t)(int64_t)(y + q.y);
              if ((rot_xz_y(ax + q.x, zy_j) & (q.side ? 1u : 3u)) != (uint32_t)q.r) { ok = 0; break; }
            }
            if (ok) {
              if (out >= max_matches) return -1;
              int32_t* dst = &results[out * 4u];
              dst[0] = ax; dst[1] = y; dst[2] = z; dst[3] = (int32_t)dir;
              out++;
            }
          }
        }
      }
      for (; i < xcount; i += 4) {
        v128_t rA = vr_xp(xp0, zy_lo4, zy_hi4);
        xp0 = wasm_i32x4_add(xp0, step4);
        if (!anchor_side) {
          v128_t nA = wasm_v128_and(wasm_i32x4_sub(rA, anchor_vec), mask3);
          int mask = 0xF;
          if (active_mask != 15u) {
            v128_t hitA = wasm_i8x16_swizzle(lut_top, wasm_i32x4_mul(nA, byte3_mul));
            mask = wasm_i32x4_bitmask(hitA);
            if (!mask) continue;
          }
          uint32_t ns[4];
          wasm_v128_store(ns, nA);
          for (; mask; mask &= mask - 1) {
            uint32_t lane = (uint32_t)__builtin_ctz(mask);
            if (i + lane >= xcount) break;
            uint32_t dir = ns[lane];
            int32_t ax = (int32_t)((uint32_t)xmin + i + lane);
            int ok = 1;
            for (uint32_t j = 1u; j < nb; j++) {
              Block q = mblocks[dir][j];
              uint64_t zy_j = z_part[dir][j] ^ (uint64_t)(int64_t)(y + q.y);
              if ((rot_xz_y(ax + q.x, zy_j) & (q.side ? 1u : 3u)) != (uint32_t)q.r) { ok = 0; break; }
            }
            if (ok) {
              if (out >= max_matches) return -1;
              int32_t* dst = &results[out * 4u];
              dst[0] = ax; dst[1] = y; dst[2] = z; dst[3] = (int32_t)dir;
              out++;
            }
          }
        } else {
          v128_t pA = wasm_v128_and(wasm_i32x4_sub(rA, anchor_vec), wasm_i32x4_splat(1));
          v128_t hitA = wasm_i8x16_swizzle(lut_side, wasm_i32x4_mul(pA, byte3_mul));
          int mask = wasm_i32x4_bitmask(hitA);
          if (!mask) continue;
          uint32_t ps[4];
          wasm_v128_store(ps, pA);
          for (; mask; mask &= mask - 1) {
            uint32_t lane = (uint32_t)__builtin_ctz(mask);
            if (i + lane >= xcount) break;
            int32_t ax = (int32_t)((uint32_t)xmin + i + lane);
            for (uint32_t dir = ps[lane]; dir < 4u; dir += 2u) {
              if (!(active_mask & (1u << dir))) continue;
              int ok = 1;
              for (uint32_t j = 1u; j < nb; j++) {
                Block q = mblocks[dir][j];
                uint64_t zy_j = z_part[dir][j] ^ (uint64_t)(int64_t)(y + q.y);
                if ((rot_xz_y(ax + q.x, zy_j) & (q.side ? 1u : 3u)) != (uint32_t)q.r) { ok = 0; break; }
              }
              if (ok) {
                if (out >= max_matches) return -1;
                int32_t* dst = &results[out * 4u];
                dst[0] = ax; dst[1] = y; dst[2] = z; dst[3] = (int32_t)dir;
                out++;
              }
            }
          }
        }
      }
    }
  }
  return (int)out;
}
