/* Freestanding browser detector: fast 4-wide f32x4 perspective bilerp & quadrant correlation. */
#include <wasm_simd128.h>
typedef unsigned char u8;
typedef unsigned short u16_unaligned __attribute__((aligned(1)));

static float refs_qc[256][4] __attribute__((aligned(16)));
static double q_fsum_div_49[4][4] __attribute__((aligned(16)));
static double sb_div_196[4] __attribute__((aligned(16)));
static double inv_sqrt_vb[4] __attribute__((aligned(16)));
static double q_inv_sqrt_vb[4][4] __attribute__((aligned(16)));
static double q_b_diff[4][4] __attribute__((aligned(16)));

void set_refs(const u8* r) {
  double raw[256][4];
  double sb[4] = {0, 0, 0, 0}, bb[4] = {0, 0, 0, 0};
  double q_sb[4][4] = {{0}}, q_bb[4][4] = {{0}};

  for (int i = 0; i < 256; i++) {
    for (int k = 0; k < 4; k++)
      raw[i][k] = (double)r[k * 256 + i];
  }
  for (int j = 1; j < 15; j++) {
    for (int i = 1; i < 15; i++) {
      int q = (i >= 8 ? 1 : 0) + (j >= 8 ? 2 : 0);
      int idx = j * 16 + i;
      for (int k = 0; k < 4; k++) {
        double b = raw[idx][k];
        sb[k] += b;
        bb[k] += b * b;
        q_sb[q][k] += b;
        q_bb[q][k] += b * b;
      }
    }
  }
  for (int k = 0; k < 4; k++) {
    double vb = bb[k] - sb[k] * sb[k] / 196.0;
    inv_sqrt_vb[k] = vb < 1e-9 ? 0.0 : 1.0 / __builtin_sqrt(vb);
    sb_div_196[k] = sb[k] / 196.0;
  }
  for (int q = 0; q < 4; q++) {
    double fsum[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; k++) {
      double qvb = q_bb[q][k] - q_sb[q][k] * q_sb[q][k] / 49.0;
      q_inv_sqrt_vb[q][k] = qvb < 1e-9 ? 0.0 : 1.0 / __builtin_sqrt(qvb);
      double qmean = q_sb[q][k] / 49.0;
      q_b_diff[q][k] = qmean - sb_div_196[k];
    }
    int j0 = (q >= 2) ? 8 : 1, i0 = (q & 1) ? 8 : 1;
    for (int j = j0; j < j0 + 7; j++) {
      for (int i = i0; i < i0 + 7; i++) {
        int idx = j * 16 + i;
        for (int k = 0; k < 4; k++) {
          float c = (float)(raw[idx][k] - q_sb[q][k] / 49.0);
          refs_qc[idx][k] = c;
          fsum[k] += (double)c;
        }
      }
    }
    for (int k = 0; k < 4; k++)
      q_fsum_div_49[q][k] = fsum[k] / 49.0;
  }
}

static inline int fit(const float* q, double* H) {
  double x0 = q[0], y0 = q[1], x1 = q[2], y1 = q[3], x2 = q[4], y2 = q[5], x3 = q[6], y3 = q[7];
  double dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  double dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  double det = dx1 * dy2 - dx2 * dy1;
  if (__builtin_fabs(det) < 1e-12) return 0;
  double inv_16_det = 0.0625 / det;
  double g = (dx3 * dy2 - dx2 * dy3) * inv_16_det;
  double h = (dx1 * dy3 - dx3 * dy1) * inv_16_det;
  H[0] = (x1 - x0) * 0.0625 + g * x1;
  H[1] = (x3 - x0) * 0.0625 + h * x3;
  H[2] = x0;
  H[3] = (y1 - y0) * 0.0625 + g * y1;
  H[4] = (y3 - y0) * 0.0625 + h * y3;
  H[5] = y0;
  H[6] = g;
  H[7] = h;
  H[8] = 1.0;
  return 1;
}

static inline v128_t bilerp4(const u8* g, int w, v128_t px, v128_t py) {
  v128_t flx = wasm_f32x4_floor(px);
  v128_t fly = wasm_f32x4_floor(py);
  v128_t vfx = wasm_f32x4_sub(px, flx);
  v128_t vfy = wasm_f32x4_sub(py, fly);

  v128_t vix = wasm_i32x4_trunc_sat_f32x4(flx);
  v128_t viy = wasm_i32x4_trunc_sat_f32x4(fly);
  v128_t voff = wasm_i32x4_add(wasm_i32x4_mul(viy, wasm_i32x4_splat(w)), vix);

  int o0 = wasm_i32x4_extract_lane(voff, 0);
  int o1 = wasm_i32x4_extract_lane(voff, 1);
  int o2 = wasm_i32x4_extract_lane(voff, 2);
  int o3 = wasm_i32x4_extract_lane(voff, 3);

  v128_t t16 = wasm_i32x4_make(*(const u16_unaligned*)(g + o0), *(const u16_unaligned*)(g + o1),
                               *(const u16_unaligned*)(g + o2), *(const u16_unaligned*)(g + o3));
  v128_t b16 = wasm_i32x4_make(*(const u16_unaligned*)(g + o0 + w), *(const u16_unaligned*)(g + o1 + w),
                               *(const u16_unaligned*)(g + o2 + w), *(const u16_unaligned*)(g + o3 + w));

  const v128_t m255 = wasm_i32x4_splat(0xff);
  v128_t g00 = wasm_f32x4_convert_i32x4(wasm_v128_and(t16, m255));
  v128_t g10 = wasm_f32x4_convert_i32x4(wasm_u32x4_shr(t16, 8));
  v128_t g01 = wasm_f32x4_convert_i32x4(wasm_v128_and(b16, m255));
  v128_t g11 = wasm_f32x4_convert_i32x4(wasm_u32x4_shr(b16, 8));

  v128_t top = wasm_f32x4_add(g00, wasm_f32x4_mul(vfx, wasm_f32x4_sub(g10, g00)));
  v128_t bot = wasm_f32x4_add(g01, wasm_f32x4_mul(vfx, wasm_f32x4_sub(g11, g01)));
  return wasm_f32x4_add(top, wasm_f32x4_mul(vfy, wasm_f32x4_sub(bot, top)));
}

static inline v128_t sample4_f32(const u8* g, int w, v128_t vnx, v128_t vny, v128_t vnd) {
  v128_t inv = wasm_f32x4_div(wasm_f32x4_splat(1.0f), vnd);
  return bilerp4(g, w, wasm_f32x4_mul(vnx, inv), wasm_f32x4_mul(vny, inv));
}

static inline double hsum_f32x4(v128_t v) {
  v128_t d0 = wasm_f64x2_promote_low_f32x4(v);
  v128_t d1 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
  v128_t s = wasm_f64x2_add(d0, d1);
  return wasm_f64x2_extract_lane(s, 0) + wasm_f64x2_extract_lane(s, 1);
}

static inline void accum_row_cov(int row_off, v128_t u1_4, v128_t u5_8, v128_t u9_12, v128_t u13_14_pair,
                                 int pair_hi, v128_t* covL, v128_t* covR, v128_t* suL, v128_t* uuL,
                                 v128_t* suR, v128_t* uuR) {
  const v128_t mask012 = wasm_i32x4_make(-1, -1, -1, 0);
  const v128_t mask3 = wasm_i32x4_make(0, 0, 0, -1);

  v128_t rL = wasm_f32x4_add(
      wasm_f32x4_add(
          wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 0, 0, 0, 0), wasm_v128_load(refs_qc[row_off + 1])),
          wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 1, 1, 1, 1), wasm_v128_load(refs_qc[row_off + 2]))),
      wasm_f32x4_add(
          wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 2, 2, 2, 2), wasm_v128_load(refs_qc[row_off + 3])),
          wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 3, 3, 3, 3), wasm_v128_load(refs_qc[row_off + 4]))));
  rL = wasm_f32x4_add(rL,
                      wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 0, 0, 0, 0),
                                                                   wasm_v128_load(refs_qc[row_off + 5])),
                                                    wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 1, 1, 1, 1),
                                                                   wasm_v128_load(refs_qc[row_off + 6]))),
                                     wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 2, 2, 2, 2),
                                                    wasm_v128_load(refs_qc[row_off + 7]))));
  *covL = wasm_f32x4_add(*covL, rL);

  v128_t u13 = pair_hi ? wasm_i32x4_shuffle(u13_14_pair, u13_14_pair, 2, 2, 2, 2)
                       : wasm_i32x4_shuffle(u13_14_pair, u13_14_pair, 0, 0, 0, 0);
  v128_t u14 = pair_hi ? wasm_i32x4_shuffle(u13_14_pair, u13_14_pair, 3, 3, 3, 3)
                       : wasm_i32x4_shuffle(u13_14_pair, u13_14_pair, 1, 1, 1, 1);

  v128_t rR = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 0, 0, 0, 0),
                                                           wasm_v128_load(refs_qc[row_off + 9])),
                                            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 1, 1, 1, 1),
                                                           wasm_v128_load(refs_qc[row_off + 10]))),
                             wasm_f32x4_add(wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 2, 2, 2, 2),
                                                           wasm_v128_load(refs_qc[row_off + 11])),
                                            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 3, 3, 3, 3),
                                                           wasm_v128_load(refs_qc[row_off + 12]))));
  rR = wasm_f32x4_add(
      rR, wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 3, 3, 3, 3),
                                                       wasm_v128_load(refs_qc[row_off + 8])),
                                        wasm_f32x4_mul(u13, wasm_v128_load(refs_qc[row_off + 13]))),
                         wasm_f32x4_mul(u14, wasm_v128_load(refs_qc[row_off + 14]))));
  *covR = wasm_f32x4_add(*covR, rR);

  v128_t u5_7 = wasm_v128_and(u5_8, mask012);
  *suL = wasm_f32x4_add(*suL, wasm_f32x4_add(u1_4, u5_7));
  *uuL = wasm_f32x4_add(*uuL, wasm_f32x4_add(wasm_f32x4_mul(u1_4, u1_4), wasm_f32x4_mul(u5_7, u5_7)));

  v128_t u8_13_14 = pair_hi ? wasm_i32x4_shuffle(wasm_v128_and(u5_8, mask3), u13_14_pair, 3, 6, 7, 0)
                            : wasm_i32x4_shuffle(wasm_v128_and(u5_8, mask3), u13_14_pair, 3, 4, 5, 0);
  *suR = wasm_f32x4_add(*suR, wasm_f32x4_add(u9_12, u8_13_14));
  *uuR =
      wasm_f32x4_add(*uuR, wasm_f32x4_add(wasm_f32x4_mul(u9_12, u9_12), wasm_f32x4_mul(u8_13_14, u8_13_14)));
}

void classify_batch(const u8* g, int w, int h, const float* quads, int count, float* out) {
  const v128_t f128 = wasm_f32x4_splat(128.0f);
  const v128_t fzero = wasm_f32x4_splat(0.0f);
  const v128_t dzero = wasm_f64x2_splat(0.0);
  const double wm1 = (double)(w - 1), hm1 = (double)(h - 1);
  const double wm2 = (double)(w - 2), hm2 = (double)(h - 2);

  for (int t = 0; t < count; t++) {
    float* o = out + t * 5;
    double H[9];
    if (!fit(quads + t * 8, H)) {
      o[0] = 0;
      o[1] = 0;
      o[2] = 0;
      o[3] = 0;
      o[4] = 0;
      continue;
    }

    static const double cx[4] = {1.5, 14.5, 1.5, 14.5}, cy[4] = {1.5, 1.5, 14.5, 14.5};
    int fast_bounds = 1, reject = 0;
    for (int c = 0; c < 4; c++) {
      double cd = H[6] * cx[c] + H[7] * cy[c] + 1.0;
      if (cd == 0.0) {
        fast_bounds = 0;
        break;
      }
      double inv_cd = 1.0 / cd;
      double cpx = (H[0] * cx[c] + H[1] * cy[c] + H[2]) * inv_cd;
      double cpy = (H[3] * cx[c] + H[4] * cy[c] + H[5]) * inv_cd;
      if (cpx < -1e-7 || cpy < -1e-7 || cpx > wm1 + 1e-7 || cpy > hm1 + 1e-7) {
        reject = 1;
        break;
      }
      if (!(cd > 0.0 && cpx >= 1.0 && cpy >= 1.0 && cpx < wm2 && cpy < hm2)) {
        fast_bounds = 0;
      }
    }
    if (reject) {
      o[0] = 0;
      o[1] = 0;
      o[2] = 0;
      o[3] = 0;
      o[4] = 0;
      continue;
    }

    double q_su[4] = {0, 0, 0, 0};
    double q_uu[4] = {0, 0, 0, 0};
    v128_t q_cov0[4] = {dzero, dzero, dzero, dzero};
    v128_t q_cov1[4] = {dzero, dzero, dzero, dzero};

    if (fast_bounds) {
      const float h0 = (float)H[0], h1 = (float)H[1], h2 = (float)H[2];
      const float h3 = (float)H[3], h4 = (float)H[4], h5 = (float)H[5];
      const float h6 = (float)H[6], h7 = (float)H[7];
      const int is_affine = (__builtin_fabs(H[6]) + __builtin_fabs(H[7]) < 1e-10);

      const v128_t off4 = wasm_f32x4_make(0.0f, 1.0f, 2.0f, 3.0f);
      const v128_t vx_off = wasm_f32x4_mul(wasm_f32x4_splat(h0), off4);
      const v128_t vy_off = wasm_f32x4_mul(wasm_f32x4_splat(h3), off4);
      const v128_t vd_off = wasm_f32x4_mul(wasm_f32x4_splat(h6), off4);
      const v128_t step4x = wasm_f32x4_splat(4.0f * h0);
      const v128_t step4y = wasm_f32x4_splat(4.0f * h3);
      const v128_t step4d = wasm_f32x4_splat(4.0f * h6);

      v128_t qcov[4] = {fzero, fzero, fzero, fzero};
      v128_t vsu[4] = {fzero, fzero, fzero, fzero};
      v128_t vuu[4] = {fzero, fzero, fzero, fzero};

      /* Pair rows (j, j+1) so cols 13..14 of both rows share one 4-wide sample call (49 calls/quad). */
      for (int j = 1; j < 15; j += 2) {
        const float vj0 = (float)j + 0.5f;
        const float vj1 = vj0 + 1.0f;
        const int qr0 = (j >= 8) ? 2 : 0;
        const int qr1 = ((j + 1) >= 8) ? 2 : 0;

        float bx0 = h0 * 1.5f + h1 * vj0 + h2;
        float by0 = h3 * 1.5f + h4 * vj0 + h5;
        float bx1 = bx0 + h1;
        float by1 = by0 + h4;

        v128_t u0_1, u0_5, u0_9, u1_1, u1_5, u1_9, u_tail;

        if (is_affine) {
          v128_t px0 = wasm_f32x4_add(wasm_f32x4_splat(bx0), vx_off);
          v128_t py0 = wasm_f32x4_add(wasm_f32x4_splat(by0), vy_off);
          u0_1 = wasm_f32x4_sub(bilerp4(g, w, px0, py0), f128);
          px0 = wasm_f32x4_add(px0, step4x);
          py0 = wasm_f32x4_add(py0, step4y);
          u0_5 = wasm_f32x4_sub(bilerp4(g, w, px0, py0), f128);
          px0 = wasm_f32x4_add(px0, step4x);
          py0 = wasm_f32x4_add(py0, step4y);
          u0_9 = wasm_f32x4_sub(bilerp4(g, w, px0, py0), f128);

          v128_t px1 = wasm_f32x4_add(wasm_f32x4_splat(bx1), vx_off);
          v128_t py1 = wasm_f32x4_add(wasm_f32x4_splat(by1), vy_off);
          u1_1 = wasm_f32x4_sub(bilerp4(g, w, px1, py1), f128);
          px1 = wasm_f32x4_add(px1, step4x);
          py1 = wasm_f32x4_add(py1, step4y);
          u1_5 = wasm_f32x4_sub(bilerp4(g, w, px1, py1), f128);
          px1 = wasm_f32x4_add(px1, step4x);
          py1 = wasm_f32x4_add(py1, step4y);
          u1_9 = wasm_f32x4_sub(bilerp4(g, w, px1, py1), f128);

          v128_t pxt =
              wasm_i32x4_shuffle(wasm_f32x4_add(px0, step4x), wasm_f32x4_add(px1, step4x), 0, 1, 4, 5);
          v128_t pyt =
              wasm_i32x4_shuffle(wasm_f32x4_add(py0, step4y), wasm_f32x4_add(py1, step4y), 0, 1, 4, 5);
          u_tail = wasm_f32x4_sub(bilerp4(g, w, pxt, pyt), f128);
        } else {
          float bd0 = h6 * 1.5f + h7 * vj0 + 1.0f;
          float bd1 = bd0 + h7;

          v128_t nx0 = wasm_f32x4_add(wasm_f32x4_splat(bx0), vx_off);
          v128_t ny0 = wasm_f32x4_add(wasm_f32x4_splat(by0), vy_off);
          v128_t nd0 = wasm_f32x4_add(wasm_f32x4_splat(bd0), vd_off);
          u0_1 = wasm_f32x4_sub(sample4_f32(g, w, nx0, ny0, nd0), f128);
          nx0 = wasm_f32x4_add(nx0, step4x);
          ny0 = wasm_f32x4_add(ny0, step4y);
          nd0 = wasm_f32x4_add(nd0, step4d);
          u0_5 = wasm_f32x4_sub(sample4_f32(g, w, nx0, ny0, nd0), f128);
          nx0 = wasm_f32x4_add(nx0, step4x);
          ny0 = wasm_f32x4_add(ny0, step4y);
          nd0 = wasm_f32x4_add(nd0, step4d);
          u0_9 = wasm_f32x4_sub(sample4_f32(g, w, nx0, ny0, nd0), f128);

          v128_t nx1 = wasm_f32x4_add(wasm_f32x4_splat(bx1), vx_off);
          v128_t ny1 = wasm_f32x4_add(wasm_f32x4_splat(by1), vy_off);
          v128_t nd1 = wasm_f32x4_add(wasm_f32x4_splat(bd1), vd_off);
          u1_1 = wasm_f32x4_sub(sample4_f32(g, w, nx1, ny1, nd1), f128);
          nx1 = wasm_f32x4_add(nx1, step4x);
          ny1 = wasm_f32x4_add(ny1, step4y);
          nd1 = wasm_f32x4_add(nd1, step4d);
          u1_5 = wasm_f32x4_sub(sample4_f32(g, w, nx1, ny1, nd1), f128);
          nx1 = wasm_f32x4_add(nx1, step4x);
          ny1 = wasm_f32x4_add(ny1, step4y);
          nd1 = wasm_f32x4_add(nd1, step4d);
          u1_9 = wasm_f32x4_sub(sample4_f32(g, w, nx1, ny1, nd1), f128);

          v128_t nxt =
              wasm_i32x4_shuffle(wasm_f32x4_add(nx0, step4x), wasm_f32x4_add(nx1, step4x), 0, 1, 4, 5);
          v128_t nyt =
              wasm_i32x4_shuffle(wasm_f32x4_add(ny0, step4y), wasm_f32x4_add(ny1, step4y), 0, 1, 4, 5);
          v128_t ndt =
              wasm_i32x4_shuffle(wasm_f32x4_add(nd0, step4d), wasm_f32x4_add(nd1, step4d), 0, 1, 4, 5);
          u_tail = wasm_f32x4_sub(sample4_f32(g, w, nxt, nyt, ndt), f128);
        }

        accum_row_cov(j * 16, u0_1, u0_5, u0_9, u_tail, 0, &qcov[qr0], &qcov[qr0 + 1], &vsu[qr0], &vuu[qr0],
                      &vsu[qr0 + 1], &vuu[qr0 + 1]);
        accum_row_cov((j + 1) * 16, u1_1, u1_5, u1_9, u_tail, 1, &qcov[qr1], &qcov[qr1 + 1], &vsu[qr1],
                      &vuu[qr1], &vsu[qr1 + 1], &vuu[qr1 + 1]);
      }

      for (int q = 0; q < 4; q++) {
        q_cov0[q] = wasm_f64x2_promote_low_f32x4(qcov[q]);
        q_cov1[q] = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(qcov[q], qcov[q], 2, 3, 0, 1));
        q_su[q] = hsum_f32x4(vsu[q]);
        q_uu[q] = hsum_f32x4(vuu[q]);
      }
    } else {
      int valid = 1;
      for (int j = 1; j < 15 && valid; j++) {
        int qrow = (j >= 8) ? 2 : 0;
        double vj = j + 0.5;
        double nx = H[0] * 1.5 + H[1] * vj + H[2];
        double ny = H[3] * 1.5 + H[4] * vj + H[5];
        double nd = H[6] * 1.5 + H[7] * vj + 1.0;
        int row_off = j * 16;
        for (int i = 1; i < 15; i++) {
          double invd = 1.0 / nd;
          double px = nx * invd, py = ny * invd;
          if (!(px >= 0.0 && py >= 0.0 && px < wm1 && py < hm1)) {
            valid = 0;
            break;
          }
          int xx = (int)px, yy = (int)py;
          double fx = px - xx, fy = py - yy;
          const u8* p = g + yy * w + xx;
          double top = p[0] + fx * (p[1] - p[0]);
          double bot = p[w] + fx * (p[w + 1] - p[w]);
          float u = (float)(top + fy * (bot - top)) - 128.0f;
          double du = (double)u;
          int q = qrow + (i >= 8 ? 1 : 0);
          q_su[q] += du;
          q_uu[q] += du * du;
          v128_t c4 = wasm_f32x4_mul(wasm_f32x4_splat(u), wasm_v128_load(refs_qc[row_off + i]));
          q_cov0[q] = wasm_f64x2_add(q_cov0[q], wasm_f64x2_promote_low_f32x4(c4));
          q_cov1[q] =
              wasm_f64x2_add(q_cov1[q], wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(c4, c4, 2, 3, 0, 1)));
          nx += H[0];
          ny += H[3];
          nd += H[6];
        }
      }
      if (!valid) {
        o[0] = 0;
        o[1] = 0;
        o[2] = 0;
        o[3] = 0;
        o[4] = 0;
        continue;
      }
    }

    double su = q_su[0] + q_su[1] + q_su[2] + q_su[3];
    double uu = q_uu[0] + q_uu[1] + q_uu[2] + q_uu[3];
    double va = uu - su * su / 196.0;
    double inv_sqrt_va = va < 1e-9 ? 0.0 : 1.0 / __builtin_sqrt(va);

    v128_t fcov0 = dzero, fcov1 = dzero;
    for (int q = 0; q < 4; q++) {
      v128_t vsuq = wasm_f64x2_splat(q_su[q]);
      v128_t vsaq = wasm_f64x2_splat(q_su[q] + 49.0 * 128.0);
      v128_t cq0 = wasm_f64x2_sub(q_cov0[q], wasm_f64x2_mul(vsuq, wasm_v128_load(q_fsum_div_49[q])));
      v128_t cq1 = wasm_f64x2_sub(q_cov1[q], wasm_f64x2_mul(vsuq, wasm_v128_load(q_fsum_div_49[q] + 2)));
      q_cov0[q] = cq0;
      q_cov1[q] = cq1;
      fcov0 = wasm_f64x2_add(fcov0, wasm_f64x2_add(cq0, wasm_f64x2_mul(vsaq, wasm_v128_load(q_b_diff[q]))));
      fcov1 =
          wasm_f64x2_add(fcov1, wasm_f64x2_add(cq1, wasm_f64x2_mul(vsaq, wasm_v128_load(q_b_diff[q] + 2))));
    }

    v128_t v_iva = wasm_f64x2_splat(inv_sqrt_va);
    v128_t sc0 = wasm_f64x2_mul(fcov0, wasm_f64x2_mul(v_iva, wasm_v128_load(inv_sqrt_vb)));
    v128_t sc1 = wasm_f64x2_mul(fcov1, wasm_f64x2_mul(v_iva, wasm_v128_load(inv_sqrt_vb + 2)));
    double sc[4];
    wasm_v128_store(sc, sc0);
    wasm_v128_store(sc + 2, sc1);

    int best = 0;
    for (int k = 1; k < 4; k++)
      if (sc[k] > sc[best]) best = k;
    int second = best == 0 ? 1 : 0;
    for (int k = 0; k < 4; k++)
      if (k != best && sc[k] > sc[second]) second = k;

    int votes = 0;
    for (int q = 0; q < 4; q++) {
      double qva = q_uu[q] - q_su[q] * q_su[q] / 49.0;
      double inv_sqrt_qva = qva < 1e-9 ? 0.0 : 1.0 / __builtin_sqrt(qva);
      v128_t v_iqva = wasm_f64x2_splat(inv_sqrt_qva);
      v128_t qsc0 = wasm_f64x2_mul(q_cov0[q], wasm_f64x2_mul(v_iqva, wasm_v128_load(q_inv_sqrt_vb[q])));
      v128_t qsc1 = wasm_f64x2_mul(q_cov1[q], wasm_f64x2_mul(v_iqva, wasm_v128_load(q_inv_sqrt_vb[q] + 2)));
      double q_sc[4];
      wasm_v128_store(q_sc, qsc0);
      wasm_v128_store(q_sc + 2, qsc1);
      int q_best = 0;
      for (int k = 1; k < 4; k++)
        if (q_sc[k] > q_sc[q_best]) q_best = k;
      if (q_best == best) votes++;
    }

    o[0] = 1.0f;
    o[1] = (float)best;
    o[2] = (float)sc[best];
    o[3] = (float)(sc[best] - sc[second]);
    o[4] = (float)votes;
  }
}

void* memset(void* p, int c, unsigned long n) {
  unsigned char* q = p;
  for (unsigned long i = 0; i < n; i++)
    q[i] = (unsigned char)c;
  return p;
}
