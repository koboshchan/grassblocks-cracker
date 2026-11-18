/* Freestanding browser detector: exact f64x2 perspective coords + 4-wide f32x4 bilerp & correlation. */
#include <wasm_simd128.h>
typedef unsigned char u8;
typedef unsigned short u16_unaligned __attribute__((aligned(1)));

static float refs_qc[256][4] __attribute__((aligned(16)));
static double q_fsum_div_49[4][4] __attribute__((aligned(16)));
static double sb_div_196[4] __attribute__((aligned(16)));
static double inv_sqrt_vb[4] __attribute__((aligned(16)));
static double q_inv_sqrt_vb[4][4] __attribute__((aligned(16)));
static double q_b_diff[4][4] __attribute__((aligned(16)));

void set_refs(const u8 *r) {
  double raw[256][4];
  double sb[4] = {0, 0, 0, 0}, bb[4] = {0, 0, 0, 0};
  double q_sb[4][4] = {{0}}, q_bb[4][4] = {{0}};

  for (int i = 0; i < 256; i++) {
    for (int k = 0; k < 4; k++) raw[i][k] = (double)r[k * 256 + i];
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
    for (int k = 0; k < 4; k++) q_fsum_div_49[q][k] = fsum[k] / 49.0;
  }
}

static inline int fit(const float *q, double *H) {
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

// Exact f64x2 coordinate projection for 4 pixels -> 4-wide f32x4 bilinear sample!
static inline v128_t sample4_f64(const u8 *g, int w,
                                 v128_t vnx01, v128_t vny01, v128_t vnd01,
                                 v128_t vnx23, v128_t vny23, v128_t vnd23) {
  const v128_t dones = wasm_f64x2_splat(1.0);
  v128_t inv01 = wasm_f64x2_div(dones, vnd01);
  v128_t inv23 = wasm_f64x2_div(dones, vnd23);

  v128_t px01 = wasm_f64x2_mul(vnx01, inv01);
  v128_t py01 = wasm_f64x2_mul(vny01, inv01);
  v128_t px23 = wasm_f64x2_mul(vnx23, inv23);
  v128_t py23 = wasm_f64x2_mul(vny23, inv23);

  v128_t flx01 = wasm_f64x2_floor(px01);
  v128_t fly01 = wasm_f64x2_floor(py01);
  v128_t flx23 = wasm_f64x2_floor(px23);
  v128_t fly23 = wasm_f64x2_floor(py23);

  v128_t fx01 = wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(px01, flx01));
  v128_t fx23 = wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(px23, flx23));
  v128_t fy01 = wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(py01, fly01));
  v128_t fy23 = wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(py23, fly23));

  v128_t vfx = wasm_i32x4_shuffle(fx01, fx23, 0, 1, 4, 5);
  v128_t vfy = wasm_i32x4_shuffle(fy01, fy23, 0, 1, 4, 5);

  v128_t vix = wasm_i32x4_shuffle(wasm_i32x4_trunc_sat_f64x2_zero(flx01),
                                  wasm_i32x4_trunc_sat_f64x2_zero(flx23), 0, 1, 4, 5);
  v128_t viy = wasm_i32x4_shuffle(wasm_i32x4_trunc_sat_f64x2_zero(fly01),
                                  wasm_i32x4_trunc_sat_f64x2_zero(fly23), 0, 1, 4, 5);

  v128_t voff = wasm_i32x4_add(wasm_i32x4_mul(viy, wasm_i32x4_splat(w)), vix);
  int o0 = wasm_i32x4_extract_lane(voff, 0);
  int o1 = wasm_i32x4_extract_lane(voff, 1);
  int o2 = wasm_i32x4_extract_lane(voff, 2);
  int o3 = wasm_i32x4_extract_lane(voff, 3);

  v128_t t16 = wasm_i32x4_make(*(const u16_unaligned *)(g + o0), *(const u16_unaligned *)(g + o1),
                               *(const u16_unaligned *)(g + o2), *(const u16_unaligned *)(g + o3));
  v128_t b16 = wasm_i32x4_make(*(const u16_unaligned *)(g + o0 + w), *(const u16_unaligned *)(g + o1 + w),
                               *(const u16_unaligned *)(g + o2 + w), *(const u16_unaligned *)(g + o3 + w));

  const v128_t m255 = wasm_i32x4_splat(0xff);
  v128_t g00 = wasm_f32x4_convert_i32x4(wasm_v128_and(t16, m255));
  v128_t g10 = wasm_f32x4_convert_i32x4(wasm_u32x4_shr(t16, 8));
  v128_t g01 = wasm_f32x4_convert_i32x4(wasm_v128_and(b16, m255));
  v128_t g11 = wasm_f32x4_convert_i32x4(wasm_u32x4_shr(b16, 8));

  v128_t top = wasm_f32x4_add(g00, wasm_f32x4_mul(vfx, wasm_f32x4_sub(g10, g00)));
  v128_t bot = wasm_f32x4_add(g01, wasm_f32x4_mul(vfx, wasm_f32x4_sub(g11, g01)));
  return wasm_f32x4_add(top, wasm_f32x4_mul(vfy, wasm_f32x4_sub(bot, top)));
}

void classify_batch(const u8 *g, int w, int h, const float *quads, int count, float *out) {
  const v128_t f128 = wasm_f32x4_splat(128.0f);
  const v128_t dzero = wasm_f64x2_splat(0.0);
  const double wm1 = (double)(w - 1), hm1 = (double)(h - 1);
  const double wm2 = (double)(w - 2), hm2 = (double)(h - 2);

  for (int t = 0; t < count; t++) {
    float *o = out + t * 5;
    double H[9];
    if (!fit(quads + t * 8, H)) {
      o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 0;
      continue;
    }

    static const double cx[4] = {1.5, 14.5, 1.5, 14.5}, cy[4] = {1.5, 1.5, 14.5, 14.5};
    int fast_bounds = 1, reject = 0;
    for (int c = 0; c < 4; c++) {
      double cd = H[6] * cx[c] + H[7] * cy[c] + 1.0;
      if (cd == 0.0) { fast_bounds = 0; break; }
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
      o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 0;
      continue;
    }

    double q_su[4] = {0, 0, 0, 0};
    double q_uu[4] = {0, 0, 0, 0};
    v128_t q_cov0[4] = {dzero, dzero, dzero, dzero};
    v128_t q_cov1[4] = {dzero, dzero, dzero, dzero};

    if (fast_bounds) {
      const v128_t dx01 = wasm_f64x2_make(0.0, H[0]);
      const v128_t dy01 = wasm_f64x2_make(0.0, H[3]);
      const v128_t dd01 = wasm_f64x2_make(0.0, H[6]);
      const v128_t step2x = wasm_f64x2_splat(2.0 * H[0]);
      const v128_t step2y = wasm_f64x2_splat(2.0 * H[3]);
      const v128_t step2d = wasm_f64x2_splat(2.0 * H[6]);
      const v128_t step4x = wasm_f64x2_splat(4.0 * H[0]);
      const v128_t step4y = wasm_f64x2_splat(4.0 * H[3]);
      const v128_t step4d = wasm_f64x2_splat(4.0 * H[6]);

      for (int j = 1; j < 15; j++) {
        int qrow = (j >= 8) ? 2 : 0;
        double vj = j + 0.5;
        v128_t vnx01 = wasm_f64x2_add(wasm_f64x2_splat(H[0] * 1.5 + H[1] * vj + H[2]), dx01);
        v128_t vny01 = wasm_f64x2_add(wasm_f64x2_splat(H[3] * 1.5 + H[4] * vj + H[5]), dy01);
        v128_t vnd01 = wasm_f64x2_add(wasm_f64x2_splat(H[6] * 1.5 + H[7] * vj + 1.0), dd01);
        v128_t vnx23 = wasm_f64x2_add(vnx01, step2x);
        v128_t vny23 = wasm_f64x2_add(vny01, step2y);
        v128_t vnd23 = wasm_f64x2_add(vnd01, step2d);

        v128_t u1_4 = wasm_f32x4_sub(sample4_f64(g, w, vnx01, vny01, vnd01, vnx23, vny23, vnd23), f128);
        vnx01 = wasm_f64x2_add(vnx01, step4x); vny01 = wasm_f64x2_add(vny01, step4y); vnd01 = wasm_f64x2_add(vnd01, step4d);
        vnx23 = wasm_f64x2_add(vnx23, step4x); vny23 = wasm_f64x2_add(vny23, step4y); vnd23 = wasm_f64x2_add(vnd23, step4d);

        v128_t u5_8 = wasm_f32x4_sub(sample4_f64(g, w, vnx01, vny01, vnd01, vnx23, vny23, vnd23), f128);
        vnx01 = wasm_f64x2_add(vnx01, step4x); vny01 = wasm_f64x2_add(vny01, step4y); vnd01 = wasm_f64x2_add(vnd01, step4d);
        vnx23 = wasm_f64x2_add(vnx23, step4x); vny23 = wasm_f64x2_add(vny23, step4y); vnd23 = wasm_f64x2_add(vnd23, step4d);

        v128_t u9_12 = wasm_f32x4_sub(sample4_f64(g, w, vnx01, vny01, vnd01, vnx23, vny23, vnd23), f128);
        // Columns 13..14 only need 1 f64x2 pair (vnx23 + step2x)! Reuse vnd23 for both pairs so lanes 2,3 = cols 13,14!
        vnx23 = wasm_f64x2_add(vnx23, step2x); vny23 = wasm_f64x2_add(vny23, step2y); vnd23 = wasm_f64x2_add(vnd23, step2d);
        v128_t u11_14 = wasm_f32x4_sub(sample4_f64(g, w, vnx23, vny23, vnd23, vnx23, vny23, vnd23), f128);

        int row_off = j * 16;

        v128_t row_covL = wasm_f32x4_add(
          wasm_f32x4_add(
            wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 0, 0, 0, 0), wasm_v128_load(refs_qc[row_off + 1])),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 1, 1, 1, 1), wasm_v128_load(refs_qc[row_off + 2]))
          ),
          wasm_f32x4_add(
            wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 2, 2, 2, 2), wasm_v128_load(refs_qc[row_off + 3])),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u1_4, u1_4, 3, 3, 3, 3), wasm_v128_load(refs_qc[row_off + 4]))
          )
        );
        row_covL = wasm_f32x4_add(
          row_covL,
          wasm_f32x4_add(
            wasm_f32x4_add(
              wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 0, 0, 0, 0), wasm_v128_load(refs_qc[row_off + 5])),
              wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 1, 1, 1, 1), wasm_v128_load(refs_qc[row_off + 6]))
            ),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 2, 2, 2, 2), wasm_v128_load(refs_qc[row_off + 7]))
          )
        );

        v128_t row_covR = wasm_f32x4_add(
          wasm_f32x4_add(
            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 0, 0, 0, 0), wasm_v128_load(refs_qc[row_off + 9])),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 1, 1, 1, 1), wasm_v128_load(refs_qc[row_off + 10]))
          ),
          wasm_f32x4_add(
            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 2, 2, 2, 2), wasm_v128_load(refs_qc[row_off + 11])),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u9_12, u9_12, 3, 3, 3, 3), wasm_v128_load(refs_qc[row_off + 12]))
          )
        );
        row_covR = wasm_f32x4_add(
          row_covR,
          wasm_f32x4_add(
            wasm_f32x4_add(
              wasm_f32x4_mul(wasm_i32x4_shuffle(u5_8, u5_8, 3, 3, 3, 3), wasm_v128_load(refs_qc[row_off + 8])),
              wasm_f32x4_mul(wasm_i32x4_shuffle(u11_14, u11_14, 2, 2, 2, 2), wasm_v128_load(refs_qc[row_off + 13]))
            ),
            wasm_f32x4_mul(wasm_i32x4_shuffle(u11_14, u11_14, 3, 3, 3, 3), wasm_v128_load(refs_qc[row_off + 14]))
          )
        );

        q_cov0[qrow] = wasm_f64x2_add(q_cov0[qrow], wasm_f64x2_promote_low_f32x4(row_covL));
        q_cov1[qrow] = wasm_f64x2_add(q_cov1[qrow], wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(row_covL, row_covL, 2, 3, 0, 1)));
        q_cov0[qrow + 1] = wasm_f64x2_add(q_cov0[qrow + 1], wasm_f64x2_promote_low_f32x4(row_covR));
        q_cov1[qrow + 1] = wasm_f64x2_add(q_cov1[qrow + 1], wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(row_covR, row_covR, 2, 3, 0, 1)));

        v128_t d12 = wasm_f64x2_promote_low_f32x4(u1_4);
        v128_t d34 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(u1_4, u1_4, 2, 3, 0, 1));
        v128_t d56 = wasm_f64x2_promote_low_f32x4(u5_8);
        v128_t d78 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(u5_8, u5_8, 2, 3, 0, 1));
        v128_t d7_0 = wasm_i64x2_shuffle(d78, dzero, 0, 2);
        v128_t sumL2 = wasm_f64x2_add(wasm_f64x2_add(d12, d34), wasm_f64x2_add(d56, d7_0));
        v128_t sqL2 = wasm_f64x2_add(wasm_f64x2_add(wasm_f64x2_mul(d12, d12), wasm_f64x2_mul(d34, d34)),
                                     wasm_f64x2_add(wasm_f64x2_mul(d56, d56), wasm_f64x2_mul(d7_0, d7_0)));
        q_su[qrow] += wasm_f64x2_extract_lane(sumL2, 0) + wasm_f64x2_extract_lane(sumL2, 1);
        q_uu[qrow] += wasm_f64x2_extract_lane(sqL2, 0) + wasm_f64x2_extract_lane(sqL2, 1);

        v128_t d9_10 = wasm_f64x2_promote_low_f32x4(u9_12);
        v128_t d11_12 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(u9_12, u9_12, 2, 3, 0, 1));
        v128_t d13_14 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(u11_14, u11_14, 2, 3, 0, 1));
        v128_t d0_8 = wasm_i64x2_shuffle(dzero, d78, 0, 3);
        v128_t sumR2 = wasm_f64x2_add(wasm_f64x2_add(d9_10, d11_12), wasm_f64x2_add(d13_14, d0_8));
        v128_t sqR2 = wasm_f64x2_add(wasm_f64x2_add(wasm_f64x2_mul(d9_10, d9_10), wasm_f64x2_mul(d11_12, d11_12)),
                                     wasm_f64x2_add(wasm_f64x2_mul(d13_14, d13_14), wasm_f64x2_mul(d0_8, d0_8)));
        q_su[qrow + 1] += wasm_f64x2_extract_lane(sumR2, 0) + wasm_f64x2_extract_lane(sumR2, 1);
        q_uu[qrow + 1] += wasm_f64x2_extract_lane(sqR2, 0) + wasm_f64x2_extract_lane(sqR2, 1);
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
          if (!(px >= 0.0 && py >= 0.0 && px < wm1 && py < hm1)) { valid = 0; break; }
          int xx = (int)px, yy = (int)py;
          double fx = px - xx, fy = py - yy;
          const u8 *p = g + yy * w + xx;
          double top = p[0] + fx * (p[1] - p[0]);
          double bot = p[w] + fx * (p[w + 1] - p[w]);
          float u = (float)(top + fy * (bot - top)) - 128.0f;
          double du = (double)u;
          int q = qrow + (i >= 8 ? 1 : 0);
          q_su[q] += du;
          q_uu[q] += du * du;
          v128_t c4 = wasm_f32x4_mul(wasm_f32x4_splat(u), wasm_v128_load(refs_qc[row_off + i]));
          q_cov0[q] = wasm_f64x2_add(q_cov0[q], wasm_f64x2_promote_low_f32x4(c4));
          q_cov1[q] = wasm_f64x2_add(q_cov1[q], wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(c4, c4, 2, 3, 0, 1)));
          nx += H[0]; ny += H[3]; nd += H[6];
        }
      }
      if (!valid) {
        o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 0;
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
      fcov1 = wasm_f64x2_add(fcov1, wasm_f64x2_add(cq1, wasm_f64x2_mul(vsaq, wasm_v128_load(q_b_diff[q] + 2))));
    }

    v128_t v_iva = wasm_f64x2_splat(inv_sqrt_va);
    v128_t sc0 = wasm_f64x2_mul(fcov0, wasm_f64x2_mul(v_iva, wasm_v128_load(inv_sqrt_vb)));
    v128_t sc1 = wasm_f64x2_mul(fcov1, wasm_f64x2_mul(v_iva, wasm_v128_load(inv_sqrt_vb + 2)));
    double sc[4];
    wasm_v128_store(sc, sc0);
    wasm_v128_store(sc + 2, sc1);

    int best = 0;
    for (int k = 1; k < 4; k++) if (sc[k] > sc[best]) best = k;
    int second = best == 0 ? 1 : 0;
    for (int k = 0; k < 4; k++) if (k != best && sc[k] > sc[second]) second = k;

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
      for (int k = 1; k < 4; k++) if (q_sc[k] > q_sc[q_best]) q_best = k;
      if (q_best == best) votes++;
    }

    o[0] = 1.0f;
    o[1] = (float)best;
    o[2] = (float)sc[best];
    o[3] = (float)(sc[best] - sc[second]);
    o[4] = (float)votes;
  }
}

void *memset(void *p, int c, unsigned long n) {
  unsigned char *q = p;
  for (unsigned long i = 0; i < n; i++) q[i] = (unsigned char)c;
  return p;
}
