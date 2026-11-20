#include "geometry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif
namespace grass {
double clockMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static const Quad unit = {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)};
static bool solve(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& out) {
  int n = b.size();
  for (int i = 0; i < n; i++)
    A[i].push_back(b[i]);
  for (int j = 0; j < n; j++) {
    int k = j;
    for (int i = j + 1; i < n; i++)
      if (std::abs(A[i][j]) > std::abs(A[k][j])) k = i;
    std::swap(A[j], A[k]);
    if (std::abs(A[j][j]) < 1e-12) return false;
    double d = A[j][j];
    for (int t = j; t <= n; t++)
      A[j][t] /= d;
    for (int i = 0; i < n; i++)
      if (i != j) {
        double f = A[i][j];
        if (f)
          for (int t = j; t <= n; t++)
            A[i][t] -= f * A[j][t];
      }
  }
  out.resize(n);
  for (int i = 0; i < n; i++)
    out[i] = A[i][n];
  return true;
}
bool homography(const std::vector<Point>& s, const std::vector<Point>& d, H3& H) {
  std::vector<std::vector<double>> N(8, std::vector<double>(8));
  std::vector<double> r(8), o;
  for (size_t i = 0; i < s.size(); i++) {
    double x = s[i].x, y = s[i].y, u = d[i].x, v = d[i].y;
    double rows[2][9] = {{x, y, 1, 0, 0, 0, -u * x, -u * y, u}, {0, 0, 0, x, y, 1, -v * x, -v * y, v}};
    for (auto& q : rows)
      for (int a = 0; a < 8; a++) {
        r[a] += q[a] * q[8];
        for (int b = 0; b < 8; b++)
          N[a][b] += q[a] * q[b];
      }
  }
  if (!solve(N, r, o)) return false;
  std::copy(o.begin(), o.end(), H.begin());
  H[8] = 1;
  return true;
}
Point applyH(const H3& H, double x, double y) {
  double d = H[6] * x + H[7] * y + H[8];
  return {(H[0] * x + H[1] * y + H[2]) / d, (H[3] * x + H[4] * y + H[5]) / d};
}
bool inv3(const H3& m, H3& o) {
  double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8],
         A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (std::abs(det) < 1e-18) return false;
  o = {A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
       B / det, (a * i - c * g) / det,  -(a * f - c * d) / det,
       C / det, -(a * h - b * g) / det, (a * e - b * d) / det};
  return true;
}
static H3 mul3(const H3& a, const H3& b) {
  H3 o{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
}
static double dist(Point a, Point b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}
static double minSide(const Quad& q) {
  double m = INFINITY;
  for (int i = 0; i < 4; i++)
    m = std::min(m, dist(q[i], q[(i + 1) % 4]));
  return m;
}
static bool convex(const Quad& q) {
  int s = 0;
  for (int i = 0; i < 4; i++) {
    Point a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    double z = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (!z) return false;
    int v = z > 0 ? 1 : -1;
    if (!s)
      s = v;
    else if (s != v)
      return false;
  }
  return true;
}
static bool inImage(const Quad& q, int w, int h) {
  for (auto p : q)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= w - 1 || p.y >= h - 1)
      return false;
  return true;
}
static Quad gridQuad(const H3& H, int x, int z) {
  Quad q;
  for (int i = 0; i < 4; i++)
    q[i] = applyH(H, x + unit[i].x, z + unit[i].y);
  return q;
}
static double corrIds(const float* patch, const double* ref, const int* ids, int count) {
  double sa = 0, sb = 0, aa = 0, bb = 0, ab = 0;
  for (int t = 0; t < count; t++) {
    int idx = ids[t];
    double a = patch[idx], b = ref[idx];
    sa += a;
    sb += b;
    aa += a * a;
    bb += b * b;
    ab += a * b;
  }
  double n = count, va = aa - sa * sa / n, vb = bb - sb * sb / n;
  if (va < 1e-9 || vb < 1e-9) return 0;
  return (ab - sa * sb / n) / std::sqrt(va * vb);
}
namespace {
struct ClassifyTables {
  int inner[196];
  int quads[4][49];
  ClassifyTables() {
    int nInner = 0, nQuad[4] = {0, 0, 0, 0};
    for (int j = 1; j < 15; j++)
      for (int i = 1; i < 15; i++) {
        int idx = j * 16 + i;
        inner[nInner++] = idx;
        int qid = (i >= 8 ? 1 : 0) + (j >= 8 ? 2 : 0);
        quads[qid][nQuad[qid]++] = idx;
      }
  }
};
const ClassifyTables& classifyTables() {
  static const ClassifyTables tbl;
  return tbl;
}
}  // namespace
Verdict classify(const Context& c, const Quad& q) {
  H3 H;
  std::vector<Point> s = {{0, 0}, {16, 0}, {16, 16}, {0, 16}}, d(q.begin(), q.end());
  if (!homography(s, d, H)) return {};
  float patch[256];
  const uint8_t* g = c.g.data();
  for (int j = 0; j < 16; j++)
    for (int i = 0; i < 16; i++) {
      Point p = applyH(H, i + .5, j + .5);
      if (!(p.x >= 0 && p.y >= 0 && p.x < c.w - 1 && p.y < c.h - 1)) return {};
      int xx = std::floor(p.x), yy = std::floor(p.y);
      double fx = p.x - xx, fy = p.y - yy;
      patch[j * 16 + i] = g[yy * c.w + xx] * (1 - fx) * (1 - fy) + g[yy * c.w + xx + 1] * fx * (1 - fy) +
                          g[(yy + 1) * c.w + xx] * (1 - fx) * fy + g[(yy + 1) * c.w + xx + 1] * fx * fy;
    }
  const auto& tbl = classifyTables();
  std::array<double, 4> sc;
  for (int k = 0; k < 4; k++)
    sc[k] = corrIds(patch, c.refs[k].data(), tbl.inner, 196);
  std::array<int, 4> order = {0, 1, 2, 3};
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] > sc[b]; });
  int k = order[0], votes = 0;
  for (int qid = 0; qid < 4; qid++) {
    int best = 0;
    double bestScore = -1e30;
    for (int a = 0; a < 4; a++) {
      double score = corrIds(patch, c.refs[a].data(), tbl.quads[qid], 49);
      if (a == 0 || score > bestScore) {
        bestScore = score;
        best = a;
      }
    }
    if (best == k) votes++;
  }
  return {true, k, votes, sc[k], sc[k] - sc[order[1]]};
}
// Batch verification deliberately uses the existing Float32 ABI and closed-form
// kernel; single/ECC verification uses the double precision reference above.
extern "C" {
void set_refs(const unsigned char*);
void classify_batch(const unsigned char*, int, int, const float*, int, float*);
}
std::vector<Verdict> classifyBatchNative(const Context& c, const std::vector<Quad>& qs) {
  static thread_local bool hasRefs = false;
  static thread_local std::array<unsigned char, 1024> cachedRefs{};
  static thread_local std::vector<float> flat, out;
  bool match = hasRefs;
  for (int k = 0; k < 4 && match; k++)
    for (int i = 0; i < 256; i++)
      if (static_cast<unsigned char>(c.refs[k][i]) != cachedRefs[k * 256 + i]) {
        match = false;
        break;
      }
  if (!match) {
    for (int k = 0; k < 4; k++)
      for (int i = 0; i < 256; i++)
        cachedRefs[k * 256 + i] = c.refs[k][i];
    set_refs(cachedRefs.data());
    hasRefs = true;
  }
  const size_t n = qs.size();
  flat.resize(n * 8);
  out.resize(n * 5);
  float* fp = flat.data();
  for (size_t j = 0; j < n; j++) {
    const Quad& q = qs[j];
    fp[0] = float(q[0].x);
    fp[1] = float(q[0].y);
    fp[2] = float(q[1].x);
    fp[3] = float(q[1].y);
    fp[4] = float(q[2].x);
    fp[5] = float(q[2].y);
    fp[6] = float(q[3].x);
    fp[7] = float(q[3].y);
    fp += 8;
  }
  classify_batch(c.g.data(), c.w, c.h, flat.data(), int(n), out.data());
  std::vector<Verdict> r(n);
  const float* op = out.data();
  for (size_t i = 0; i < n; i++, op += 5)
    r[i] = {op[0] != 0, int(op[1]), int(op[4]), op[2], op[3]};
  return r;
}
namespace {
struct EccKey {
  std::array<int64_t, 8> q;
  int rot;
  bool operator==(const EccKey& o) const {
    return rot == o.rot && q == o.q;
  }
};

struct EccKeyHash {
  size_t operator()(const EccKey& k) const noexcept {
    uint64_t h = uint64_t(k.rot) * 0x9e3779b97f4a7c15ULL;
    for (int64_t v : k.q)
      h ^= uint64_t(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

struct EccCached {
  bool ok = false;
  Quad nq{};
};

inline float dotF32(const float* a, const float* b, int n) {
#ifdef __wasm_simd128__
  v128_t acc0 = wasm_f32x4_splat(0.0f);
  v128_t acc1 = wasm_f32x4_splat(0.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
  }
  v128_t sum = wasm_f32x4_add(acc0, acc1);
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
  sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
  float s = wasm_f32x4_extract_lane(sum, 0);
  for (; i < n; ++i)
    s += a[i] * b[i];
  return s;
#else
  float s = 0.0f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
#endif
}

bool invert8x8(const double A[8][8], double inv[8][8]) {
  double aug[8][16]{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j)
      aug[i][j] = A[i][j];
    aug[i][8 + i] = 1.0;
  }
  for (int col = 0; col < 8; ++col) {
    int piv = col;
    for (int r = col + 1; r < 8; ++r) {
      if (std::abs(aug[r][col]) > std::abs(aug[piv][col])) piv = r;
    }
    if (std::abs(aug[piv][col]) < 1e-12) return false;
    if (piv != col) {
      for (int j = 0; j < 16; ++j)
        std::swap(aug[col][j], aug[piv][j]);
    }
    double d = aug[col][col];
    for (int j = col; j < 16; ++j)
      aug[col][j] /= d;
    for (int r = 0; r < 8; ++r) {
      if (r == col) continue;
      double f = aug[r][col];
      if (f == 0.0) continue;
      for (int j = col; j < 16; ++j)
        aug[r][j] -= f * aug[col][j];
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j)
      inv[i][j] = aug[i][8 + j];
  }
  return true;
}
}  // namespace

// Precomputed Inverse-Compositional ECC on 64x64 upscaled grass templates.
// Eliminates per-iteration blurring, gradient, Jacobian, and Hessian solves.
class ECC {
  static constexpr int N = 64;
  static constexpr int M = (N - 2) * (N - 2);  // 62 * 62 = 3844 (multiple of 4)
  int w = 0, h = 0;
  const std::vector<uint8_t>* gSrc = nullptr;
  std::vector<float> img;
  bool imgReady = false;
  struct Template {
    bool valid = false;
    std::vector<float> tBar;
    std::vector<float> pMat8;  // interleaved [M][8] for single-pass SIMD dot product
  };
  std::array<Template, 4> templates;
  std::vector<float> patchBuf;
  std::unordered_map<EccKey, EccCached, EccKeyHash> cache;

  void ensureImg() {
    if (imgReady || !gSrc) return;
    const size_t n = gSrc->size();
    img.resize(n);
    const uint8_t* src = gSrc->data();
    float* dst = img.data();
    const float scale = 1.0f / 255.0f;
    size_t i = 0;
#ifdef __wasm_simd128__
    const v128_t vscale = wasm_f32x4_splat(scale);
    for (; i + 8 <= n; i += 8) {
      v128_t u16 = wasm_u16x8_load8x8(src + i);
      v128_t u32_lo = wasm_u32x4_extend_low_u16x8(u16);
      v128_t u32_hi = wasm_u32x4_extend_high_u16x8(u16);
      wasm_v128_store(dst + i, wasm_f32x4_mul(wasm_f32x4_convert_u32x4(u32_lo), vscale));
      wasm_v128_store(dst + i + 4, wasm_f32x4_mul(wasm_f32x4_convert_u32x4(u32_hi), vscale));
    }
#endif
    for (; i < n; ++i)
      dst[i] = src[i] * scale;
    imgReady = true;
  }

  static bool fitNormQuad(const Quad& q, H3& W) {
    // Maps [-1,-1], [1,-1], [1,1], [-1,1] -> q[0..3] via unit square [0,1]^2 first.
    double x0 = q[0].x, y0 = q[0].y, x1 = q[1].x, y1 = q[1].y;
    double x2 = q[2].x, y2 = q[2].y, x3 = q[3].x, y3 = q[3].y;
    double dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    double dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    double det = dx1 * dy2 - dx2 * dy1;
    if (std::abs(det) < 1e-12) return false;
    double g = (dx3 * dy2 - dx2 * dy3) / det;
    double hh = (dx1 * dy3 - dx3 * dy1) / det;
    double a = (x1 - x0) + g * x1;
    double b = (x3 - x0) + hh * x3;
    double c = x0;
    double d = (y1 - y0) + g * y1;
    double e = (y3 - y0) + hh * y3;
    double f = y0;
    // Compose with u = 0.5*(xi+1), v = 0.5*(eta+1):
    // H_unit * [0.5 0 0.5; 0 0.5 0.5; 0 0 1]
    double w8 = 0.5 * g + 0.5 * hh + 1.0;
    if (std::abs(w8) < 1e-12) return false;
    double inv8 = 1.0 / w8;
    W[0] = 0.5 * a * inv8;
    W[1] = 0.5 * b * inv8;
    W[2] = (0.5 * a + 0.5 * b + c) * inv8;
    W[3] = 0.5 * d * inv8;
    W[4] = 0.5 * e * inv8;
    W[5] = (0.5 * d + 0.5 * e + f) * inv8;
    W[6] = 0.5 * g * inv8;
    W[7] = 0.5 * hh * inv8;
    W[8] = 1.0;
    return true;
  }

  static void blurSeparable121(std::vector<double>& grid) {
    std::vector<double> tmp(N * N, 0.0);
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        int xm = std::max(0, x - 1), xp = std::min(N - 1, x + 1);
        tmp[y * N + x] = 0.25 * grid[y * N + xm] + 0.5 * grid[y * N + x] + 0.25 * grid[y * N + xp];
      }
    }
    for (int y = 0; y < N; ++y) {
      int ym = std::max(0, y - 1), yp = std::min(N - 1, y + 1);
      for (int x = 0; x < N; ++x) {
        grid[y * N + x] = 0.25 * tmp[ym * N + x] + 0.5 * tmp[y * N + x] + 0.25 * tmp[yp * N + x];
      }
    }
  }

  void buildTemplate(int rot, const std::array<double, 256>& ref) {
    std::vector<double> grid(N * N, 0.0);
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        grid[y * N + x] = ref[(y >> 2) * 16 + (x >> 2)] / 255.0;
      }
    }
    blurSeparable121(grid);
    blurSeparable121(grid);

    double meanT = 0.0;
    std::vector<double> tVec(M, 0.0);
    int idx = 0;
    for (int y = 1; y < N - 1; ++y) {
      for (int x = 1; x < N - 1; ++x) {
        double v = grid[y * N + x];
        tVec[idx++] = v;
        meanT += v;
      }
    }
    meanT /= M;
    double norm2 = 0.0;
    for (int i = 0; i < M; ++i) {
      tVec[i] -= meanT;
      norm2 += tVec[i] * tVec[i];
    }
    if (norm2 < 1e-12) return;
    double invNorm = 1.0 / std::sqrt(norm2);
    templates[rot].tBar.resize(M);
    for (int i = 0; i < M; ++i) {
      tVec[i] *= invNorm;
      templates[rot].tBar[i] = static_cast<float>(tVec[i]);
    }

    const double invTwoStep = double(N) / 4.0;
    std::vector<double> J(8 * M, 0.0);
    idx = 0;
    for (int y = 1; y < N - 1; ++y) {
      double eta = (2.0 * (y + 0.5)) / N - 1.0;
      for (int x = 1; x < N - 1; ++x) {
        double xi = (2.0 * (x + 0.5)) / N - 1.0;
        double gx = (grid[y * N + (x + 1)] - grid[y * N + (x - 1)]) * invTwoStep * invNorm;
        double gy = (grid[(y + 1) * N + x] - grid[(y - 1) * N + x]) * invTwoStep * invNorm;
        J[0 * M + idx] = gx * xi;
        J[1 * M + idx] = gx * eta;
        J[2 * M + idx] = gx;
        J[3 * M + idx] = gy * xi;
        J[4 * M + idx] = gy * eta;
        J[5 * M + idx] = gy;
        J[6 * M + idx] = -xi * (gx * xi + gy * eta);
        J[7 * M + idx] = -eta * (gx * xi + gy * eta);
        ++idx;
      }
    }
    for (int k = 0; k < 8; ++k) {
      double* jk = J.data() + k * M;
      double meanK = 0.0;
      for (int i = 0; i < M; ++i)
        meanK += jk[i];
      meanK /= M;
      double proj = 0.0;
      for (int i = 0; i < M; ++i) {
        jk[i] -= meanK;
        proj += jk[i] * tVec[i];
      }
      for (int i = 0; i < M; ++i)
        jk[i] -= proj * tVec[i];
    }

    double H[8][8]{}, Hinv[8][8]{};
    for (int a = 0; a < 8; ++a) {
      const double* ja = J.data() + a * M;
      for (int b = a; b < 8; ++b) {
        const double* jb = J.data() + b * M;
        double s = 0.0;
        for (int i = 0; i < M; ++i)
          s += ja[i] * jb[i];
        H[a][b] = H[b][a] = s;
      }
      H[a][a] += 1e-7;
    }
    if (!invert8x8(H, Hinv)) return;

    templates[rot].pMat8.resize(M * 8);
    for (int i = 0; i < M; ++i) {
      for (int a = 0; a < 8; ++a) {
        double s = 0.0;
        for (int b = 0; b < 8; ++b)
          s += Hinv[a][b] * J[b * M + i];
        templates[rot].pMat8[i * 8 + a] = static_cast<float>(s);
      }
    }
    templates[rot].valid = true;
  }

  bool samplePatchAndDot(const H3& W, const float* tBar, std::vector<float>& out, double& sumW, double& sumWW,
                         double& tw) const {
    const float* src = img.data();
    const int maxX = w - 1, maxY = h - 1;
    const double step = 2.0 / N;
    const double xiStart = (2.0 * 1.5) / N - 1.0;
    const double xiEnd = (2.0 * (N - 1.5)) / N - 1.0;

    // Check if all 4 interior corners map strictly inside [0, maxX) x [0, maxY) with rd > 1e-6.
    bool interior = true;
    {
      const double limX = maxX - 1e-6, limY = maxY - 1e-6;
      const double cs[4][2] = {{xiStart, xiStart}, {xiEnd, xiStart}, {xiEnd, xiEnd}, {xiStart, xiEnd}};
      for (int c = 0; c < 4; ++c) {
        double rd = W[6] * cs[c][0] + W[7] * cs[c][1] + W[8];
        if (rd <= 1e-6) {
          interior = false;
          break;
        }
        double px = (W[0] * cs[c][0] + W[1] * cs[c][1] + W[2]) / rd;
        double py = (W[3] * cs[c][0] + W[4] * cs[c][1] + W[5]) / rd;
        if (!(px >= 0.0 && py >= 0.0 && px < limX && py < limY)) {
          interior = false;
          break;
        }
      }
    }

    float* outPtr = out.data();
    double sw = 0.0, sww = 0.0;
    int idx = 0;

#ifdef __wasm_simd128__
    if (interior) {
      const double dx = W[0] * step, dy = W[3] * step, dd = W[6] * step;
      const v128_t vdx2 = wasm_f64x2_splat(dx * 2.0);
      const v128_t vdy2 = wasm_f64x2_splat(dy * 2.0);
      const v128_t vdd2 = wasm_f64x2_splat(dd * 2.0);
      const v128_t vdx_off = wasm_f64x2_make(0.0, dx);
      const v128_t vdy_off = wasm_f64x2_make(0.0, dy);
      const v128_t vdd_off = wasm_f64x2_make(0.0, dd);
      const v128_t vone = wasm_f32x4_splat(1.0f);
      v128_t vsw0 = wasm_f64x2_splat(0.0), vsw1 = wasm_f64x2_splat(0.0);
      v128_t vsww0 = wasm_f64x2_splat(0.0), vsww1 = wasm_f64x2_splat(0.0);
      v128_t vtw = wasm_f32x4_splat(0.0f);
      float twTail = 0.0f;

      for (int y = 1; y < N - 1; ++y) {
        double eta = (2.0 * (y + 0.5)) / N - 1.0;
        double rx0 = W[0] * xiStart + W[1] * eta + W[2];
        double ry0 = W[3] * xiStart + W[4] * eta + W[5];
        double rd0 = W[6] * xiStart + W[7] * eta + W[8];
        v128_t vrx = wasm_f64x2_add(wasm_f64x2_splat(rx0), vdx_off);
        v128_t vry = wasm_f64x2_add(wasm_f64x2_splat(ry0), vdy_off);
        v128_t vrd = wasm_f64x2_add(wasm_f64x2_splat(rd0), vdd_off);

        int x = 1;
        for (; x + 3 < N - 1; x += 4) {
          v128_t px01 = wasm_f64x2_div(vrx, vrd);
          v128_t py01 = wasm_f64x2_div(vry, vrd);
          vrx = wasm_f64x2_add(vrx, vdx2);
          vry = wasm_f64x2_add(vry, vdy2);
          vrd = wasm_f64x2_add(vrd, vdd2);
          v128_t px23 = wasm_f64x2_div(vrx, vrd);
          v128_t py23 = wasm_f64x2_div(vry, vrd);
          vrx = wasm_f64x2_add(vrx, vdx2);
          vry = wasm_f64x2_add(vry, vdy2);
          vrd = wasm_f64x2_add(vrd, vdd2);

          v128_t ix01 = wasm_f64x2_floor(px01), iy01 = wasm_f64x2_floor(py01);
          v128_t ix23 = wasm_f64x2_floor(px23), iy23 = wasm_f64x2_floor(py23);
          v128_t fx =
              wasm_i32x4_shuffle(wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(px01, ix01)),
                                 wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(px23, ix23)), 0, 1, 4, 5);
          v128_t fy =
              wasm_i32x4_shuffle(wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(py01, iy01)),
                                 wasm_f32x4_demote_f64x2_zero(wasm_f64x2_sub(py23, iy23)), 0, 1, 4, 5);

          int i0 = int(wasm_f64x2_extract_lane(iy01, 0)) * w + int(wasm_f64x2_extract_lane(ix01, 0));
          int i1 = int(wasm_f64x2_extract_lane(iy01, 1)) * w + int(wasm_f64x2_extract_lane(ix01, 1));
          int i2 = int(wasm_f64x2_extract_lane(iy23, 0)) * w + int(wasm_f64x2_extract_lane(ix23, 0));
          int i3 = int(wasm_f64x2_extract_lane(iy23, 1)) * w + int(wasm_f64x2_extract_lane(ix23, 1));

          v128_t p00 = wasm_f32x4_make(src[i0], src[i1], src[i2], src[i3]);
          v128_t p10 = wasm_f32x4_make(src[i0 + 1], src[i1 + 1], src[i2 + 1], src[i3 + 1]);
          v128_t p01 = wasm_f32x4_make(src[i0 + w], src[i1 + w], src[i2 + w], src[i3 + w]);
          v128_t p11 = wasm_f32x4_make(src[i0 + w + 1], src[i1 + w + 1], src[i2 + w + 1], src[i3 + w + 1]);

          v128_t v0 = wasm_f32x4_add(p00, wasm_f32x4_mul(wasm_f32x4_sub(p10, p00), fx));
          v128_t v1 = wasm_f32x4_add(p01, wasm_f32x4_mul(wasm_f32x4_sub(p11, p01), fx));
          v128_t v = wasm_f32x4_add(v0, wasm_f32x4_mul(wasm_f32x4_sub(v1, v0), fy));

          wasm_v128_store(outPtr + idx, v);
          vtw = wasm_f32x4_add(vtw, wasm_f32x4_mul(v, wasm_v128_load(tBar + idx)));
          v128_t d0 = wasm_f64x2_promote_low_f32x4(v);
          v128_t d1 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
          vsw0 = wasm_f64x2_add(vsw0, d0);
          vsw1 = wasm_f64x2_add(vsw1, d1);
          vsww0 = wasm_f64x2_add(vsww0, wasm_f64x2_mul(d0, d0));
          vsww1 = wasm_f64x2_add(vsww1, wasm_f64x2_mul(d1, d1));
          idx += 4;
        }
        // 2 remaining pixels in the 62-wide row (x = 61, 62)
        v128_t px01 = wasm_f64x2_div(vrx, vrd);
        v128_t py01 = wasm_f64x2_div(vry, vrd);
        for (int lane = 0; lane < 2; ++lane) {
          double px = lane == 0 ? wasm_f64x2_extract_lane(px01, 0) : wasm_f64x2_extract_lane(px01, 1);
          double py = lane == 0 ? wasm_f64x2_extract_lane(py01, 0) : wasm_f64x2_extract_lane(py01, 1);
          int ix = int(px), iy = int(py);
          float fxx = float(px - ix), fyy = float(py - iy);
          const float* r0 = src + iy * w + ix;
          float v0 = r0[0] + (r0[1] - r0[0]) * fxx;
          float v1 = r0[w] + (r0[w + 1] - r0[w]) * fxx;
          float v = v0 + (v1 - v0) * fyy;
          outPtr[idx] = v;
          twTail += v * tBar[idx];
          sw += v;
          sww += double(v) * double(v);
          ++idx;
        }
      }
      v128_t swSum = wasm_f64x2_add(vsw0, vsw1);
      v128_t swwSum = wasm_f64x2_add(vsww0, vsww1);
      sumW = sw + wasm_f64x2_extract_lane(swSum, 0) + wasm_f64x2_extract_lane(swSum, 1);
      sumWW = sww + wasm_f64x2_extract_lane(swwSum, 0) + wasm_f64x2_extract_lane(swwSum, 1);
      vtw = wasm_f32x4_add(vtw, wasm_i32x4_shuffle(vtw, vtw, 2, 3, 0, 1));
      vtw = wasm_f32x4_add(vtw, wasm_i32x4_shuffle(vtw, vtw, 1, 0, 3, 2));
      tw = double(wasm_f32x4_extract_lane(vtw, 0) + twTail);
      (void)vone;
      return true;
    }
#endif

    float twScalar = 0.0f;
    for (int y = 1; y < N - 1; ++y) {
      double eta = (2.0 * (y + 0.5)) / N - 1.0;
      double rx = W[0] * xiStart + W[1] * eta + W[2];
      double ry = W[3] * xiStart + W[4] * eta + W[5];
      double rd = W[6] * xiStart + W[7] * eta + W[8];
      double dx = W[0] * step, dy = W[3] * step, dd = W[6] * step;
      for (int x = 1; x < N - 1; ++x) {
        if (std::abs(rd) < 1e-12) return false;
        double invD = 1.0 / rd;
        double px = rx * invD;
        double py = ry * invD;
        if (px < 0.0)
          px = 0.0;
        else if (px > maxX)
          px = maxX;
        if (py < 0.0)
          py = 0.0;
        else if (py > maxY)
          py = maxY;
        int ix = int(px), iy = int(py);
        int ix1 = std::min(ix + 1, maxX), iy1 = std::min(iy + 1, maxY);
        float fx = float(px - ix), fy = float(py - iy);
        const float* r0 = src + iy * w;
        const float* r1 = src + iy1 * w;
        float v0 = r0[ix] + (r0[ix1] - r0[ix]) * fx;
        float v1 = r1[ix] + (r1[ix1] - r1[ix]) * fx;
        float v = v0 + (v1 - v0) * fy;
        outPtr[idx] = v;
        twScalar += v * tBar[idx];
        sw += v;
        sww += double(v) * double(v);
        ++idx;
        rx += dx;
        ry += dy;
        rd += dd;
      }
    }
    sumW = sw;
    sumWW = sww;
    tw = double(twScalar);
    return true;
  }

  static void computeDp8(const float* pMat8, const float* patch, double invTw, double dp[8]) {
#ifdef __wasm_simd128__
    v128_t a0 = wasm_f32x4_splat(0.0f), a1 = wasm_f32x4_splat(0.0f);
    v128_t b0 = wasm_f32x4_splat(0.0f), b1 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 2 <= M; i += 2) {
      v128_t vp0 = wasm_f32x4_splat(patch[i]);
      v128_t vp1 = wasm_f32x4_splat(patch[i + 1]);
      const float* pm = pMat8 + i * 8;
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(vp0, wasm_v128_load(pm)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(vp0, wasm_v128_load(pm + 4)));
      b0 = wasm_f32x4_add(b0, wasm_f32x4_mul(vp1, wasm_v128_load(pm + 8)));
      b1 = wasm_f32x4_add(b1, wasm_f32x4_mul(vp1, wasm_v128_load(pm + 12)));
    }
    v128_t s0 = wasm_f32x4_add(a0, b0);
    v128_t s1 = wasm_f32x4_add(a1, b1);
    float tmp[8];
    wasm_v128_store(tmp, s0);
    wasm_v128_store(tmp + 4, s1);
    for (int k = 0; k < 8; ++k)
      dp[k] = double(tmp[k]) * invTw;
#else
    float acc[8]{};
    for (int i = 0; i < M; ++i) {
      float v = patch[i];
      const float* pm = pMat8 + i * 8;
      for (int k = 0; k < 8; ++k)
        acc[k] += pm[k] * v;
    }
    for (int k = 0; k < 8; ++k)
      dp[k] = double(acc[k]) * invTw;
#endif
  }

 public:
  ECC(const Context& c) : w(c.w), h(c.h), gSrc(&c.g), patchBuf(M, 0.0f) {
    for (int k = 0; k < 4; ++k)
      buildTemplate(k, c.refs[k]);
  }

  bool operator()(const Quad& q, int rot, Quad& nq) {
    if (rot < 0 || rot >= 4 || !templates[rot].valid) return false;
    EccKey key{{}, rot};
    for (int i = 0; i < 4; ++i) {
      key.q[i * 2] = std::llround(q[i].x * 8.0);
      key.q[i * 2 + 1] = std::llround(q[i].y * 8.0);
    }
    auto it = cache.find(key);
    if (it != cache.end()) {
      nq = it->second.nq;
      return it->second.ok;
    }

    ensureImg();

    static const Point normCorners[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
    H3 W;
    if (!fitNormQuad(q, W)) {
      if (cache.size() < 32768) cache.emplace(key, EccCached{false, {}});
      return false;
    }

    const auto& tpl = templates[rot];
    auto& patch = patchBuf;
    H3 bestW = W;
    double bestRho = -2.0;
    int drops = 0;

    for (int iter = 0; iter < 30; ++iter) {
      double sw = 0.0, sww = 0.0, tw = 0.0;
      if (!samplePatchAndDot(W, tpl.tBar.data(), patch, sw, sww, tw)) break;
      double var = sww - (sw * sw) / M;
      if (var < 1e-10) break;
      double rho = tw / std::sqrt(var);
      if (rho > bestRho) {
        bestRho = rho;
        bestW = W;
        drops = 0;
      } else {
        if (++drops >= 2) break;
      }
      if (tw <= 1e-6) break;

      double invTw = 1.0 / tw;
      double dp[8];
      computeDp8(tpl.pMat8.data(), patch.data(), invTw, dp);
      double maxStep = 0.0;
      for (int k = 0; k < 8; ++k)
        maxStep = std::max(maxStep, std::abs(dp[k]));
      if (!std::isfinite(maxStep)) break;
      if (maxStep > 0.25) {
        double scale = 0.25 / maxStep;
        for (int k = 0; k < 8; ++k)
          dp[k] *= scale;
      }
      H3 dW = {1.0 + dp[0], dp[1], dp[2], dp[3], 1.0 + dp[4], dp[5], dp[6], dp[7], 1.0};
      H3 dWi;
      if (!inv3(dW, dWi)) break;
      W = mul3(W, dWi);
      if (std::abs(W[8]) < 1e-12) break;
      double invW8 = 1.0 / W[8];
      for (double& v : W)
        v *= invW8;
      if (maxStep < 1e-4) {
        double sw2 = 0.0, sww2 = 0.0, tw2 = 0.0;
        if (samplePatchAndDot(W, tpl.tBar.data(), patch, sw2, sww2, tw2)) {
          double var2 = sww2 - (sw2 * sw2) / M;
          if (var2 >= 1e-10) {
            double rho2 = tw2 / std::sqrt(var2);
            if (rho2 > bestRho) {
              bestRho = rho2;
              bestW = W;
            }
          }
        }
        break;
      }
    }

    bool ok = bestRho > 0.0;
    if (ok) {
      for (int i = 0; i < 4; ++i)
        nq[i] = applyH(bestW, normCorners[i].x, normCorners[i].y);
    }
    if (cache.size() < 32768) cache.emplace(key, EccCached{ok, nq});
    return ok;
  }
};

Refiner makeEccRefiner(const Context& c) {
  auto ecc = std::make_shared<ECC>(c);
  return [ecc](const Quad& q, int rot, Quad& nq) { return (*ecc)(q, rot, nq); };
}
static Block verify(const Context& c, Quad q, const Refiner& refine, Verdict r, bool supplied = false) {
  if (!supplied) r = classify(c, q);
  Block b;
  if (!r.valid) return b;
  auto& t = c.opts;
  double rmin = std::isnan(t.refineMin) ? t.relaxed : t.refineMin,
         rmargin = std::isnan(t.refineMargin) ? t.margin : t.refineMargin;
  if (refine && r.score >= rmin && r.margin >= rmargin && r.votes >= t.refineVotes) {
    Quad nq;
    if (refine(q, r.rotation, nq) && convex(nq) && inImage(nq, c.w, c.h)) {
      double shift = 0;
      for (int i = 0; i < 4; i++)
        shift = std::max(shift, dist(nq[i], q[i]));
      if (shift < t.refineShift) {
        auto r2 = classify(c, nq);
        if (r2.valid && r2.score > r.score) {
          q = nq;
          r = r2;
        }
      }
    }
  }
  static_cast<Verdict&>(b) = r;
  b.quad = q;
  return b;
}
static Plane planeFromSeeds(const std::vector<Block>& seeds) {
  Plane best;
  std::vector<Point> u(unit.begin(), unit.end());
  for (auto& s : seeds) {
    H3 H0, I;
    if (!homography(u, std::vector<Point>(s.quad.begin(), s.quad.end()), H0) || !inv3(H0, I)) continue;
    std::vector<int> ids;
    std::vector<Point> src, dst;
    for (size_t k = 0; k < seeds.size(); k++) {
      Quad loc;
      for (int i = 0; i < 4; i++)
        loc[i] = applyH(I, seeds[k].quad[i].x, seeds[k].quad[i].y);
      double ox = std::floor(loc[0].x + .5), oz = std::floor(loc[0].y + .5), err = 0;
      for (int i = 0; i < 4; i++)
        err = std::max(err, dist(loc[i], unit[i] + Point(ox, oz)));
      if (err < .2) {
        ids.push_back(k);
        for (int i = 0; i < 4; i++) {
          src.push_back(unit[i] + Point(ox, oz));
          dst.push_back(seeds[k].quad[i]);
        }
      }
    }
    H3 H;
    if (src.size() >= 4 && homography(src, dst, H) && (!best.valid || ids.size() > best.ids.size()))
      best = {true, H, ids};
  }
  return best;
}
struct SharedTrialCache {
  static constexpr int TotalVoxels = 25 * 61 * 61;

  struct PlaneEntry {
    H3 H;
    int range;
    std::vector<Block> cand;
    std::vector<Verdict> verdicts;
  };

  struct CameraEntry {
    Camera P;
    std::vector<Block> cand;
    std::vector<int32_t> flatIdx;
    std::vector<Verdict> verdicts;
  };

  struct CompactRawEntry {
    float q[8];
    float score;
    float margin;
    uint8_t valid;
    uint8_t rot;
    uint8_t votes;
    uint8_t pad;
  };

  struct RawVoxelSlot {
    uint32_t gen = 0;
    uint8_t count = 0;
    uint8_t next = 0;
    uint16_t pad = 0;
    CompactRawEntry entries[2];
  };

  std::vector<PlaneEntry> planeEntries;
  std::vector<CameraEntry> cameraEntries;
  uint32_t gen = 0;

  static std::vector<RawVoxelSlot>& sharedSlots() {
    static thread_local std::vector<RawVoxelSlot> slots(TotalVoxels);
    return slots;
  }

  SharedTrialCache() {
    static thread_local uint32_t globalGen = 0;
    gen = ++globalGen;
    if (gen == 0) {
      auto& slots = sharedSlots();
      for (auto& s : slots)
        s.gen = 0;
      gen = ++globalGen;
    }
  }

  inline bool lookupVoxel(int32_t flatIdx, const Quad& q, Verdict& out) const {
    const auto& slot = sharedSlots()[flatIdx];
    if (slot.gen != gen) return false;
    const float q0x = float(q[0].x), q0y = float(q[0].y);
    const float q1x = float(q[1].x), q1y = float(q[1].y);
    const float q2x = float(q[2].x), q2y = float(q[2].y);
    const float q3x = float(q[3].x), q3y = float(q[3].y);
    for (uint8_t s = 0; s < slot.count; ++s) {
      const auto& e = slot.entries[s];
      float d0 = (q0x - e.q[0]) * (q0x - e.q[0]) + (q0y - e.q[1]) * (q0y - e.q[1]);
      float d1 = (q1x - e.q[2]) * (q1x - e.q[2]) + (q1y - e.q[3]) * (q1y - e.q[3]);
      float d2 = (q2x - e.q[4]) * (q2x - e.q[4]) + (q2y - e.q[5]) * (q2y - e.q[5]);
      float d3 = (q3x - e.q[6]) * (q3x - e.q[6]) + (q3y - e.q[7]) * (q3y - e.q[7]);
      float maxD = std::max(std::max(d0, d1), std::max(d2, d3));
      // Only allow sub-pixel reuse for clear background rejects; re-evaluate any candidate block unless near-exact.
      const float limitSq = e.score < 0.45f ? (0.12f * 0.12f) : 1e-6f;
      if (maxD < limitSq) {
        out = {e.valid != 0, int(e.rot), int(e.votes), double(e.score), double(e.margin)};
        return true;
      }
    }
    return false;
  }

  inline void storeVoxel(int32_t flatIdx, const Quad& q, const Verdict& v) {
    auto& slot = sharedSlots()[flatIdx];
    if (slot.gen != gen) {
      slot.gen = gen;
      slot.count = 0;
      slot.next = 0;
    }
    uint8_t pos = slot.count < 2 ? slot.count++ : slot.next;
    slot.next = (pos + 1) & 1;
    slot.entries[pos] = {{float(q[0].x), float(q[0].y), float(q[1].x), float(q[1].y), float(q[2].x),
                          float(q[2].y), float(q[3].x), float(q[3].y)},
                         float(v.score),
                         float(v.margin),
                         uint8_t(v.valid ? 1 : 0),
                         uint8_t(v.rotation),
                         uint8_t(v.votes),
                         0};
  }
};

std::shared_ptr<SharedTrialCache> makeSharedTrialCache() {
  return std::make_shared<SharedTrialCache>();
}

static std::vector<Block> growPlane(const Context& c, const H3& H, const Refiner& r, int range,
                                    SharedTrialCache* sharedCache = nullptr) {
  const std::vector<Block>* candPtr = nullptr;
  const std::vector<Verdict>* vsPtr = nullptr;
  std::vector<Block> localCand;
  std::vector<Verdict> localVs;

  if (sharedCache) {
    for (const auto& entry : sharedCache->planeEntries) {
      if (entry.range == range && entry.H == H) {
        candPtr = &entry.cand;
        vsPtr = &entry.verdicts;
        break;
      }
    }
  }

  if (!candPtr) {
    const int side = 2 * range + 2;
    const double wm1 = c.w - 1;
    const double hm1 = c.h - 1;
    std::vector<Point> grid(size_t(side) * side);
    std::vector<uint8_t> ok(size_t(side) * side, 0);
    std::vector<uint8_t> rowAny(side, 0);

    for (int iz = 0; iz < side; ++iz) {
      const double Z = iz - range;
      double nx = H[0] * (-range) + H[1] * Z + H[2];
      double ny = H[3] * (-range) + H[4] * Z + H[5];
      double nd = H[6] * (-range) + H[7] * Z + H[8];
      uint8_t any = 0;
      const size_t rowBase = size_t(iz) * side;
      for (int ix = 0; ix < side; ++ix, nx += H[0], ny += H[3], nd += H[6]) {
        if (std::abs(nd) > 1e-12) {
          const double invD = 1.0 / nd;
          const double u = nx * invD;
          const double v = ny * invD;
          if (std::isfinite(u) && std::isfinite(v) && u >= 0.0 && v >= 0.0 && u < wm1 && v < hm1) {
            grid[rowBase + ix] = {u, v};
            ok[rowBase + ix] = 1;
            any = 1;
          }
        }
      }
      rowAny[iz] = any;
    }

    std::vector<Quad> qs;
    for (int iz = 0; iz < side - 1; ++iz) {
      if (!rowAny[iz] || !rowAny[iz + 1]) continue;
      const size_t r0 = size_t(iz) * side;
      const size_t r1 = size_t(iz + 1) * side;
      for (int ix = 0; ix < side - 1; ++ix) {
        if (!(ok[r0 + ix] & ok[r0 + ix + 1] & ok[r1 + ix + 1] & ok[r1 + ix])) continue;
        Quad q = {grid[r0 + ix], grid[r0 + ix + 1], grid[r1 + ix + 1], grid[r1 + ix]};
        if (!convex(q) || minSide(q) < 10.0) continue;
        Block b;
        b.quad = q;
        b.x = ix - range;
        b.z = iz - range;
        localCand.push_back(b);
        qs.push_back(q);
      }
    }

    if (!qs.empty()) localVs = classifyBatchNative(c, qs);
    if (sharedCache && sharedCache->planeEntries.size() < 32) {
      sharedCache->planeEntries.push_back({H, range, std::move(localCand), std::move(localVs)});
      candPtr = &sharedCache->planeEntries.back().cand;
      vsPtr = &sharedCache->planeEntries.back().verdicts;
    } else {
      candPtr = &localCand;
      vsPtr = &localVs;
    }
  }

  std::vector<Block> out;
  const auto& cand = *candPtr;
  const auto& vs = *vsPtr;
  for (size_t i = 0; i < cand.size(); ++i) {
    auto b = verify(c, cand[i].quad, r, vs[i], true);
    if (b.valid && b.score >= c.opts.accept && b.margin >= c.opts.margin && b.votes == 4) {
      b.x = cand[i].x;
      b.z = cand[i].z;
      out.push_back(b);
    }
  }
  return out;
}
static bool cameraFromPlane(const H3& H, int w, int h, Camera& P, double& f) {
  double cx = w / 2., cy = h / 2., v1x = H[0] / H[6], v1y = H[3] / H[6], v2x = H[1] / H[7], v2y = H[4] / H[7];
  if (!std::isfinite(v1x + v1y + v2x + v2y)) return false;
  double f2 = -((v1x - cx) * (v2x - cx) + (v1y - cy) * (v2y - cy));
  if (!(f2 > 100)) return false;
  f = std::sqrt(f2);
  std::array<double, 3> a = {(H[0] - cx * H[6]) / f, (H[3] - cy * H[6]) / f, H[6]},
                        c = {(H[1] - cx * H[7]) / f, (H[4] - cy * H[7]) / f, H[7]},
                        b = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]};
  auto norm = [](auto a) { return std::hypot(a[0], a[1], a[2]); };
  double s = (norm(a) + norm(c)) / 2 / norm(b);
  for (double& v : b)
    v *= s;
  P = {H[0], f * b[0] + cx * b[2], H[1], H[2], H[3], f * b[1] + cy * b[2], H[4], H[5], H[6], b[2], H[7],
       H[8]};
  if ((P[7] + P[5]) / (P[11] + P[9]) > P[7] / P[11])
    for (int i : {1, 5, 9})
      P[i] = -P[i];
  return true;
}
static bool projectQuad(const Camera& P, int x, int y, int z, int w, int h, Quad& q) {
  for (int i = 0; i < 4; i++) {
    double X = x + unit[i].x, Z = z + unit[i].y, d = P[8] * X + P[9] * y + P[10] * Z + P[11];
    if (d <= 0) return false;
    double u = (P[0] * X + P[1] * y + P[2] * Z + P[3]) / d;
    if (!(u >= 0 && u < w - 1)) return false;
    double v = (P[4] * X + P[5] * y + P[6] * Z + P[7]) / d;
    if (!(v >= 0 && v < h - 1)) return false;
    q[i] = {u, v};
  }
  if (!convex(q)) return false;
  for (int i = 0; i < 4; i++) {
    Point d = q[i] - q[(i + 1) % 4];
    if (d.x * d.x + d.y * d.y < 64) return false;
  }
  return true;
}
static void jacobi(std::vector<std::vector<double>> A, int n, std::vector<double>& vals,
                   std::vector<std::vector<double>>& V) {
  V.assign(n, std::vector<double>(n));
  for (int i = 0; i < n; i++)
    V[i][i] = 1;
  for (int sweep = 0; sweep < 60; sweep++) {
    double off = 0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        off += A[i][j] * A[i][j];
    if (off < 1e-24) break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        if (std::abs(A[p][q]) < 1e-30) continue;
        double th = (A[q][q] - A[p][p]) / (2 * A[p][q]),
               t = (th < 0 ? -1. : 1.) / (std::abs(th) + std::sqrt(th * th + 1)),
               c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; k++) {
          double ap = A[k][p], aq = A[k][q];
          A[k][p] = c * ap - s * aq;
          A[k][q] = s * ap + c * aq;
        }
        for (int k = 0; k < n; k++) {
          double ap = A[p][k], aq = A[q][k];
          A[p][k] = c * ap - s * aq;
          A[q][k] = s * ap + c * aq;
        }
        for (int k = 0; k < n; k++) {
          double vp = V[k][p], vq = V[k][q];
          V[k][p] = c * vp - s * vq;
          V[k][q] = s * vp + c * vq;
        }
      }
  }
  vals.resize(n);
  for (int i = 0; i < n; i++)
    vals[i] = A[i][i];
}
static bool dlt(const std::vector<std::array<double, 3>>& pts, const std::vector<Point>& pix, Camera& out) {
  double n = pts.size();
  std::array<double, 3> c3{};
  Point c2{};
  for (auto p : pts)
    for (int i = 0; i < 3; i++)
      c3[i] += p[i] / n;
  for (auto p : pix) {
    c2.x += p.x / n;
    c2.y += p.y / n;
  }
  double d3 = 0, d2 = 0;
  for (auto p : pts)
    d3 += std::hypot(p[0] - c3[0], p[1] - c3[1], p[2] - c3[2]) / n;
  for (auto p : pix)
    d2 += dist(p, c2) / n;
  double s3 = std::sqrt(3) / (d3 ? d3 : 1), s2 = std::sqrt(2) / (d2 ? d2 : 1);
  std::vector<std::vector<double>> S(12, std::vector<double>(12));
  for (size_t i = 0; i < pts.size(); i++) {
    std::array<double, 4> v = {(pts[i][0] - c3[0]) * s3, (pts[i][1] - c3[1]) * s3, (pts[i][2] - c3[2]) * s3,
                               1};
    double u = (pix[i].x - c2.x) * s2, w = (pix[i].y - c2.y) * s2;
    std::array<double, 12> r1{}, r2{};
    for (int k = 0; k < 4; k++) {
      r1[k] = v[k];
      r1[k + 8] = -u * v[k];
      r2[k + 4] = v[k];
      r2[k + 8] = -w * v[k];
    }
    for (auto& r : {r1, r2})
      for (int a = 0; a < 12; a++)
        for (int b = 0; b < 12; b++)
          S[a][b] += r[a] * r[b];
  }
  std::vector<double> vals;
  std::vector<std::vector<double>> vecs;
  jacobi(S, 12, vals, vecs);
  int mi = 0;
  for (int i = 0; i < 12; i++)
    if (vals[i] < vals[mi]) mi = i;
  Camera pn;
  for (int i = 0; i < 12; i++)
    pn[i] = vecs[i][mi];
  double T3[4][4] = {{s3, 0, 0, -s3 * c3[0]}, {0, s3, 0, -s3 * c3[1]}, {0, 0, s3, -s3 * c3[2]}, {0, 0, 0, 1}},
         T2i[3][3] = {{1 / s2, 0, c2.x}, {0, 1 / s2, c2.y}, {0, 0, 1}}, tmp[3][4]{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        tmp[i][j] += pn[i * 4 + k] * T3[k][j];
  out.fill(0);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        out[i * 4 + j] += T2i[i][k] * tmp[k][j];
  double sc = out[11];
  if (std::abs(sc) < 1e-12) return false;
  for (double& v : out)
    v /= sc;
  return true;
}
static double reproj(const Camera& P, const std::vector<std::array<double, 3>>& pts,
                     const std::vector<Point>& pix) {
  double s = 0;
  for (size_t i = 0; i < pts.size(); i++) {
    auto a = pts[i];
    double w = P[8] * a[0] + P[9] * a[1] + P[10] * a[2] + P[11],
           u = (P[0] * a[0] + P[1] * a[1] + P[2] * a[2] + P[3]) / w,
           v = (P[4] * a[0] + P[5] * a[1] + P[6] * a[2] + P[7]) / w;
    s += dist({u, v}, pix[i]);
  }
  return s / pts.size();
}
static uint64_t packVoxel(int x, int y, int z) {
  return ((uint64_t(x + 1024) & 0xFFFFFULL) << 40) | ((uint64_t(y + 1024) & 0xFFFFFULL) << 20) |
         (uint64_t(z + 1024) & 0xFFFFFULL);
}
static bool refineCameraWithOffPlaneSeeds(const Context& c, const Plane& plane,
                                          const std::vector<Block>& planeBlocks,
                                          const std::vector<Block>& seeds, Camera& P) {
  if (planeBlocks.size() < 3 || seeds.empty()) return false;
  H3 planeInv;
  if (!inv3(plane.H, planeInv)) return false;
  std::vector<std::array<double, 3>> p3;
  std::vector<Point> pix;
  for (const auto& b : planeBlocks)
    for (int i = 0; i < 4; i++) {
      p3.push_back({b.x + unit[i].x, 0.0, b.z + unit[i].y});
      pix.push_back(b.quad[i]);
    }
  int added = 0;
  for (const auto& s : seeds) {
    Quad loc;
    for (int i = 0; i < 4; i++)
      loc[i] = applyH(planeInv, s.quad[i].x, s.quad[i].y);
    double ox = std::floor(loc[0].x + .5), oz = std::floor(loc[0].y + .5), onPlaneErr = 0;
    for (int i = 0; i < 4; i++)
      onPlaneErr = std::max(onPlaneErr, dist(loc[i], unit[i] + Point(ox, oz)));
    if (onPlaneErr < 0.25) continue;
    double bestErr = INFINITY, secondErr = INFINITY;
    int bx = 0, by = 0, bz = 0;
    for (int y = -12; y <= 12; y++) {
      if (y == 0) continue;
      H3 Hy = {P[0], P[2], P[3] + P[1] * y, P[4], P[6], P[7] + P[5] * y, P[8], P[10], P[11] + P[9] * y}, Iy;
      if (!inv3(Hy, Iy)) continue;
      Point p0 = applyH(Iy, s.quad[0].x, s.quad[0].y);
      if (!std::isfinite(p0.x) || !std::isfinite(p0.y)) continue;
      int x0 = int(std::llround(p0.x)), z0 = int(std::llround(p0.y));
      for (int dz = -1; dz <= 1; dz++)
        for (int dx = -1; dx <= 1; dx++) {
          int x = x0 + dx, z = z0 + dz;
          if (x < -30 || x > 30 || z < -30 || z > 30) continue;
          Quad q;
          if (!projectQuad(P, x, y, z, c.w, c.h, q)) continue;
          double err = 0;
          for (int i = 0; i < 4; i++)
            err += dist(q[i], s.quad[i]);
          err *= 0.25;
          if (err < bestErr) {
            secondErr = bestErr;
            bestErr = err;
            bx = x;
            by = y;
            bz = z;
          } else if (err < secondErr)
            secondErr = err;
        }
    }
    double scale = minSide(s.quad);
    if (bestErr < std::min(15.0, scale * 0.25) && bestErr < secondErr * 0.55) {
      for (int i = 0; i < 4; i++) {
        p3.push_back({bx + unit[i].x, double(by), bz + unit[i].y});
        pix.push_back(s.quad[i]);
      }
      added++;
    }
  }
  if (!added) return false;
  Camera P2;
  if (!dlt(p3, pix, P2) || !(reproj(P2, p3, pix) < 2.5)) return false;
  P = P2;
  return true;
}
static std::vector<Block> voxel(const Context& c, const Plane& plane, const std::vector<Block>& planeBlocks,
                                const std::vector<Block>& seeds, const Refiner& r, Camera& bestP, bool& has,
                                SharedTrialCache* sharedCache = nullptr) {
  struct CachedVoxel {
    Quad projQ;
    Verdict rawV;
    Block verified;
    bool accepted;
  };

  Camera P;
  double f;
  if (!cameraFromPlane(plane.H, c.w, c.h, P, f)) return {};
  refineCameraWithOffPlaneSeeds(c, plane, planeBlocks, seeds, P);

  constexpr int YMin = -12, YMax = 12, YCount = YMax - YMin + 1;
  constexpr int XZMin = -30, XZMax = 30, XZCount = XZMax - XZMin + 1;
  constexpr int CornerN = XZCount + 1;                     // 62
  constexpr int TotalVoxels = YCount * XZCount * XZCount;  // 93,025

  int prev = -1;
  std::vector<Block> best;
  has = false;

  // Flat direct-indexed voxel cache across rounds (replaces std::unordered_map).
  static thread_local std::vector<int32_t> slotCur(TotalVoxels, -1);
  static thread_local std::vector<int32_t> slotNext(TotalVoxels, -1);
  static thread_local std::vector<CachedVoxel> poolCur;
  static thread_local std::vector<CachedVoxel> poolNext;
  static thread_local std::vector<int32_t> touchedCur;
  static thread_local std::vector<int32_t> touchedNext;
  static thread_local Point grid[CornerN][CornerN];
  static thread_local uint8_t okGrid[CornerN][CornerN];
  uint8_t rowAny[CornerN];

  for (int32_t idx : touchedCur)
    slotCur[idx] = -1;
  touchedCur.clear();
  poolCur.clear();
  for (int32_t idx : touchedNext)
    slotNext[idx] = -1;
  touchedNext.clear();
  poolNext.clear();
  if (poolCur.capacity() < 8192) {
    poolCur.reserve(8192);
    poolNext.reserve(8192);
    touchedCur.reserve(8192);
    touchedNext.reserve(8192);
  }

  const double wm1 = c.w - 1;
  const double hm1 = c.h - 1;
  constexpr double DriftSqLimit = 0.15 * 0.15;

  for (int round = 0; round < 4; ++round) {
    std::vector<Block> accept, kept;
    poolNext.clear();
    for (int32_t idx : touchedNext)
      slotNext[idx] = -1;
    touchedNext.clear();

    // Level 1 cross-trial cache: exact Camera P hit reuses all projected quads & raw verdicts directly.
    const SharedTrialCache::CameraEntry* camHit = nullptr;
    if (sharedCache) {
      for (const auto& entry : sharedCache->cameraEntries) {
        if (entry.P == P) {
          camHit = &entry;
          break;
        }
      }
    }

    if (camHit) {
      const auto& allCand = camHit->cand;
      const auto& allFlat = camHit->flatIdx;
      const auto& allVs = camHit->verdicts;
      for (size_t i = 0; i < allCand.size(); ++i) {
        const int32_t flatIdx = allFlat[i];
        const int32_t cachedSlot = slotCur[flatIdx];
        if (cachedSlot >= 0) {
          const CachedVoxel& cv = poolCur[cachedSlot];
          double maxDriftSq = 0.0;
          for (int k = 0; k < 4; ++k) {
            const double dx = allCand[i].quad[k].x - cv.projQ[k].x;
            const double dy = allCand[i].quad[k].y - cv.projQ[k].y;
            const double d2 = dx * dx + dy * dy;
            if (d2 > maxDriftSq) maxDriftSq = d2;
          }
          if (maxDriftSq < DriftSqLimit) {
            slotNext[flatIdx] = int32_t(poolNext.size());
            touchedNext.push_back(flatIdx);
            poolNext.push_back(cv);
            if (cv.accepted) accept.push_back(cv.verified);
            continue;
          }
        }
        auto b = verify(c, allCand[i].quad, r, allVs[i], true);
        b.x = allCand[i].x;
        b.y = allCand[i].y;
        b.z = allCand[i].z;
        const bool ok = b.valid && b.score > c.opts.relaxed && b.margin > c.opts.margin && b.votes == 4;
        slotNext[flatIdx] = int32_t(poolNext.size());
        touchedNext.push_back(flatIdx);
        poolNext.push_back(CachedVoxel{allCand[i].quad, allVs[i], b, ok});
        if (ok) accept.push_back(b);
      }
    } else {
      std::vector<Block> cand;
      std::vector<Quad> qs;
      std::vector<int32_t> candFlatIdx;
      // Record full raw camera result across all rounds (up to 48 distinct cameras).
      const bool recordCamera = sharedCache && sharedCache->cameraEntries.size() < 48;
      std::vector<Block> camCand;
      std::vector<int32_t> camFlat;
      std::vector<Verdict> camVs;
      std::vector<int32_t> camBatchPos;

      std::vector<Block> crossHitCand;
      std::vector<int32_t> crossHitFlat;
      std::vector<Verdict> crossHitVs;

      for (int iy = 0; iy < YCount; ++iy) {
        const int y = iy + YMin;
        const double p1y = P[1] * y + P[3];
        const double p5y = P[5] * y + P[7];
        const double p9y = P[9] * y + P[11];

        // Project shared 62x62 corner grid for this y-slice once (3.87x fewer projections).
        bool sliceAny = false;
        for (int iz = 0; iz < CornerN; ++iz) {
          const double Z = iz + XZMin;
          double nd0 = P[8] * XZMin + P[10] * Z + p9y;
          double ndEnd = nd0 + P[8] * (CornerN - 1);
          if (nd0 <= 0.0 && ndEnd <= 0.0) {
            rowAny[iz] = 0;
            continue;
          }
          double nx = P[0] * XZMin + P[2] * Z + p1y;
          double ny = P[4] * XZMin + P[6] * Z + p5y;
          double nd = nd0;
          uint8_t any = 0;
          for (int ix = 0; ix < CornerN; ++ix, nx += P[0], ny += P[4], nd += P[8]) {
            uint8_t ok = 0;
            if (nd > 0.0) {
              const double invD = 1.0 / nd;
              const double u = nx * invD;
              if (u >= 0.0 && u < wm1) {
                const double v = ny * invD;
                if (v >= 0.0 && v < hm1) {
                  grid[iz][ix] = {u, v};
                  ok = 1;
                  any = 1;
                }
              }
            }
            okGrid[iz][ix] = ok;
          }
          rowAny[iz] = any;
          sliceAny |= (any != 0);
        }

        if (!sliceAny) continue;

        const int32_t yOffset = iy * XZCount * XZCount;
        for (int iz = 0; iz < XZCount; ++iz) {
          if (!rowAny[iz] || !rowAny[iz + 1]) continue;
          const int z = iz + XZMin;
          const int32_t yzOffset = yOffset + iz * XZCount;
          for (int ix = 0; ix < XZCount; ++ix) {
            if (!(okGrid[iz][ix] & okGrid[iz][ix + 1] & okGrid[iz + 1][ix + 1] & okGrid[iz + 1][ix])) {
              continue;
            }
            Quad q = {grid[iz][ix], grid[iz][ix + 1], grid[iz + 1][ix + 1], grid[iz + 1][ix]};
            if (!convex(q)) continue;
            bool shortSide = false;
            for (int k = 0; k < 4; ++k) {
              const Point& a = q[k];
              const Point& b = q[(k + 1) & 3];
              const double dx = a.x - b.x;
              const double dy = a.y - b.y;
              if (dx * dx + dy * dy < 64.0) {
                shortSide = true;
                break;
              }
            }
            if (shortSide) continue;

            const int32_t flatIdx = yzOffset + ix;
            const int32_t cachedSlot = slotCur[flatIdx];
            if (cachedSlot >= 0) {
              const CachedVoxel& cv = poolCur[cachedSlot];
              double maxDriftSq = 0.0;
              for (int k = 0; k < 4; ++k) {
                const double dx = q[k].x - cv.projQ[k].x;
                const double dy = q[k].y - cv.projQ[k].y;
                const double d2 = dx * dx + dy * dy;
                if (d2 > maxDriftSq) maxDriftSq = d2;
              }
              if (maxDriftSq < DriftSqLimit) {
                slotNext[flatIdx] = int32_t(poolNext.size());
                touchedNext.push_back(flatIdx);
                poolNext.push_back(cv);
                if (recordCamera) {
                  Block rb;
                  rb.quad = q;
                  rb.x = ix + XZMin;
                  rb.y = y;
                  rb.z = z;
                  camCand.push_back(rb);
                  camFlat.push_back(flatIdx);
                  camVs.push_back(cv.rawV);
                  camBatchPos.push_back(-1);
                }
                if (cv.accepted) accept.push_back(cv.verified);
                continue;
              }
            }

            Block b;
            b.quad = q;
            b.x = ix + XZMin;
            b.y = y;
            b.z = z;

            // Level 2 cross-trial sequential voxel cache (zero allocation, L1/L2 prefetch friendly).
            if (sharedCache) {
              Verdict rawV;
              if (sharedCache->lookupVoxel(flatIdx, q, rawV)) {
                crossHitCand.push_back(b);
                crossHitFlat.push_back(flatIdx);
                crossHitVs.push_back(rawV);
                if (recordCamera) {
                  camCand.push_back(b);
                  camFlat.push_back(flatIdx);
                  camVs.push_back(rawV);
                  camBatchPos.push_back(-1);
                }
                continue;
              }
            }

            if (recordCamera) {
              camCand.push_back(b);
              camFlat.push_back(flatIdx);
              camVs.push_back(Verdict{});
              camBatchPos.push_back(int32_t(qs.size()));
            }
            cand.push_back(b);
            qs.push_back(q);
            candFlatIdx.push_back(flatIdx);
          }
        }
      }

      auto vs = qs.empty() ? std::vector<Verdict>{} : classifyBatchNative(c, qs);
      for (size_t i = 0; i < cand.size(); ++i) {
        const int32_t flatIdx = candFlatIdx[i];
        if (sharedCache) sharedCache->storeVoxel(flatIdx, cand[i].quad, vs[i]);
        auto b = verify(c, cand[i].quad, r, vs[i], true);
        b.x = cand[i].x;
        b.y = cand[i].y;
        b.z = cand[i].z;
        const bool ok = b.valid && b.score > c.opts.relaxed && b.margin > c.opts.margin && b.votes == 4;
        slotNext[flatIdx] = int32_t(poolNext.size());
        touchedNext.push_back(flatIdx);
        poolNext.push_back(CachedVoxel{cand[i].quad, vs[i], b, ok});
        if (ok) accept.push_back(b);
      }

      for (size_t i = 0; i < crossHitCand.size(); ++i) {
        const int32_t flatIdx = crossHitFlat[i];
        auto b = verify(c, crossHitCand[i].quad, r, crossHitVs[i], true);
        b.x = crossHitCand[i].x;
        b.y = crossHitCand[i].y;
        b.z = crossHitCand[i].z;
        const bool ok = b.valid && b.score > c.opts.relaxed && b.margin > c.opts.margin && b.votes == 4;
        slotNext[flatIdx] = int32_t(poolNext.size());
        touchedNext.push_back(flatIdx);
        poolNext.push_back(CachedVoxel{crossHitCand[i].quad, crossHitVs[i], b, ok});
        if (ok) accept.push_back(b);
      }

      if (recordCamera) {
        for (size_t i = 0; i < camBatchPos.size(); ++i) {
          if (camBatchPos[i] >= 0) camVs[i] = vs[camBatchPos[i]];
        }
        sharedCache->cameraEntries.push_back({P, std::move(camCand), std::move(camFlat), std::move(camVs)});
      }
    }

    slotCur.swap(slotNext);
    poolCur.swap(poolNext);
    touchedCur.swap(touchedNext);

    std::stable_sort(accept.begin(), accept.end(),
                     [](const Block& a, const Block& b) { return a.score > b.score; });

    std::vector<Point> centers;
    std::vector<double> sides;
    centers.reserve(accept.size());
    sides.reserve(accept.size());
    for (const auto& b : accept) {
      Point center{0.0, 0.0};
      for (const auto& p : b.quad) {
        center.x += p.x * 0.25;
        center.y += p.y * 0.25;
      }
      const double sd = minSide(b.quad);
      bool overlap = false;
      for (size_t i = 0; i < kept.size(); ++i) {
        if (dist(centers[i], center) < 0.5 * std::min(sd, sides[i])) {
          overlap = true;
          break;
        }
      }
      if (!overlap) {
        kept.push_back(b);
        centers.push_back(center);
        sides.push_back(sd);
      }
    }

    std::set<int> heights;
    for (const auto& b : kept)
      heights.insert(b.y);
    if (!has || kept.size() > best.size()) {
      best = kept;
      bestP = P;
      has = true;
    }
    if (int(kept.size()) <= prev || kept.size() < size_t(c.opts.refitFlat && heights.size() == 1 ? 3 : 6)) {
      break;
    }
    prev = int(kept.size());

    if (heights.size() == 1) {
      if (!c.opts.refitFlat) break;
      std::vector<Point> src, dst;
      for (const auto& b : kept) {
        for (int i = 0; i < 4; ++i) {
          src.push_back(unit[i] + Point(b.x, b.z));
          dst.push_back(b.quad[i]);
        }
      }
      H3 H;
      Camera flat;
      if (!homography(src, dst, H) || !cameraFromPlane(H, c.w, c.h, flat, f)) break;
      P = flat;
      const int yLevel = *heights.begin();
      for (int i : {0, 4, 8})
        P[i + 3] -= P[i + 1] * yLevel;
      continue;
    }

    std::vector<std::array<double, 3>> p3;
    std::vector<Point> pix;
    for (const auto& b : kept) {
      for (int i = 0; i < 4; ++i) {
        p3.push_back({b.x + unit[i].x, double(b.y), b.z + unit[i].y});
        pix.push_back(b.quad[i]);
      }
    }
    Camera p2;
    if (!dlt(p3, pix, p2) || !(reproj(p2, p3, pix) < 3.0)) break;
    P = p2;
  }
  return best;
}

// In-place iterative radix-2 Cooley-Tukey FFT on complex interleaved (re, im) arrays.
static void fftRadix2(std::vector<double>& re, std::vector<double>& im, bool inverse) {
  const size_t n = re.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  const double pi = std::acos(-1.0);
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = (inverse ? 2.0 : -2.0) * pi / double(len);
    const double wlenRe = std::cos(ang);
    const double wlenIm = std::sin(ang);
    const size_t half = len >> 1;
    for (size_t i = 0; i < n; i += len) {
      double wRe = 1.0, wIm = 0.0;
      for (size_t j = 0; j < half; ++j) {
        const size_t u = i + j;
        const size_t v = u + half;
        const double vr = re[v] * wRe - im[v] * wIm;
        const double vi = re[v] * wIm + im[v] * wRe;
        re[v] = re[u] - vr;
        im[v] = im[u] - vi;
        re[u] += vr;
        im[u] += vi;
        const double nextRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextRe;
      }
    }
  }
  if (inverse) {
    const double invN = 1.0 / double(n);
    for (size_t i = 0; i < n; ++i) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}

// Exact arbitrary-length DFT power spectrum via Bluestein's chirp-z transform in O(M log M).
static std::vector<double> dftPowerSpectrum(const std::vector<double>& x) {
  const size_t m = x.size();
  size_t L = 1;
  while (L < 2 * m - 1)
    L <<= 1;

  const double pi = std::acos(-1.0);
  const uint64_t twoM = 2ULL * m;
  std::vector<double> aRe(L, 0.0), aIm(L, 0.0);
  std::vector<double> bRe(L, 0.0), bIm(L, 0.0);

  bRe[0] = 1.0;
  for (size_t i = 0; i < m; ++i) {
    const double ang = pi * double((uint64_t(i) * uint64_t(i)) % twoM) / double(m);
    const double wr = std::cos(ang);
    const double wi = std::sin(ang);
    aRe[i] = x[i] * wr;
    aIm[i] = x[i] * wi;
    if (i > 0) {
      bRe[i] = wr;
      bIm[i] = -wi;
      bRe[L - i] = wr;
      bIm[L - i] = -wi;
    }
  }

  fftRadix2(aRe, aIm, false);
  fftRadix2(bRe, bIm, false);
  for (size_t i = 0; i < L; ++i) {
    const double rr = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    const double ri = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = rr;
    aIm[i] = ri;
  }
  fftRadix2(aRe, aIm, true);

  // |X[k]|^2 = |a[k] * w[k]|^2 = |a[k]|^2 because |w[k]| == 1.
  std::vector<double> power(m);
  for (size_t k = 0; k < m; ++k) {
    power[k] = aRe[k] * aRe[k] + aIm[k] * aIm[k];
  }
  return power;
}

static std::vector<Block> topDown(const Context& c) {
  std::vector<double> periods;
  const uint8_t* g = c.g.data();
  for (int axis = 0; axis < 2; ++axis) {
    const int n = axis ? c.h : c.w;
    const int other = axis ? c.w : c.h;
    std::vector<double> e(n - 1, 0.0);
    if (axis == 0) {
      for (int j = 0; j < other; j += 3) {
        const uint8_t* row = g + size_t(j) * c.w;
        for (int i = 0; i < n - 1; ++i) {
          e[i] += std::abs(int(row[i + 1]) - int(row[i]));
        }
      }
    } else {
      for (int i = 0; i < n - 1; ++i) {
        const uint8_t* r0 = g + size_t(i) * c.w;
        const uint8_t* r1 = r0 + c.w;
        double acc = 0.0;
        for (int j = 0; j < other; j += 3) {
          acc += std::abs(int(r1[j]) - int(r0[j]));
        }
        e[i] = acc;
      }
    }

    const double mean = std::accumulate(e.begin(), e.end(), 0.0) / double(e.size());
    for (double& v : e)
      v -= mean;

    const std::vector<double> power = dftPowerSpectrum(e);
    std::vector<std::pair<double, double>> pk;
    const int kStart = int(std::ceil(n / 24.0));
    const int kEnd = int(std::ceil(n / 2.0));
    pk.reserve(std::max(0, kEnd - kStart));
    for (int k = kStart; k < n / 2.0; ++k) {
      pk.push_back({double(e.size()) / double(k) * 16.0, power[k]});
    }
    std::stable_sort(pk.begin(), pk.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < std::min(size_t(2), pk.size()); ++i) {
      periods.push_back(pk[i].first);
    }
  }

  std::vector<int> sizes;
  const double maxSide = std::min(c.w, c.h) * 0.6;
  for (double p : periods) {
    for (int d : {-1, 0, 1}) {
      const int n = int(std::floor(p + 0.5)) + d;
      if (n >= 48 && n < maxSide && std::find(sizes.begin(), sizes.end(), n) == sizes.end()) {
        sizes.push_back(n);
      }
    }
  }

  // Precompute zero-mean unit-norm 14x14 reference templates for fast top-down square scanning.
  float tRef[4][196];
  for (int k = 0; k < 4; ++k) {
    double sb = 0.0, bb = 0.0;
    int p = 0;
    for (int j = 1; j < 15; ++j) {
      for (int i = 1; i < 15; ++i) {
        const double b = c.refs[k][j * 16 + i];
        sb += b;
        bb += b * b;
        ++p;
      }
    }
    const double meanB = sb / 196.0;
    const double vb = bb - sb * sb / 196.0;
    const double invSqrtVb = vb < 1e-9 ? 0.0 : 1.0 / std::sqrt(vb);
    p = 0;
    for (int j = 1; j < 15; ++j) {
      for (int i = 1; i < 15; ++i) {
        tRef[k][p++] = static_cast<float>((c.refs[k][j * 16 + i] - meanB) * invSqrtVb);
      }
    }
  }

  Block best;
  Quad bestQuad{};
  bool hasBest = false;
  double bestScore = 0.72;

  const int rowStride = c.w + 8;
  std::vector<float> vRowTmp(rowStride, 0.0f);
  std::vector<float> phaseBuf;

  for (int n : sizes) {
    const int st = std::max(2, n / 32);
    int dOff[14];
    float fOff[14];
    int qA[14], rA[14], qB[14], rB[14];
    bool allZeroFx = true;
    for (int i = 0; i < 14; ++i) {
      const double pos = (double(i + 1) + 0.5) * double(n) / 16.0;
      const int ip = int(std::floor(pos));
      dOff[i] = ip;
      fOff[i] = float(pos - ip);
      if (fOff[i] != 0.0f) allZeroFx = false;
      qA[i] = ip / st;
      rA[i] = ip % st;
      qB[i] = (ip + 1) / st;
      rB[i] = (ip + 1) % st;
    }

    const int maxY = c.h - n;
    const int maxX = c.w - n;
    if (maxY <= 0 || maxX <= 0) continue;
    const int numK = (maxX + st - 1) / st;
    const int phaseLen = (c.w + st - 1) / st + 8;
    const size_t rowPhaseStride = size_t(st) * phaseLen;
    phaseBuf.assign(size_t(14) * rowPhaseStride, 0.0f);

    int offA[14], offB[14];
    for (int i = 0; i < 14; ++i) {
      offA[i] = rA[i] * phaseLen + qA[i];
      offB[i] = rB[i] * phaseLen + qB[i];
    }

    for (int y = 0; y < maxY; y += st) {
      // Vertically interpolate the 14 sampled image rows and deinterleave by stride phase.
      for (int j = 0; j < 14; ++j) {
        const uint8_t* r0 = g + size_t(y + dOff[j]) * c.w;
        const uint8_t* r1 = r0 + c.w;
        float* tmp = vRowTmp.data();
        const float fy = fOff[j];
        int X = 0;
#ifdef __wasm_simd128__
        const v128_t vfy = wasm_f32x4_splat(fy);
        const v128_t v128 = wasm_f32x4_splat(128.0f);
        if (fy == 0.0f) {
          for (; X + 8 <= c.w; X += 8) {
            v128_t a16 = wasm_u16x8_load8x8(r0 + X);
            wasm_v128_store(tmp + X,
                            wasm_f32x4_sub(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(a16)), v128));
            wasm_v128_store(
                tmp + X + 4,
                wasm_f32x4_sub(wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(a16)), v128));
          }
        } else {
          for (; X + 8 <= c.w; X += 8) {
            v128_t a16 = wasm_u16x8_load8x8(r0 + X);
            v128_t b16 = wasm_u16x8_load8x8(r1 + X);
            v128_t aLo = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(a16));
            v128_t aHi = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(a16));
            v128_t bLo = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(b16));
            v128_t bHi = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(b16));
            wasm_v128_store(
                tmp + X,
                wasm_f32x4_sub(wasm_f32x4_add(aLo, wasm_f32x4_mul(vfy, wasm_f32x4_sub(bLo, aLo))), v128));
            wasm_v128_store(
                tmp + X + 4,
                wasm_f32x4_sub(wasm_f32x4_add(aHi, wasm_f32x4_mul(vfy, wasm_f32x4_sub(bHi, aHi))), v128));
          }
        }
#endif
        for (; X < c.w; ++X) {
          tmp[X] = (float(r0[X]) + fy * (float(r1[X]) - float(r0[X]))) - 128.0f;
        }

        float* pBase = phaseBuf.data() + size_t(j) * rowPhaseStride;
        if (st == 2) {
          float* p0 = pBase;
          float* p1 = pBase + phaseLen;
          int k = 0;
#ifdef __wasm_simd128__
          for (; (k + 4) * 2 <= c.w; k += 4) {
            v128_t v0 = wasm_v128_load(tmp + k * 2);
            v128_t v1 = wasm_v128_load(tmp + k * 2 + 4);
            wasm_v128_store(p0 + k, wasm_i32x4_shuffle(v0, v1, 0, 2, 4, 6));
            wasm_v128_store(p1 + k, wasm_i32x4_shuffle(v0, v1, 1, 3, 5, 7));
          }
#endif
          for (; k * 2 < c.w; ++k) {
            p0[k] = tmp[k * 2];
            if (k * 2 + 1 < c.w) p1[k] = tmp[k * 2 + 1];
          }
        } else {
          for (int r = 0; r < st; ++r) {
            float* pr = pBase + size_t(r) * phaseLen;
            for (int k = 0, idx = r; idx < c.w; ++k, idx += st) {
              pr[k] = tmp[idx];
            }
          }
        }
      }

      int k = 0;
#ifdef __wasm_simd128__
      const v128_t vInv196 = wasm_f32x4_splat(1.0f / 196.0f);
      const v128_t vEps = wasm_f32x4_splat(1e-9f);
      v128_t vBest = wasm_f32x4_splat(float(bestScore));

      // Process 8 quads (2 x f32x4) per inner pass to break FMA latency and amortize template splats.
      for (; k + 8 <= numK; k += 8) {
        v128_t suA = wasm_f32x4_splat(0.0f), suB = wasm_f32x4_splat(0.0f);
        v128_t suuA = wasm_f32x4_splat(0.0f), suuB = wasm_f32x4_splat(0.0f);
        v128_t c0A = wasm_f32x4_splat(0.0f), c0B = wasm_f32x4_splat(0.0f);
        v128_t c1A = wasm_f32x4_splat(0.0f), c1B = wasm_f32x4_splat(0.0f);
        v128_t c2A = wasm_f32x4_splat(0.0f), c2B = wasm_f32x4_splat(0.0f);
        v128_t c3A = wasm_f32x4_splat(0.0f), c3B = wasm_f32x4_splat(0.0f);
        int p = 0;
        if (allZeroFx) {
          for (int j = 0; j < 14; ++j) {
            const float* rowP = phaseBuf.data() + size_t(j) * rowPhaseStride + k;
            for (int i = 0; i < 14; ++i, ++p) {
              const float* pa = rowP + offA[i];
              const v128_t uA = wasm_v128_load(pa);
              const v128_t uB = wasm_v128_load(pa + 4);
              const v128_t t0 = wasm_f32x4_splat(tRef[0][p]);
              const v128_t t1 = wasm_f32x4_splat(tRef[1][p]);
              const v128_t t2 = wasm_f32x4_splat(tRef[2][p]);
              const v128_t t3 = wasm_f32x4_splat(tRef[3][p]);
              suA = wasm_f32x4_add(suA, uA);
              suB = wasm_f32x4_add(suB, uB);
              suuA = wasm_f32x4_add(suuA, wasm_f32x4_mul(uA, uA));
              suuB = wasm_f32x4_add(suuB, wasm_f32x4_mul(uB, uB));
              c0A = wasm_f32x4_add(c0A, wasm_f32x4_mul(uA, t0));
              c0B = wasm_f32x4_add(c0B, wasm_f32x4_mul(uB, t0));
              c1A = wasm_f32x4_add(c1A, wasm_f32x4_mul(uA, t1));
              c1B = wasm_f32x4_add(c1B, wasm_f32x4_mul(uB, t1));
              c2A = wasm_f32x4_add(c2A, wasm_f32x4_mul(uA, t2));
              c2B = wasm_f32x4_add(c2B, wasm_f32x4_mul(uB, t2));
              c3A = wasm_f32x4_add(c3A, wasm_f32x4_mul(uA, t3));
              c3B = wasm_f32x4_add(c3B, wasm_f32x4_mul(uB, t3));
            }
          }
        } else {
          for (int j = 0; j < 14; ++j) {
            const float* rowP = phaseBuf.data() + size_t(j) * rowPhaseStride + k;
            for (int i = 0; i < 14; ++i, ++p) {
              const float* pa = rowP + offA[i];
              const float* pb = rowP + offB[i];
              const v128_t aA = wasm_v128_load(pa);
              const v128_t aB = wasm_v128_load(pa + 4);
              const v128_t bA = wasm_v128_load(pb);
              const v128_t bB = wasm_v128_load(pb + 4);
              const v128_t vfx = wasm_f32x4_splat(fOff[i]);
              const v128_t uA = wasm_f32x4_add(aA, wasm_f32x4_mul(vfx, wasm_f32x4_sub(bA, aA)));
              const v128_t uB = wasm_f32x4_add(aB, wasm_f32x4_mul(vfx, wasm_f32x4_sub(bB, aB)));
              const v128_t t0 = wasm_f32x4_splat(tRef[0][p]);
              const v128_t t1 = wasm_f32x4_splat(tRef[1][p]);
              const v128_t t2 = wasm_f32x4_splat(tRef[2][p]);
              const v128_t t3 = wasm_f32x4_splat(tRef[3][p]);
              suA = wasm_f32x4_add(suA, uA);
              suB = wasm_f32x4_add(suB, uB);
              suuA = wasm_f32x4_add(suuA, wasm_f32x4_mul(uA, uA));
              suuB = wasm_f32x4_add(suuB, wasm_f32x4_mul(uB, uB));
              c0A = wasm_f32x4_add(c0A, wasm_f32x4_mul(uA, t0));
              c0B = wasm_f32x4_add(c0B, wasm_f32x4_mul(uB, t0));
              c1A = wasm_f32x4_add(c1A, wasm_f32x4_mul(uA, t1));
              c1B = wasm_f32x4_add(c1B, wasm_f32x4_mul(uB, t1));
              c2A = wasm_f32x4_add(c2A, wasm_f32x4_mul(uA, t2));
              c2B = wasm_f32x4_add(c2B, wasm_f32x4_mul(uB, t2));
              c3A = wasm_f32x4_add(c3A, wasm_f32x4_mul(uA, t3));
              c3B = wasm_f32x4_add(c3B, wasm_f32x4_mul(uB, t3));
            }
          }
        }

        v128_t vaA = wasm_f32x4_sub(suuA, wasm_f32x4_mul(wasm_f32x4_mul(suA, suA), vInv196));
        v128_t vaB = wasm_f32x4_sub(suuB, wasm_f32x4_mul(wasm_f32x4_mul(suB, suB), vInv196));
        v128_t maxCA = wasm_f32x4_pmax(wasm_f32x4_pmax(c0A, c1A), wasm_f32x4_pmax(c2A, c3A));
        v128_t maxCB = wasm_f32x4_pmax(wasm_f32x4_pmax(c0B, c1B), wasm_f32x4_pmax(c2B, c3B));
        v128_t scA = wasm_f32x4_div(maxCA, wasm_f32x4_sqrt(wasm_f32x4_pmax(vaA, vEps)));
        v128_t scB = wasm_f32x4_div(maxCB, wasm_f32x4_sqrt(wasm_f32x4_pmax(vaB, vEps)));
        v128_t betA = wasm_v128_and(wasm_f32x4_ge(vaA, vEps), wasm_f32x4_gt(scA, vBest));
        v128_t betB = wasm_v128_and(wasm_f32x4_ge(vaB, vEps), wasm_f32x4_gt(scB, vBest));
        if (__builtin_expect(wasm_v128_any_true(wasm_v128_or(betA, betB)), 0)) {
          float sc[8];
          wasm_v128_store(sc, scA);
          wasm_v128_store(sc + 4, scB);
          int mask = wasm_i32x4_bitmask(betA) | (wasm_i32x4_bitmask(betB) << 4);
          for (int lane = 0; lane < 8; ++lane) {
            if ((mask & (1 << lane)) && double(sc[lane]) > bestScore) {
              bestScore = double(sc[lane]);
              const int bx = (k + lane) * st;
              bestQuad = {Point(bx, y), Point(bx + n, y), Point(bx + n, y + n), Point(bx, y + n)};
              hasBest = true;
              vBest = wasm_f32x4_splat(float(bestScore));
            }
          }
        }
      }
#endif
      for (; k < numK; ++k) {
        float su = 0.0f, suu = 0.0f;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        int p = 0;
        for (int j = 0; j < 14; ++j) {
          const float* rowP = phaseBuf.data() + size_t(j) * rowPhaseStride;
          for (int i = 0; i < 14; ++i, ++p) {
            const float a = rowP[size_t(rA[i]) * phaseLen + k + qA[i]];
            const float b = rowP[size_t(rB[i]) * phaseLen + k + qB[i]];
            const float u = a + fOff[i] * (b - a);
            su += u;
            suu += u * u;
            c0 += u * tRef[0][p];
            c1 += u * tRef[1][p];
            c2 += u * tRef[2][p];
            c3 += u * tRef[3][p];
          }
        }
        const float va = suu - (su * su) * (1.0f / 196.0f);
        if (va >= 1e-9f) {
          const float maxC = std::max(std::max(c0, c1), std::max(c2, c3));
          const double sc = double(maxC) / std::sqrt(double(va));
          if (sc > bestScore) {
            bestScore = sc;
            const int bx = k * st;
            bestQuad = {Point(bx, y), Point(bx + n, y), Point(bx + n, y + n), Point(bx, y + n)};
            hasBest = true;
          }
        }
      }
    }
  }
  if (hasBest) {
    auto r = classify(c, bestQuad);
    if (r.valid && r.score > 0.72) {
      static_cast<Verdict&>(best) = r;
      best.quad = bestQuad;
    }
  }
  return best.valid ? std::vector<Block>{best} : std::vector<Block>{};
}

TopDownCache makeTopDownCache(const Context& c) {
  struct State {
    bool computed = false;
    std::vector<Block> blocks;
  };
  auto state = std::make_shared<State>();
  return [&c, state]() -> std::vector<Block> {
    if (!state->computed) {
      state->blocks = topDown(c);
      state->computed = true;
    }
    return state->blocks;
  };
}

Result runDetection(Context& c, const std::vector<Quad>& raw, bool manual, Timings& t,
                    const Refiner* sharedRefine, const TopDownCache* sharedTopDown,
                    SharedTrialCache* sharedCache) {
  Result result;
  Refiner baseRefine = (sharedRefine && *sharedRefine) ? *sharedRefine : makeEccRefiner(c);
  Refiner refine = [&](const Quad& q, int rot, Quad& out) {
    double start = clockMs();
    bool ok = baseRefine(q, rot, out);
    t.ecc += clockMs() - start;
    return ok;
  };

  double start = clockMs();
  for (const auto& q : raw) {
    auto b = verify(c, q, refine, {});
    if (b.valid && b.score >= c.opts.accept && b.margin >= c.opts.margin && b.votes == 4) {
      result.seeds.push_back(b);
    }
  }
  if (result.seeds.empty() && !manual) {
    double fs = clockMs();
    auto seeds = (sharedTopDown && *sharedTopDown) ? (*sharedTopDown)() : topDown(c);
    t.latticeFallback += clockMs() - fs;
    for (const auto& b : seeds) {
      if (b.score >= c.opts.accept) result.seeds.push_back(b);
    }
  }
  t.verify += clockMs() - start;

  if (result.seeds.empty()) {
    result.reason =
        manual ? "your quad did not verify as a grass top (correlation too low). Make the corners tighter."
               : "no reliable grass tile found automatically. Draw one tile quad by hand.";
    return result;
  }

  start = clockMs();
  Plane plane = planeFromSeeds(result.seeds);
  if (!plane.valid) {
    result.reason = "seeds do not share a plane lattice";
    return result;
  }
  auto blocks = growPlane(c, plane.H, refine, 28, sharedCache);
  result.mode = "planar lattice (all y=0)";
  if (c.opts.refitPlane) {
    for (int round = 0; round < 3 && blocks.size() >= 3; ++round) {
      std::vector<Point> src, dst;
      for (const auto& b : blocks) {
        for (int i = 0; i < 4; ++i) {
          src.push_back(unit[i] + Point(b.x, b.z));
          dst.push_back(b.quad[i]);
        }
      }
      H3 H;
      if (!homography(src, dst, H)) break;
      auto next = growPlane(c, H, refine, 28, sharedCache);
      if (next.size() <= blocks.size()) break;
      blocks = next;
      plane.H = H;
    }
  }
  t.plane += clockMs() - start;

  start = clockMs();
  bool has = false;
  Camera cam;
  auto v = voxel(c, plane, blocks, result.seeds, refine, cam, has, sharedCache);
  if (has && v.size() >= blocks.size()) {
    blocks = v;
    result.camera = cam;
    result.hasCamera = true;
    std::set<int> heights;
    for (const auto& b : blocks)
      heights.insert(b.y);
    result.mode = "3D voxel lattice (heights ";
    for (int h : heights) {
      if (result.mode.back() != ' ') result.mode += ",";
      result.mode += std::to_string(h);
    }
    result.mode += ")";
  }
  t.voxel += clockMs() - start;

  std::stable_sort(blocks.begin(), blocks.end(),
                   [](const Block& a, const Block& b) { return a.score > b.score; });
  result.blocks = blocks;
  result.planeH = plane.H;
  result.hasPlane = true;
  result.reason = blocks.empty() ? "seed found but no neighbours verified" : "";
  return result;
}
std::vector<Block> mergeAligned(const std::vector<Block>& a, const std::vector<Block>& incoming) {
  using Key = std::array<int, 3>;
  std::vector<Key> evidence, order;
  std::map<Key, int> votes;
  for (auto& b : incoming) {
    const Block* nearest = nullptr;
    double best = INFINITY;
    for (auto& o : a) {
      double scale = minSide(o.quad), err = 0;
      for (int i = 0; i < 4; i++)
        err += dist(b.quad[i], o.quad[i]);
      err /= 4;
      if (err < std::max(3., scale * .12) && err < best) {
        best = err;
        nearest = &o;
      }
    }
    if (!nearest) continue;
    if (nearest->rotation != b.rotation) return a;
    Key key = {nearest->x - b.x, nearest->y - b.y, nearest->z - b.z};
    evidence.push_back(key);
    if (!votes.count(key)) order.push_back(key);
    votes[key]++;
  }
  if (order.empty()) return a;
  Key winner = order[0];
  for (auto key : order)
    if (votes[key] > votes[winner]) winner = key;
  int minVotes = std::min<int>(3, int(a.size()));
  if (a.size() >= 3) {
    if (votes[winner] < 3 || votes[winner] < evidence.size() * .9) return a;
  } else if (votes[winner] < minVotes || votes[winner] < int(evidence.size()))
    return a;
  auto merged = a;
  std::map<Key, int> coords;
  for (auto& b : a)
    coords[{b.x, b.y, b.z}] = b.rotation;
  for (auto& b : incoming) {
    Key key = {b.x + winner[0], b.y + winner[1], b.z + winner[2]};
    if (coords.count(key) && coords[key] != b.rotation) return a;
  }
  for (auto b : incoming) {
    b.x += winner[0];
    b.y += winner[1];
    b.z += winner[2];
    Key key = {b.x, b.y, b.z};
    if (!coords.count(key)) {
      merged.push_back(b);
      coords[key] = b.rotation;
    }
  }
  return merged;
}
}  // namespace grass
