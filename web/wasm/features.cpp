#include "features.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

extern "C" {
int extract(float* image, int width, int height);
float* get_out();
}

namespace grass {
namespace {
constexpr std::size_t MaxInputPixels = 16 * 1024 * 1024;
constexpr int MaxSiftPixels = 1024 * 1024;
constexpr int FeatureStride = 130;

// Scope-based accounting also records work done before an extraction throws.
class PhaseTimer {
  using Clock = std::chrono::steady_clock;
  double* destination;
  Clock::time_point start;

 public:
  explicit PhaseTimer(double* output) : destination(output) {
    if (destination) start = Clock::now();
  }
  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;
  void stop() noexcept {
    if (!destination) return;
    *destination += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    destination = nullptr;
  }
  ~PhaseTimer() {
    stop();
  }
};

struct PreparedImage {
  std::vector<float> data;
  int width, height;
  double sx, sy;
};

// Deliberately not cv::resize: prepareSiftImage computes in JS doubles and
// rounds to float only when assigning each Float32Array output pixel.
PreparedImage prepareImage(const std::vector<float>& green, int w, int h) {
  if (w < 8 || h < 8 || static_cast<std::size_t>(w) * h > MaxInputPixels ||
      green.size() != static_cast<std::size_t>(w) * h)
    throw std::invalid_argument("Invalid SIFT grayscale dimensions or stride");
  const double scale =
      std::min({1.0, std::sqrt(static_cast<double>(MaxSiftPixels) / (static_cast<double>(w) * h)),
                2048.0 / std::max(w, h)});
  const int width = std::max(8, static_cast<int>(std::floor(w * scale)));
  const int height = std::max(8, static_cast<int>(std::floor(h * scale)));
  PreparedImage image{{}, width, height, static_cast<double>(w) / width, static_cast<double>(h) / height};
  if (width == w && height == h) {
    image.data = green;
    return image;
  }
  image.data.resize(static_cast<std::size_t>(width) * height);
  std::vector<int> x0Arr(width), x1Arr(width);
  std::vector<double> dxArr(width);
  for (int x = 0; x < width; ++x) {
    const double fx = (x + 0.5) * image.sx - 0.5;
    const int x0 = std::max(0, static_cast<int>(std::floor(fx)));
    x0Arr[x] = x0;
    x1Arr[x] = std::min(w - 1, x0 + 1);
    dxArr[x] = fx - x0;
  }
  const float* gPtr = green.data();
  float* outPtr = image.data.data();
  for (int y = 0; y < height; ++y) {
    const double fy = (y + 0.5) * image.sy - 0.5;
    const int y0 = std::max(0, static_cast<int>(std::floor(fy)));
    const int y1 = std::min(h - 1, y0 + 1);
    const double dy = fy - y0;
    const double omdy = 1.0 - dy;
    const float* row0 = gPtr + static_cast<std::size_t>(y0) * w;
    const float* row1 = gPtr + static_cast<std::size_t>(y1) * w;
    float* dstRow = outPtr + static_cast<std::size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
      const int x0 = x0Arr[x];
      const int x1 = x1Arr[x];
      const double dx = dxArr[x];
      const double omdx = 1.0 - dx;
      const double top = omdx * row0[x0] + dx * row0[x1];
      const double bot = omdx * row1[x0] + dx * row1[x1];
      dstRow[x] = static_cast<float>(omdy * top + dy * bot);
    }
  }
  return image;
}

struct SiftFeatures {
  int count;
  std::vector<float> data;
};

SiftFeatures features(const std::vector<float>& green, int w, int h) {
  auto image = prepareImage(green, w, h);
  const int count = extract(image.data.data(), image.width, image.height);
  if (count < 0 || count > 20000)
    throw std::runtime_error("SIFT extraction failed or exceeded memory budget");
  const float* output = get_out();
  if (!output && count) throw std::runtime_error("Invalid SIFT result buffer");
  SiftFeatures result{count, {}};
  // Copy now: the next extract call overwrites the C static output buffer.
  if (count) result.data.assign(output, output + count * FeatureStride);
  for (int i = 0; i < count; ++i) {
    result.data[i * FeatureStride] =
        static_cast<float>((static_cast<double>(result.data[i * FeatureStride]) + .5) * image.sx - .5);
    result.data[i * FeatureStride + 1] =
        static_cast<float>((static_cast<double>(result.data[i * FeatureStride + 1]) + .5) * image.sy - .5);
  }
  return result;
}

double signedAreaTwice(const Quad& q) {
  double area = 0;
  for (int i = 0; i < 4; ++i)
    area = area + q[i].x * q[(i + 1) % 4].y - q[(i + 1) % 4].x * q[i].y;
  return area;
}

int winding(const Quad& q) {
  int sign = 0;
  for (int i = 0; i < 4; ++i) {
    const auto& a = q[i];
    const auto& b = q[(i + 1) % 4];
    const auto& c = q[(i + 2) % 4];
    const double z = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (!std::isfinite(z) || z == 0) return 0;
    const int next = z > 0 ? 1 : -1;
    if (sign && next != sign) return 0;
    sign = next;
  }
  return sign;
}

bool inImage(const Quad& q, int w, int h) {
  for (const auto& p : q)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= w - 1 || p.y >= h - 1)
      return false;
  return true;
}

Quad canonical(const Quad& q, bool reverse) {
  int start = 0;
  for (int i = 0; i < 4; ++i)
    if (q[i].x + q[i].y < q[start].x + q[start].y) start = i;
  Quad result;
  for (int i = 0; i < 4; ++i)
    result[i] = q[(start + i) % 4];
  if (reverse) std::swap(result[1], result[3]);
  return result;
}

Quad project(const cv::Mat& H, int size) {
  if (H.rows != 3 || H.cols != 3 || H.type() != CV_64FC1)
    throw std::runtime_error("Invalid feature homography result");
  const double* m = H.ptr<double>();
  const Quad corners{{{0, 0},
                      {static_cast<double>(size), 0},
                      {static_cast<double>(size), static_cast<double>(size)},
                      {0, static_cast<double>(size)}}};
  Quad q;
  for (int i = 0; i < 4; ++i) {
    const double x = corners[i].x, y = corners[i].y;
    const double d = m[6] * x + m[7] * y + m[8];
    q[i] = {(m[0] * x + m[1] * y + m[2]) / d, (m[3] * x + m[4] * y + m[5]) / d};
  }
  return q;
}

// Preserve insertion order exactly as Array.filter(!mask[j]); OpenCV must see
// CV_32FC2 points, not doubles, for parity with matFromArray in OpenCV.js.
int removeInliers(std::vector<cv::Point2f>& P, std::vector<cv::Point2f>& Q, const cv::Mat& mask,
                  int required) {
  if (mask.type() != CV_8UC1 || mask.total() != P.size() || !mask.isContinuous())
    throw std::runtime_error("Invalid feature RANSAC inlier mask");
  const auto* flags = mask.ptr<unsigned char>();
  int inliers = 0;
  for (std::size_t j = 0; j < P.size(); ++j)
    inliers += flags[j] != 0;
  if (inliers < required) return inliers;
  std::size_t out = 0;
  for (std::size_t j = 0; j < P.size(); ++j) {
    if (!flags[j]) {
      P[out] = P[j];
      Q[out] = Q[j];
      ++out;
    }
  }
  P.resize(out);
  Q.resize(out);
  return inliers;
}

bool solveLinear8(double A[8][9], double out[8]) {
  for (int col = 0; col < 8; ++col) {
    int piv = col;
    double best = std::abs(A[col][col]);
    for (int r = col + 1; r < 8; ++r) {
      double v = std::abs(A[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-12) return false;
    if (piv != col) {
      for (int j = col; j < 9; ++j)
        std::swap(A[col][j], A[piv][j]);
    }
    double invD = 1.0 / A[col][col];
    for (int j = col; j < 9; ++j)
      A[col][j] *= invD;
    for (int r = 0; r < 8; ++r) {
      if (r == col) continue;
      double f = A[r][col];
      if (f == 0.0) continue;
      for (int j = col; j < 9; ++j)
        A[r][j] -= f * A[col][j];
    }
  }
  for (int i = 0; i < 8; ++i)
    out[i] = A[i][8];
  return true;
}

bool solveHomography4(const cv::Point2f p[4], const cv::Point2f q[4], double H[9]) {
  // Reject collinear 3-point subsets cheaply before solving 8x8.
  static constexpr int tri[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  for (const auto& t : tri) {
    float cp =
        (p[t[1]].x - p[t[0]].x) * (p[t[2]].y - p[t[0]].y) - (p[t[1]].y - p[t[0]].y) * (p[t[2]].x - p[t[0]].x);
    float cq =
        (q[t[1]].x - q[t[0]].x) * (q[t[2]].y - q[t[0]].y) - (q[t[1]].y - q[t[0]].y) * (q[t[2]].x - q[t[0]].x);
    if (std::abs(cp) < 4.0f || std::abs(cq) < 4.0f) return false;
  }
  double A[8][9]{};
  for (int i = 0; i < 4; ++i) {
    double x = p[i].x, y = p[i].y, u = q[i].x, v = q[i].y;
    A[2 * i][0] = x;
    A[2 * i][1] = y;
    A[2 * i][2] = 1.0;
    A[2 * i][6] = -u * x;
    A[2 * i][7] = -u * y;
    A[2 * i][8] = u;
    A[2 * i + 1][3] = x;
    A[2 * i + 1][4] = y;
    A[2 * i + 1][5] = 1.0;
    A[2 * i + 1][6] = -v * x;
    A[2 * i + 1][7] = -v * y;
    A[2 * i + 1][8] = v;
  }
  double sol[8];
  if (!solveLinear8(A, sol)) return false;
  for (int i = 0; i < 8; ++i)
    H[i] = sol[i];
  H[8] = 1.0;
  return true;
}

bool solveHomographyLS(const std::vector<cv::Point2f>& P, const std::vector<cv::Point2f>& Q,
                       const std::vector<int>& ids, double H[9]) {
  if (ids.size() < 4) return false;
  double cxp = 0, cyp = 0, cxq = 0, cyq = 0;
  const double invN = 1.0 / ids.size();
  for (int idx : ids) {
    cxp += P[idx].x;
    cyp += P[idx].y;
    cxq += Q[idx].x;
    cyq += Q[idx].y;
  }
  cxp *= invN;
  cyp *= invN;
  cxq *= invN;
  cyq *= invN;
  double dp = 0, dq = 0;
  for (int idx : ids) {
    dp += std::hypot(P[idx].x - cxp, P[idx].y - cyp);
    dq += std::hypot(Q[idx].x - cxq, Q[idx].y - cyq);
  }
  dp *= invN;
  dq *= invN;
  if (dp < 1e-6 || dq < 1e-6) return false;
  double sp = std::sqrt(2.0) / dp, sq = std::sqrt(2.0) / dq;

  double Nmat[8][9]{};
  for (int idx : ids) {
    double x = (P[idx].x - cxp) * sp, y = (P[idx].y - cyp) * sp;
    double u = (Q[idx].x - cxq) * sq, v = (Q[idx].y - cyq) * sq;
    double r1[9] = {x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u};
    double r2[9] = {0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v};
    for (int a = 0; a < 8; ++a) {
      Nmat[a][8] += r1[a] * r1[8] + r2[a] * r2[8];
      for (int b = 0; b < 8; ++b)
        Nmat[a][b] += r1[a] * r1[b] + r2[a] * r2[b];
    }
  }
  double sol[8];
  if (!solveLinear8(Nmat, sol)) return false;
  double Hn[9] = {sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], 1.0};
  // Denormalize: H = Tq^{-1} * Hn * Tp
  double Tp[9] = {sp, 0, -sp * cxp, 0, sp, -sp * cyp, 0, 0, 1};
  double Tqi[9] = {1.0 / sq, 0, cxq, 0, 1.0 / sq, cyq, 0, 0, 1};
  double tmp[9]{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        tmp[i * 3 + j] += Hn[i * 3 + k] * Tp[k * 3 + j];
  for (int i = 0; i < 9; ++i)
    H[i] = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        H[i * 3 + j] += Tqi[i * 3 + k] * tmp[k * 3 + j];
  if (std::abs(H[8]) < 1e-12) return false;
  double invH8 = 1.0 / H[8];
  for (int i = 0; i < 9; ++i)
    H[i] *= invH8;
  return true;
}

int countInliersHSoA(const float* Px, const float* Py, const float* Qx, const float* Qy, int n,
                     const double H[9], float thresh2, int minBeat) {
  const float h0 = float(H[0]), h1 = float(H[1]), h2 = float(H[2]);
  const float h3 = float(H[3]), h4 = float(H[4]), h5 = float(H[5]);
  const float h6 = float(H[6]), h7 = float(H[7]), h8 = float(H[8]);
  int count = 0;
  int i = 0;
#ifdef __wasm_simd128__
  const v128_t vh0 = wasm_f32x4_splat(h0), vh1 = wasm_f32x4_splat(h1), vh2 = wasm_f32x4_splat(h2);
  const v128_t vh3 = wasm_f32x4_splat(h3), vh4 = wasm_f32x4_splat(h4), vh5 = wasm_f32x4_splat(h5);
  const v128_t vh6 = wasm_f32x4_splat(h6), vh7 = wasm_f32x4_splat(h7), vh8 = wasm_f32x4_splat(h8);
  const v128_t veps = wasm_f32x4_splat(1e-6f);
  const v128_t vone = wasm_f32x4_splat(1.0f);
  const v128_t vthresh2 = wasm_f32x4_splat(thresh2);

  for (; i + 4 <= n; i += 4) {
    if ((i & 31) == 0 && count + (n - i) <= minBeat) return count;
    const v128_t px = wasm_v128_load(Px + i);
    const v128_t py = wasm_v128_load(Py + i);
    const v128_t d = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(vh6, px), wasm_f32x4_mul(vh7, py)), vh8);
    const v128_t valid = wasm_f32x4_gt(d, veps);
    const v128_t invD = wasm_f32x4_div(vone, d);
    const v128_t ux = wasm_f32x4_mul(
        wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(vh0, px), wasm_f32x4_mul(vh1, py)), vh2), invD);
    const v128_t uy = wasm_f32x4_mul(
        wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(vh3, px), wasm_f32x4_mul(vh4, py)), vh5), invD);
    const v128_t dx = wasm_f32x4_sub(ux, wasm_v128_load(Qx + i));
    const v128_t dy = wasm_f32x4_sub(uy, wasm_v128_load(Qy + i));
    const v128_t dist2 = wasm_f32x4_add(wasm_f32x4_mul(dx, dx), wasm_f32x4_mul(dy, dy));
    const v128_t hit = wasm_v128_and(valid, wasm_f32x4_le(dist2, vthresh2));
    count += __builtin_popcount(wasm_i32x4_bitmask(hit));
  }
#endif
  for (; i < n; ++i) {
    const float px = Px[i], py = Py[i];
    const float d = h6 * px + h7 * py + h8;
    if (d <= 1e-6f) continue;
    const float invD = 1.0f / d;
    const float dx = (h0 * px + h1 * py + h2) * invD - Qx[i];
    const float dy = (h3 * px + h4 * py + h5) * invD - Qy[i];
    if (dx * dx + dy * dy <= thresh2) ++count;
  }
  return count;
}

// Spatially-guided NAPSAC + global RANSAC for multi-tile homography extraction.
// Grass tiles are spatially compact in image space, so sampling 3 neighbors near
// an anchor raises 4-point same-tile hit rate by orders of magnitude over uniform RANSAC.
int findAndRemoveHomographyNapsac(std::vector<cv::Point2f>& P, std::vector<cv::Point2f>& Q, int S, int w,
                                  int h, int minInliers, Quad& outQuad, bool& outValid, std::uint64_t& rng) {
  outValid = false;
  const int n = static_cast<int>(P.size());
  if (n < 10) return 0;

  auto nextRand = [&]() -> std::uint32_t {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return static_cast<std::uint32_t>(rng);
  };

  const float thresh2 = 2.5f * 2.5f;
  const float radii2[3] = {
      std::max(80.0f, std::min(w, h) * 0.12f), std::max(180.0f, std::min(w, h) * 0.28f),
      0.0f  // global uniform fallback
  };
  const float r2Sq[2] = {radii2[0] * radii2[0], radii2[1] * radii2[1]};

  // Structure-of-Arrays coordinates for SIMD inlier counting + uniform spatial grid for O(1) NAPSAC queries.
  std::vector<float> Px(n), Py(n), Qx(n), Qy(n);
  const float cellSize = radii2[0];
  const float invCell = 1.0f / cellSize;
  const int gridW = std::max(1, static_cast<int>(std::ceil(w * invCell)) + 1);
  const int gridH = std::max(1, static_cast<int>(std::ceil(h * invCell)) + 1);
  const int numCells = gridW * gridH;
  std::vector<int> cellCount(numCells, 0);
  std::vector<int> ptCell(n);
  for (int i = 0; i < n; ++i) {
    Px[i] = P[i].x;
    Py[i] = P[i].y;
    Qx[i] = Q[i].x;
    Qy[i] = Q[i].y;
    int cx = std::clamp(static_cast<int>(Q[i].x * invCell), 0, gridW - 1);
    int cy = std::clamp(static_cast<int>(Q[i].y * invCell), 0, gridH - 1);
    int c = cy * gridW + cx;
    ptCell[i] = c;
    cellCount[c]++;
  }
  std::vector<int> cellStart(numCells + 1, 0);
  for (int c = 0; c < numCells; ++c)
    cellStart[c + 1] = cellStart[c] + cellCount[c];
  std::vector<int> cellCursor = cellStart;
  std::vector<int> cellPts(n);
  for (int i = 0; i < n; ++i) {
    cellPts[cellCursor[ptCell[i]]++] = i;
  }
  const int wideSpan = static_cast<int>(std::ceil(radii2[1] * invCell));

  std::vector<int> pool;
  pool.reserve(n);
  double bestH[9]{};
  int bestCount = 0;
  int maxIters = 600;

  for (int iter = 0; iter < maxIters; ++iter) {
    int ids[4];
    const int mode = iter & 3;  // 0,1: tight NAPSAC; 2: wide NAPSAC; 3: global
    if (mode < 3) {
      const float limit2 = (mode < 2) ? r2Sq[0] : r2Sq[1];
      const int span = (mode < 2) ? 1 : wideSpan;
      ids[0] = static_cast<int>(nextRand() % static_cast<std::uint32_t>(n));
      const float ax = Qx[ids[0]], ay = Qy[ids[0]];
      const int acx = std::clamp(static_cast<int>(ax * invCell), 0, gridW - 1);
      const int acy = std::clamp(static_cast<int>(ay * invCell), 0, gridH - 1);
      const int cy0 = std::max(0, acy - span), cy1 = std::min(gridH - 1, acy + span);
      const int cx0 = std::max(0, acx - span), cx1 = std::min(gridW - 1, acx + span);
      pool.clear();
      for (int cy = cy0; cy <= cy1; ++cy) {
        const int rowCell = cy * gridW;
        for (int cx = cx0; cx <= cx1; ++cx) {
          const int start = cellStart[rowCell + cx];
          const int end = cellStart[rowCell + cx + 1];
          for (int idx = start; idx < end; ++idx) {
            const int j = cellPts[idx];
            if (j == ids[0]) continue;
            const float dx = Qx[j] - ax, dy = Qy[j] - ay;
            if (dx * dx + dy * dy <= limit2) pool.push_back(j);
          }
        }
      }
      if (pool.size() >= 3) {
        const std::uint32_t m = static_cast<std::uint32_t>(pool.size());
        int a = static_cast<int>(nextRand() % m);
        int b = static_cast<int>(nextRand() % (m - 1));
        if (b >= a) ++b;
        int c = static_cast<int>(nextRand() % (m - 2));
        int lo = std::min(a, b), hi = std::max(a, b);
        if (c >= lo) ++c;
        if (c >= hi) ++c;
        ids[1] = pool[a];
        ids[2] = pool[b];
        ids[3] = pool[c];
      } else {
        // Fall back to global uniform if local neighborhood is sparse.
        for (int k = 1; k < 4; ++k) {
          int cand;
          do {
            cand = static_cast<int>(nextRand() % static_cast<std::uint32_t>(n));
          } while (std::find(ids, ids + k, cand) != ids + k);
          ids[k] = cand;
        }
      }
    } else {
      for (int k = 0; k < 4; ++k) {
        int cand;
        do {
          cand = static_cast<int>(nextRand() % static_cast<std::uint32_t>(n));
        } while (std::find(ids, ids + k, cand) != ids + k);
        ids[k] = cand;
      }
    }

    cv::Point2f sp[4] = {P[ids[0]], P[ids[1]], P[ids[2]], P[ids[3]]};
    cv::Point2f sq[4] = {Q[ids[0]], Q[ids[1]], Q[ids[2]], Q[ids[3]]};
    double H[9];
    if (!solveHomography4(sp, sq, H)) continue;
    // Quick corner sanity check before scanning all points.
    bool badCorner = false;
    for (int c = 0; c < 4; ++c) {
      double cx = (c == 1 || c == 2) ? S : 0.0;
      double cy = (c >= 2) ? S : 0.0;
      if (H[6] * cx + H[7] * cy + H[8] <= 1e-4) {
        badCorner = true;
        break;
      }
    }
    if (badCorner) continue;

    int cnt = countInliersHSoA(Px.data(), Py.data(), Qx.data(), Qy.data(), n, H, thresh2,
                               std::max(bestCount, minInliers - 1));
    if (cnt > bestCount) {
      bestCount = cnt;
      std::copy(H, H + 9, bestH);
      if (bestCount >= 16 && iter >= 160) {
        double ratioIn = double(bestCount) / double(n);
        double p4 = ratioIn * ratioIn * ratioIn * ratioIn;
        if (p4 > 1e-4 && p4 < 0.999) {
          int est = static_cast<int>(std::ceil(std::log(0.005) / std::log(1.0 - p4)));
          maxIters = std::min(maxIters, std::max(160, est));
        }
      }
    }
  }

  if (bestCount < minInliers) return bestCount;

  // Two-pass least-squares polish on all inliers.
  std::vector<int> inlierIds;
  inlierIds.reserve(bestCount + 16);
  for (int pass = 0; pass < 2; ++pass) {
    inlierIds.clear();
    for (int i = 0; i < n; ++i) {
      double d = bestH[6] * P[i].x + bestH[7] * P[i].y + bestH[8];
      if (d <= 1e-6) continue;
      double dx = (bestH[0] * P[i].x + bestH[1] * P[i].y + bestH[2]) / d - Q[i].x;
      double dy = (bestH[3] * P[i].x + bestH[4] * P[i].y + bestH[5]) / d - Q[i].y;
      if (dx * dx + dy * dy <= thresh2) inlierIds.push_back(i);
    }
    if (static_cast<int>(inlierIds.size()) < minInliers) return static_cast<int>(inlierIds.size());
    double refined[9];
    if (solveHomographyLS(P, Q, inlierIds, refined)) {
      int cntRef = countInliersHSoA(Px.data(), Py.data(), Qx.data(), Qy.data(), n, refined, thresh2, -1);
      if (cntRef >= static_cast<int>(inlierIds.size())) {
        std::copy(refined, refined + 9, bestH);
      }
    }
  }

  // Final inlier mask & removal preserving order.
  std::vector<unsigned char> isInlier(n, 0);
  int finalCount = 0;
  for (int i = 0; i < n; ++i) {
    double d = bestH[6] * P[i].x + bestH[7] * P[i].y + bestH[8];
    if (d <= 1e-6) continue;
    double dx = (bestH[0] * P[i].x + bestH[1] * P[i].y + bestH[2]) / d - Q[i].x;
    double dy = (bestH[3] * P[i].x + bestH[4] * P[i].y + bestH[5]) / d - Q[i].y;
    if (dx * dx + dy * dy <= thresh2) {
      isInlier[i] = 1;
      ++finalCount;
    }
  }
  if (finalCount < minInliers) return finalCount;

  std::size_t out = 0;
  for (int i = 0; i < n; ++i) {
    if (!isInlier[i]) {
      P[out] = P[i];
      Q[out] = Q[i];
      ++out;
    }
  }
  P.resize(out);
  Q.resize(out);

  static constexpr double cs[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  for (int i = 0; i < 4; ++i) {
    double x = cs[i][0] * S, y = cs[i][1] * S;
    double d = bestH[6] * x + bestH[7] * y + bestH[8];
    outQuad[i] = {(bestH[0] * x + bestH[1] * y + bestH[2]) / d, (bestH[3] * x + bestH[4] * y + bestH[5]) / d};
  }
  outValid = true;
  return finalCount;
}

// Typed-array assignment uses ToUint8, not OpenCV saturate_cast/round.
unsigned char jsUint8(double value) {
  if (!std::isfinite(value)) return 0;
  double n = std::fmod(std::trunc(value), 256.0);
  if (n < 0) n += 256;
  return static_cast<unsigned char>(n);
}

struct SiftMatchTuple {
  double best;
  double second;
  int jbest;
};

struct SiftCache {
  bool valid = false;
  int w = 0;
  int h = 0;
  int s = 0;
  std::array<double, 256> ref{};
  std::uint64_t green_hash = 0;
  SiftFeatures a{};
  SiftFeatures b{};
  std::vector<SiftMatchTuple> matches{};
};

std::uint64_t hashGreen(const std::vector<float>& green) {
  std::uint64_t h = 14695981039346656037ull;
  const auto* ptr = reinterpret_cast<const std::uint32_t*>(green.data());
  for (std::size_t i = 0; i < green.size(); ++i) {
    h ^= ptr[i];
    h *= 1099511628211ull;
  }
  return h;
}
}  // namespace

std::vector<Quad> siftSeeds(const std::vector<float>& green, int w, int h, const std::array<double, 256>& ref,
                            double ratio, FeatureTimings* timings) {
  if (!std::isfinite(ratio)) throw std::invalid_argument("Invalid SIFT ratio");
  if (w < 8 || h < 8 || static_cast<std::size_t>(w) * h > MaxInputPixels ||
      green.size() != static_cast<std::size_t>(w) * h)
    throw std::invalid_argument("Invalid SIFT grayscale dimensions or stride");
  constexpr int S = 256;
  const std::uint64_t green_hash = hashGreen(green);
  static thread_local SiftCache cache;
  const bool hit = cache.valid && cache.w == w && cache.h == h && cache.s == S &&
                   cache.green_hash == green_hash && cache.ref == ref;
  if (!hit) {
    PhaseTimer extractionTimer(timings ? &timings->extraction : nullptr);
    std::vector<float> texture(S * S);
    for (int y = 0; y < S; ++y)
      for (int x = 0; x < S; ++x)
        texture[y * S + x] = static_cast<float>(ref[(y / 16) * 16 + x / 16]);
    auto a = features(texture, S, S);
    auto b = features(green, w, h);
    extractionTimer.stop();
    PhaseTimer matchingTimer(timings && b.count > 0 ? &timings->matchingHomography : nullptr);
    std::vector<SiftMatchTuple> matches(b.count);
#ifdef __wasm_simd128__
    // Pack reference descriptors contiguously (128 floats each) and precompute squared norms
    // so matching reduces to 4-way tiled dot products: ||b - a||^2 = ||b||^2 + ||a||^2 - 2*(b . a).
    std::vector<float> aPack(static_cast<std::size_t>(a.count) * 128);
    std::vector<float> aNorm2(a.count);
    for (int j = 0; j < a.count; ++j) {
      const float* src = a.data.data() + j * FeatureStride + 2;
      float* dst = aPack.data() + j * 128;
      v128_t nAcc = wasm_f32x4_splat(0.0f);
      for (int k = 0; k < 128; k += 4) {
        v128_t v = wasm_v128_load(src + k);
        wasm_v128_store(dst + k, v);
        nAcc = wasm_f32x4_add(nAcc, wasm_f32x4_mul(v, v));
      }
      nAcc = wasm_f32x4_add(nAcc, wasm_i32x4_shuffle(nAcc, nAcc, 2, 3, 0, 1));
      nAcc = wasm_f32x4_add(nAcc, wasm_i32x4_shuffle(nAcc, nAcc, 1, 0, 3, 2));
      aNorm2[j] = wasm_f32x4_extract_lane(nAcc, 0);
    }

    auto hsum4 = [](v128_t v) -> float {
      v = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
      v = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 1, 0, 3, 2));
      return wasm_f32x4_extract_lane(v, 0);
    };

    // Precompute b descriptor squared norms once for 2x4 tiled GEMM-style dot products.
    std::vector<float> bNorm2(b.count);
    for (int i = 0; i < b.count; ++i) {
      const float* bp = b.data.data() + i * FeatureStride + 2;
      v128_t bnAcc = wasm_f32x4_splat(0.0f);
      for (int k = 0; k < 128; k += 4) {
        v128_t v = wasm_v128_load(bp + k);
        bnAcc = wasm_f32x4_add(bnAcc, wasm_f32x4_mul(v, v));
      }
      bNorm2[i] = hsum4(bnAcc);
    }

    int i = 0;
    for (; i + 2 <= b.count; i += 2) {
      const float* bp0 = b.data.data() + i * FeatureStride + 2;
      const float* bp1 = b.data.data() + (i + 1) * FeatureStride + 2;
      const float bn0 = bNorm2[i], bn1 = bNorm2[i + 1];
      double best0 = std::numeric_limits<double>::infinity(), second0 = best0;
      double best1 = std::numeric_limits<double>::infinity(), second1 = best1;
      int jb0 = 0, jb1 = 0;

      int j = 0;
      for (; j + 4 <= a.count; j += 4) {
        const float* ap0 = aPack.data() + (j + 0) * 128;
        const float* ap1 = aPack.data() + (j + 1) * 128;
        const float* ap2 = aPack.data() + (j + 2) * 128;
        const float* ap3 = aPack.data() + (j + 3) * 128;
        v128_t d00 = wasm_f32x4_splat(0.0f), d01 = wasm_f32x4_splat(0.0f);
        v128_t d02 = wasm_f32x4_splat(0.0f), d03 = wasm_f32x4_splat(0.0f);
        v128_t d10 = wasm_f32x4_splat(0.0f), d11 = wasm_f32x4_splat(0.0f);
        v128_t d12 = wasm_f32x4_splat(0.0f), d13 = wasm_f32x4_splat(0.0f);
        for (int k = 0; k < 128; k += 4) {
          const v128_t vb0 = wasm_v128_load(bp0 + k);
          const v128_t vb1 = wasm_v128_load(bp1 + k);
          const v128_t va0 = wasm_v128_load(ap0 + k);
          const v128_t va1 = wasm_v128_load(ap1 + k);
          const v128_t va2 = wasm_v128_load(ap2 + k);
          const v128_t va3 = wasm_v128_load(ap3 + k);
          d00 = wasm_f32x4_add(d00, wasm_f32x4_mul(vb0, va0));
          d01 = wasm_f32x4_add(d01, wasm_f32x4_mul(vb0, va1));
          d02 = wasm_f32x4_add(d02, wasm_f32x4_mul(vb0, va2));
          d03 = wasm_f32x4_add(d03, wasm_f32x4_mul(vb0, va3));
          d10 = wasm_f32x4_add(d10, wasm_f32x4_mul(vb1, va0));
          d11 = wasm_f32x4_add(d11, wasm_f32x4_mul(vb1, va1));
          d12 = wasm_f32x4_add(d12, wasm_f32x4_mul(vb1, va2));
          d13 = wasm_f32x4_add(d13, wasm_f32x4_mul(vb1, va3));
        }
        const float dots0[4] = {hsum4(d00), hsum4(d01), hsum4(d02), hsum4(d03)};
        const float dots1[4] = {hsum4(d10), hsum4(d11), hsum4(d12), hsum4(d13)};
        for (int m = 0; m < 4; ++m) {
          const float an = aNorm2[j + m];
          const double dist0 = std::max(0.0, double(bn0 + an - 2.0f * dots0[m]));
          if (dist0 < best0) {
            second0 = best0;
            best0 = dist0;
            jb0 = j + m;
          } else if (dist0 < second0) {
            second0 = dist0;
          }
          const double dist1 = std::max(0.0, double(bn1 + an - 2.0f * dots1[m]));
          if (dist1 < best1) {
            second1 = best1;
            best1 = dist1;
            jb1 = j + m;
          } else if (dist1 < second1) {
            second1 = dist1;
          }
        }
      }
      for (; j < a.count; ++j) {
        const float* ap = aPack.data() + j * 128;
        v128_t d0 = wasm_f32x4_splat(0.0f), d1 = wasm_f32x4_splat(0.0f);
        for (int k = 0; k < 128; k += 4) {
          const v128_t va = wasm_v128_load(ap + k);
          d0 = wasm_f32x4_add(d0, wasm_f32x4_mul(wasm_v128_load(bp0 + k), va));
          d1 = wasm_f32x4_add(d1, wasm_f32x4_mul(wasm_v128_load(bp1 + k), va));
        }
        const float an = aNorm2[j];
        const double dist0 = std::max(0.0, double(bn0 + an - 2.0f * hsum4(d0)));
        if (dist0 < best0) {
          second0 = best0;
          best0 = dist0;
          jb0 = j;
        } else if (dist0 < second0) {
          second0 = dist0;
        }
        const double dist1 = std::max(0.0, double(bn1 + an - 2.0f * hsum4(d1)));
        if (dist1 < best1) {
          second1 = best1;
          best1 = dist1;
          jb1 = j;
        } else if (dist1 < second1) {
          second1 = dist1;
        }
      }
      matches[i] = {best0, second0, jb0};
      matches[i + 1] = {best1, second1, jb1};
    }
    for (; i < b.count; ++i) {
      const float* bp = b.data.data() + i * FeatureStride + 2;
      const float bn = bNorm2[i];
      double best = std::numeric_limits<double>::infinity();
      double second = best;
      int jbest = 0;
      for (int j = 0; j < a.count; ++j) {
        const float* ap = aPack.data() + j * 128;
        v128_t d0 = wasm_f32x4_splat(0.0f);
        for (int k = 0; k < 128; k += 4) {
          d0 = wasm_f32x4_add(d0, wasm_f32x4_mul(wasm_v128_load(bp + k), wasm_v128_load(ap + k)));
        }
        const double dist = std::max(0.0, double(bn + aNorm2[j] - 2.0f * hsum4(d0)));
        if (dist < best) {
          second = best;
          best = dist;
          jbest = j;
        } else if (dist < second) {
          second = dist;
        }
      }
      matches[i] = {best, second, jbest};
    }
#else
    for (int i = 0; i < b.count; ++i) {
      const int bi = i * FeatureStride + 2;
      const float* bp = b.data.data() + bi;
      double best = std::numeric_limits<double>::infinity();
      double second = best;
      int jbest = 0;
      for (int j = 0; j < a.count; ++j) {
        const int aj = j * FeatureStride + 2;
        const float* ap = a.data.data() + aj;
        double distance = 0;
        for (int k = 0; k < 128; k += 8) {
          const double v0 = static_cast<double>(bp[k]) - static_cast<double>(ap[k]);
          const double v1 = static_cast<double>(bp[k + 1]) - static_cast<double>(ap[k + 1]);
          const double v2 = static_cast<double>(bp[k + 2]) - static_cast<double>(ap[k + 2]);
          const double v3 = static_cast<double>(bp[k + 3]) - static_cast<double>(ap[k + 3]);
          const double v4 = static_cast<double>(bp[k + 4]) - static_cast<double>(ap[k + 4]);
          const double v5 = static_cast<double>(bp[k + 5]) - static_cast<double>(ap[k + 5]);
          const double v6 = static_cast<double>(bp[k + 6]) - static_cast<double>(ap[k + 6]);
          const double v7 = static_cast<double>(bp[k + 7]) - static_cast<double>(ap[k + 7]);
          distance += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
          if (distance >= second) break;
        }
        if (distance < best) {
          second = best;
          best = distance;
          jbest = j;
        } else if (distance < second) {
          second = distance;
        }
      }
      matches[i] = {best, second, jbest};
    }
#endif
    cache.valid = true;
    cache.w = w;
    cache.h = h;
    cache.s = S;
    cache.ref = ref;
    cache.green_hash = green_hash;
    cache.a = std::move(a);
    cache.b = std::move(b);
    cache.matches = std::move(matches);
  }
  PhaseTimer matchingTimer(timings && cache.b.count > 0 ? &timings->matchingHomography : nullptr);
  std::vector<cv::Point2f> P, Q;
  const double ratio2 = ratio * ratio;
  for (int i = 0; i < cache.b.count; ++i) {
    const auto& m = cache.matches[i];
    if (m.best < ratio2 * m.second) {
      P.emplace_back(cache.a.data[m.jbest * FeatureStride], cache.a.data[m.jbest * FeatureStride + 1]);
      Q.emplace_back(cache.b.data[i * FeatureStride], cache.b.data[i * FeatureStride + 1]);
    }
  }
  std::vector<Quad> found;
  std::uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(P.size()) * 0xbf58476d1ce4e5b9ULL);
  for (int it = 0; it < 20 && P.size() >= 10; ++it) {
    Quad q{};
    bool ok = false;
    if (findAndRemoveHomographyNapsac(P, Q, S, w, h, 8, q, ok, rng) < 8 || !ok) break;
    if (!inImage(q, w, h)) continue;
    const int sign = winding(q);
    const double area = std::abs(signedAreaTwice(q)) / 2;
    if (!sign || area < 1000 || area > static_cast<double>(w) * h * .2) continue;
    found.push_back(canonical(q, sign < 0));
  }
  return found;
}

std::vector<Quad> binarySeeds(const std::vector<float>& green, int w, int h,
                              const std::array<double, 256>& ref, FeatureTimings* timings) {
  PhaseTimer preparationTimer(timings ? &timings->extraction : nullptr);
  const auto image = prepareImage(green, w, h);
  cv::Mat img(image.height, image.width, CV_8UC1);
  for (std::size_t i = 0; i < image.data.size(); ++i)
    img.data[i] = jsUint8(image.data[i]);
  cv::Mat refSmall(16, 16, CV_8UC1);
  for (int i = 0; i < 256; ++i)
    refSmall.data[i] = jsUint8(ref[i]);
  auto akaze = cv::AKAZE::create();
  auto orb = cv::ORB::create(6000);
  orb->setScaleFactor(1.15);
  orb->setNLevels(12);
  orb->setEdgeThreshold(8);
  orb->setFastThreshold(5);
  std::vector<cv::KeyPoint> ksAkaze, ksOrb;
  cv::Mat dsAkaze, dsOrb;
  akaze->detectAndCompute(img, cv::noArray(), ksAkaze, dsAkaze);
  orb->detectAndCompute(img, cv::noArray(), ksOrb, dsOrb);
  preparationTimer.stop();
  std::vector<Quad> found;
  for (int S : {64, 96, 128}) {
    for (int interp : {cv::INTER_NEAREST, cv::INTER_LINEAR}) {
      for (int feat = 0; feat < 2; ++feat) {
        PhaseTimer extractionTimer(timings ? &timings->extraction : nullptr);
        cv::Mat refBig;
        cv::resize(refSmall, refBig, cv::Size(S, S), 0, 0, interp);
        cv::Ptr<cv::Feature2D> detector =
            (feat == 0) ? cv::Ptr<cv::Feature2D>(akaze) : cv::Ptr<cv::Feature2D>(orb);
        const auto& ks = (feat == 0) ? ksAkaze : ksOrb;
        const auto& ds = (feat == 0) ? dsAkaze : dsOrb;
        std::vector<cv::KeyPoint> kr;
        cv::Mat dr;
        detector->detectAndCompute(refBig, cv::noArray(), kr, dr);
        extractionTimer.stop();
        PhaseTimer matchingTimer(timings && dr.rows >= 4 && ds.rows >= 4 ? &timings->matchingHomography
                                                                         : nullptr);
        std::vector<cv::Point2f> P, Q;
        if (dr.rows >= 4 && ds.rows >= 4) {
          cv::BFMatcher matcher(cv::NORM_HAMMING, false);
          std::vector<std::vector<cv::DMatch>> matches;
          matcher.knnMatch(ds, dr, matches, 2);
          for (const auto& m : matches) {
            if (m.size() < 2) continue;
            if (m[0].distance < .85 * m[1].distance) {
              P.push_back(kr.at(m[0].trainIdx).pt);
              Q.push_back(ks.at(m[0].queryIdx).pt);
            }
          }
        }
        for (int it = 0; it < 30 && P.size() >= 6; ++it) {
          cv::Mat mask;
          const cv::Mat H = cv::findHomography(P, Q, cv::RANSAC, 4.0, mask, 4000, .995);
          if (H.empty() || removeInliers(P, Q, mask, 5) < 5) break;
          Quad q = project(H, S);
          if (!winding(q) || !inImage(q, image.width, image.height)) continue;
          double minSide = std::numeric_limits<double>::infinity();
          for (int i = 0; i < 4; ++i)
            minSide = std::min(minSide, std::hypot(q[i].x - q[(i + 1) % 4].x, q[i].y - q[(i + 1) % 4].y));
          if (minSide < 14) continue;
          const double area = signedAreaTwice(q);
          if (std::abs(area) / 2 > static_cast<double>(image.width) * image.height * .25) continue;
          q = canonical(q, area < 0);
          for (auto& p : q) {
            p.x = (p.x + .5) * image.sx - .5;
            p.y = (p.y + .5) * image.sy - .5;
          }
          found.push_back(q);
        }
      }
    }
  }
  return found;
}
}  // namespace grass
