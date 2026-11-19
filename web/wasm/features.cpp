#include "features.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
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
    ~PhaseTimer() { stop(); }
};

struct PreparedImage {
    std::vector<float> data;
    int width, height;
    double sx, sy;
};

// Deliberately not cv::resize: prepareSiftImage computes in JS doubles and
// rounds to float only when assigning each Float32Array output pixel.
PreparedImage prepareImage(const std::vector<float>& green, int w, int h) {
    if (w < 8 || h < 8 ||
        static_cast<std::size_t>(w) * h > MaxInputPixels ||
        green.size() != static_cast<std::size_t>(w) * h)
        throw std::invalid_argument("Invalid SIFT grayscale dimensions or stride");
    const double scale = std::min({1.0, std::sqrt(
        static_cast<double>(MaxSiftPixels) / (static_cast<double>(w) * h)),
        2048.0 / std::max(w, h)});
    const int width = std::max(8, static_cast<int>(std::floor(w * scale)));
    const int height = std::max(8, static_cast<int>(std::floor(h * scale)));
    PreparedImage image{{}, width, height,
                        static_cast<double>(w) / width,
                        static_cast<double>(h) / height};
    if (width == w && height == h) {
        image.data = green;
        return image;
    }
    image.data.resize(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        const double fy = (y + .5) * image.sy - .5;
        const int y0 = std::max(0, static_cast<int>(std::floor(fy)));
        const int y1 = std::min(h - 1, y0 + 1);
        const double dy = fy - y0;
        for (int x = 0; x < width; ++x) {
            const double fx = (x + .5) * image.sx - .5;
            const int x0 = std::max(0, static_cast<int>(std::floor(fx)));
            const int x1 = std::min(w - 1, x0 + 1);
            const double dx = fx - x0;
            image.data[y * width + x] = static_cast<float>(
                (1 - dy) * ((1 - dx) * green[y0 * w + x0] +
                            dx * green[y0 * w + x1]) +
                dy * ((1 - dx) * green[y1 * w + x0] +
                      dx * green[y1 * w + x1]));
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
    if (!output && count)
        throw std::runtime_error("Invalid SIFT result buffer");
    SiftFeatures result{count, {}};
    // Copy now: the next extract call overwrites the C static output buffer.
    if (count) result.data.assign(output, output + count * FeatureStride);
    for (int i = 0; i < count; ++i) {
        result.data[i * FeatureStride] = static_cast<float>(
            (static_cast<double>(result.data[i * FeatureStride]) + .5) * image.sx - .5);
        result.data[i * FeatureStride + 1] = static_cast<float>(
            (static_cast<double>(result.data[i * FeatureStride + 1]) + .5) * image.sy - .5);
    }
    return result;
}

double signedAreaTwice(const Quad& q) {
    double area = 0;
    for (int i = 0; i < 4; ++i)
        area = area + q[i].x * q[(i + 1) % 4].y -
                      q[(i + 1) % 4].x * q[i].y;
    return area;
}

int winding(const Quad& q) {
    int sign = 0;
    for (int i = 0; i < 4; ++i) {
        const auto& a = q[i];
        const auto& b = q[(i + 1) % 4];
        const auto& c = q[(i + 2) % 4];
        const double z = (b.x - a.x) * (c.y - b.y) -
                         (b.y - a.y) * (c.x - b.x);
        if (!std::isfinite(z) || z == 0) return 0;
        const int next = z > 0 ? 1 : -1;
        if (sign && next != sign) return 0;
        sign = next;
    }
    return sign;
}

bool inImage(const Quad& q, int w, int h) {
    for (const auto& p : q)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            p.x < 0 || p.y < 0 || p.x >= w - 1 || p.y >= h - 1)
            return false;
    return true;
}

Quad canonical(const Quad& q, bool reverse) {
    int start = 0;
    for (int i = 0; i < 4; ++i)
        if (q[i].x + q[i].y < q[start].x + q[start].y) start = i;
    Quad result;
    for (int i = 0; i < 4; ++i) result[i] = q[(start + i) % 4];
    if (reverse) std::swap(result[1], result[3]);
    return result;
}

Quad project(const cv::Mat& H, int size) {
    if (H.rows != 3 || H.cols != 3 || H.type() != CV_64FC1)
        throw std::runtime_error("Invalid feature homography result");
    const double* m = H.ptr<double>();
    const Quad corners{{{0, 0}, {static_cast<double>(size), 0},
                        {static_cast<double>(size), static_cast<double>(size)},
                        {0, static_cast<double>(size)}}};
    Quad q;
    for (int i = 0; i < 4; ++i) {
        const double x = corners[i].x, y = corners[i].y;
        const double d = m[6] * x + m[7] * y + m[8];
        q[i] = {(m[0] * x + m[1] * y + m[2]) / d,
                (m[3] * x + m[4] * y + m[5]) / d};
    }
    return q;
}

// Preserve insertion order exactly as Array.filter(!mask[j]); OpenCV must see
// CV_32FC2 points, not doubles, for parity with matFromArray in OpenCV.js.
int removeInliers(std::vector<cv::Point2f>& P, std::vector<cv::Point2f>& Q,
                  const cv::Mat& mask, int required) {
    if (mask.type() != CV_8UC1 || mask.total() != P.size() || !mask.isContinuous())
        throw std::runtime_error("Invalid feature RANSAC inlier mask");
    const auto* flags = mask.ptr<unsigned char>();
    int inliers = 0;
    for (std::size_t j = 0; j < P.size(); ++j) inliers += flags[j] != 0;
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
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12) return false;
        if (piv != col) {
            for (int j = col; j < 9; ++j) std::swap(A[col][j], A[piv][j]);
        }
        double invD = 1.0 / A[col][col];
        for (int j = col; j < 9; ++j) A[col][j] *= invD;
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            double f = A[r][col];
            if (f == 0.0) continue;
            for (int j = col; j < 9; ++j) A[r][j] -= f * A[col][j];
        }
    }
    for (int i = 0; i < 8; ++i) out[i] = A[i][8];
    return true;
}

bool solveHomography4(const cv::Point2f p[4], const cv::Point2f q[4], double H[9]) {
    // Reject collinear 3-point subsets cheaply before solving 8x8.
    static constexpr int tri[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    for (const auto& t : tri) {
        float cp = (p[t[1]].x - p[t[0]].x) * (p[t[2]].y - p[t[0]].y) -
                   (p[t[1]].y - p[t[0]].y) * (p[t[2]].x - p[t[0]].x);
        float cq = (q[t[1]].x - q[t[0]].x) * (q[t[2]].y - q[t[0]].y) -
                   (q[t[1]].y - q[t[0]].y) * (q[t[2]].x - q[t[0]].x);
        if (std::abs(cp) < 4.0f || std::abs(cq) < 4.0f) return false;
    }
    double A[8][9]{};
    for (int i = 0; i < 4; ++i) {
        double x = p[i].x, y = p[i].y, u = q[i].x, v = q[i].y;
        A[2 * i][0] = x; A[2 * i][1] = y; A[2 * i][2] = 1.0;
        A[2 * i][6] = -u * x; A[2 * i][7] = -u * y; A[2 * i][8] = u;
        A[2 * i + 1][3] = x; A[2 * i + 1][4] = y; A[2 * i + 1][5] = 1.0;
        A[2 * i + 1][6] = -v * x; A[2 * i + 1][7] = -v * y; A[2 * i + 1][8] = v;
    }
    double sol[8];
    if (!solveLinear8(A, sol)) return false;
    for (int i = 0; i < 8; ++i) H[i] = sol[i];
    H[8] = 1.0;
    return true;
}

bool solveHomographyLS(const std::vector<cv::Point2f>& P,
                       const std::vector<cv::Point2f>& Q,
                       const std::vector<int>& ids,
                       double H[9]) {
    if (ids.size() < 4) return false;
    double cxp = 0, cyp = 0, cxq = 0, cyq = 0;
    const double invN = 1.0 / ids.size();
    for (int idx : ids) {
        cxp += P[idx].x; cyp += P[idx].y;
        cxq += Q[idx].x; cyq += Q[idx].y;
    }
    cxp *= invN; cyp *= invN; cxq *= invN; cyq *= invN;
    double dp = 0, dq = 0;
    for (int idx : ids) {
        dp += std::hypot(P[idx].x - cxp, P[idx].y - cyp);
        dq += std::hypot(Q[idx].x - cxq, Q[idx].y - cyq);
    }
    dp *= invN; dq *= invN;
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
            for (int b = 0; b < 8; ++b) Nmat[a][b] += r1[a] * r1[b] + r2[a] * r2[b];
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
            for (int k = 0; k < 3; ++k) tmp[i * 3 + j] += Hn[i * 3 + k] * Tp[k * 3 + j];
    for (int i = 0; i < 9; ++i) H[i] = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) H[i * 3 + j] += Tqi[i * 3 + k] * tmp[k * 3 + j];
    if (std::abs(H[8]) < 1e-12) return false;
    double invH8 = 1.0 / H[8];
    for (int i = 0; i < 9; ++i) H[i] *= invH8;
    return true;
}

int countInliersH(const std::vector<cv::Point2f>& P,
                  const std::vector<cv::Point2f>& Q,
                  const double H[9], float thresh2) {
    const float h0 = float(H[0]), h1 = float(H[1]), h2 = float(H[2]);
    const float h3 = float(H[3]), h4 = float(H[4]), h5 = float(H[5]);
    const float h6 = float(H[6]), h7 = float(H[7]), h8 = float(H[8]);
    int count = 0;
    const int n = static_cast<int>(P.size());
    for (int i = 0; i < n; ++i) {
        const float px = P[i].x, py = P[i].y;
        const float d = h6 * px + h7 * py + h8;
        if (d <= 1e-6f) continue;
        const float invD = 1.0f / d;
        const float dx = (h0 * px + h1 * py + h2) * invD - Q[i].x;
        const float dy = (h3 * px + h4 * py + h5) * invD - Q[i].y;
        if (dx * dx + dy * dy <= thresh2) ++count;
    }
    return count;
}

// Spatially-guided NAPSAC + global RANSAC for multi-tile homography extraction.
// Grass tiles are spatially compact in image space, so sampling 3 neighbors near
// an anchor raises 4-point same-tile hit rate by orders of magnitude over uniform RANSAC.
int findAndRemoveHomographyNapsac(std::vector<cv::Point2f>& P,
                                  std::vector<cv::Point2f>& Q,
                                  int S, int w, int h,
                                  int minInliers,
                                  Quad& outQuad,
                                  bool& outValid,
                                  std::uint64_t& rng) {
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
        std::max(80.0f, std::min(w, h) * 0.12f),
        std::max(180.0f, std::min(w, h) * 0.28f),
        0.0f // global uniform fallback
    };
    const float r2Sq[2] = {radii2[0] * radii2[0], radii2[1] * radii2[1]};

    std::vector<int> pool;
    pool.reserve(n);
    double bestH[9]{};
    int bestCount = 0;
    int maxIters = 600;

    for (int iter = 0; iter < maxIters; ++iter) {
        int ids[4];
        const int mode = iter & 3; // 0,1: tight NAPSAC; 2: wide NAPSAC; 3: global
        if (mode < 3) {
            const float limit2 = (mode < 2) ? r2Sq[0] : r2Sq[1];
            ids[0] = static_cast<int>(nextRand() % static_cast<std::uint32_t>(n));
            const float ax = Q[ids[0]].x, ay = Q[ids[0]].y;
            pool.clear();
            for (int j = 0; j < n; ++j) {
                if (j == ids[0]) continue;
                float dx = Q[j].x - ax, dy = Q[j].y - ay;
                if (dx * dx + dy * dy <= limit2) pool.push_back(j);
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
            if (H[6] * cx + H[7] * cy + H[8] <= 1e-4) { badCorner = true; break; }
        }
        if (badCorner) continue;

        int cnt = countInliersH(P, Q, H, thresh2);
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
            int cntRef = countInliersH(P, Q, refined, thresh2);
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
        outQuad[i] = {(bestH[0] * x + bestH[1] * y + bestH[2]) / d,
                      (bestH[3] * x + bestH[4] * y + bestH[5]) / d};
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
} // namespace

std::vector<Quad> siftSeeds(const std::vector<float>& green, int w, int h,
                           const std::array<double, 256>& ref, double ratio,
                           FeatureTimings* timings) {
    if (!std::isfinite(ratio))
        throw std::invalid_argument("Invalid SIFT ratio");
    if (w < 8 || h < 8 ||
        static_cast<std::size_t>(w) * h > MaxInputPixels ||
        green.size() != static_cast<std::size_t>(w) * h)
        throw std::invalid_argument("Invalid SIFT grayscale dimensions or stride");
    constexpr int S = 256;
    const std::uint64_t green_hash = hashGreen(green);
    static thread_local SiftCache cache;
    const bool hit = cache.valid && cache.w == w && cache.h == h &&
                     cache.s == S && cache.green_hash == green_hash &&
                     cache.ref == ref;
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
        for (int i = 0; i < b.count; ++i) {
            const int bi = i * FeatureStride + 2;
            const float* bp = b.data.data() + bi;
            double best = std::numeric_limits<double>::infinity();
            double second = best;
            int jbest = 0;
            for (int j = 0; j < a.count; ++j) {
                const int aj = j * FeatureStride + 2;
                const float* ap = a.data.data() + aj;
#ifdef __wasm_simd128__
                v128_t acc0 = wasm_f32x4_splat(0.0f);
                v128_t acc1 = wasm_f32x4_splat(0.0f);
                float distance = 0.0f;
                const float cutoff = static_cast<float>(second);
                for (int k = 0; k < 128; k += 32) {
                    v128_t a0 = wasm_f32x4_sub(wasm_v128_load(bp + k), wasm_v128_load(ap + k));
                    v128_t a1 = wasm_f32x4_sub(wasm_v128_load(bp + k + 4), wasm_v128_load(ap + k + 4));
                    v128_t a2 = wasm_f32x4_sub(wasm_v128_load(bp + k + 8), wasm_v128_load(ap + k + 8));
                    v128_t a3 = wasm_f32x4_sub(wasm_v128_load(bp + k + 12), wasm_v128_load(ap + k + 12));
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(a0, a0));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(a1, a1));
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(a2, a2));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(a3, a3));
                    v128_t a4 = wasm_f32x4_sub(wasm_v128_load(bp + k + 16), wasm_v128_load(ap + k + 16));
                    v128_t a5 = wasm_f32x4_sub(wasm_v128_load(bp + k + 20), wasm_v128_load(ap + k + 20));
                    v128_t a6 = wasm_f32x4_sub(wasm_v128_load(bp + k + 24), wasm_v128_load(ap + k + 24));
                    v128_t a7 = wasm_f32x4_sub(wasm_v128_load(bp + k + 28), wasm_v128_load(ap + k + 28));
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(a4, a4));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(a5, a5));
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(a6, a6));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(a7, a7));
                    if (k == 64) {
                        v128_t sum = wasm_f32x4_add(acc0, acc1);
                        sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
                        sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
                        if (wasm_f32x4_extract_lane(sum, 0) >= cutoff) {
                            distance = cutoff;
                            break;
                        }
                    }
                }
                if (distance < cutoff) {
                    v128_t sum = wasm_f32x4_add(acc0, acc1);
                    sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
                    sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
                    distance = wasm_f32x4_extract_lane(sum, 0);
                }
#else
                double distance = 0;
                for (int k = 0; k < 128; k += 8) {
                    // JS converts both Float32 samples to double BEFORE subtracting.
                    const double v0 = static_cast<double>(bp[k]) - static_cast<double>(ap[k]);
                    const double v1 = static_cast<double>(bp[k + 1]) - static_cast<double>(ap[k + 1]);
                    const double v2 = static_cast<double>(bp[k + 2]) - static_cast<double>(ap[k + 2]);
                    const double v3 = static_cast<double>(bp[k + 3]) - static_cast<double>(ap[k + 3]);
                    const double v4 = static_cast<double>(bp[k + 4]) - static_cast<double>(ap[k + 4]);
                    const double v5 = static_cast<double>(bp[k + 5]) - static_cast<double>(ap[k + 5]);
                    const double v6 = static_cast<double>(bp[k + 6]) - static_cast<double>(ap[k + 6]);
                    const double v7 = static_cast<double>(bp[k + 7]) - static_cast<double>(ap[k + 7]);
                    distance += v0 * v0;
                    distance += v1 * v1;
                    distance += v2 * v2;
                    distance += v3 * v3;
                    distance += v4 * v4;
                    distance += v5 * v5;
                    distance += v6 * v6;
                    distance += v7 * v7;
                    if (distance >= second) break;
                }
#endif
                if (distance < best) {
                    second = best;
                    best = distance;
                    jbest = j;
                } else if (distance < second) second = distance;
            }
            matches[i] = {best, second, jbest};
        }
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
                             const std::array<double, 256>& ref,
                             FeatureTimings* timings) {
    PhaseTimer preparationTimer(timings ? &timings->extraction : nullptr);
    const auto image = prepareImage(green, w, h);
    cv::Mat img(image.height, image.width, CV_8UC1);
    for (std::size_t i = 0; i < image.data.size(); ++i)
        img.data[i] = jsUint8(image.data[i]);
    cv::Mat refSmall(16, 16, CV_8UC1);
    for (int i = 0; i < 256; ++i) refSmall.data[i] = jsUint8(ref[i]);
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
                cv::Ptr<cv::Feature2D> detector = (feat == 0)
                    ? cv::Ptr<cv::Feature2D>(akaze)
                    : cv::Ptr<cv::Feature2D>(orb);
                const auto& ks = (feat == 0) ? ksAkaze : ksOrb;
                const auto& ds = (feat == 0) ? dsAkaze : dsOrb;
                std::vector<cv::KeyPoint> kr;
                cv::Mat dr;
                detector->detectAndCompute(refBig, cv::noArray(), kr, dr);
                extractionTimer.stop();
                PhaseTimer matchingTimer(timings && dr.rows >= 4 && ds.rows >= 4
                    ? &timings->matchingHomography : nullptr);
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
                        minSide = std::min(minSide, std::hypot(q[i].x - q[(i + 1) % 4].x,
                                                            q[i].y - q[(i + 1) % 4].y));
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
} // namespace grass
