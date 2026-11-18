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
    for (int it = 0; it < 20 && P.size() >= 10; ++it) {
        cv::Mat mask;
        const cv::Mat H = cv::findHomography(P, Q, cv::RANSAC, 2.5, mask, 2500, .99);
        if (H.empty() || removeInliers(P, Q, mask, 8) < 8) break;
        const Quad q = project(H, S);
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
