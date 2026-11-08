#pragma once

#include <array>
#include <vector>
#include <opencv2/core/types.hpp>

namespace grass {
using Point = cv::Point2d;
using Quad = std::array<Point, 4>;

struct FeatureOptions {
    double siftRatio = .68;
};

// Accumulated milliseconds; untouched phases stay zero on a fresh instance.
struct FeatureTimings {
    double extraction = 0;
    double matchingHomography = 0;
};

// ref[y * 16 + x] contains the original numeric (double) reference sample.
// Errors throw; an empty list of feature hypotheses is a successful result.
// VLFeat's extract/get_out use shared buffers, so callers must serialize calls.
std::vector<Quad> siftSeeds(const std::vector<float>& green, int w, int h,
                           const std::array<double, 256>& ref, double ratio,
                           FeatureTimings* timings = nullptr);
std::vector<Quad> binarySeeds(const std::vector<float>& green, int w, int h,
                             const std::array<double, 256>& ref,
                             FeatureTimings* timings = nullptr);
} // namespace grass
