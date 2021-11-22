// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <limits>
#include "single_layer_tests/slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<SliceSpecificParams> shapes = {
        SliceSpecificParams{ { 1, 2, 12, 100 }, { 0, 1, 0, 1 }, { 1, 2, 5, 100 }, { 1, 1, 1, 10 }, {} },
        SliceSpecificParams{ { 1, 12, 100 }, { 0, 9, 0 }, { 1, 11, 1 }, { 1, 1, 1 }, {} },
        SliceSpecificParams{ { 1, 12, 100 }, { 0, 1, 0 }, { 10, -1, 10 }, { 1, 1, 1 }, {} },
        SliceSpecificParams{ { 2, 12, 100 }, { 1, 12, 100 }, { 0, 7, 0 }, { -1, -1, -1 }, {} },
        SliceSpecificParams{ { 2, 12, 100 }, { 1, 4, 99 }, { 0, 9, 0 }, { -1, 2, -1 }, {} },
        SliceSpecificParams{ { 2, 12, 100 }, { -1, -1, -1 }, { 0, 4, 0 }, { -1, -2, -1 }, {} },
        SliceSpecificParams{ { 2, 12, 100 }, { -1, -1, -1 }, { 0, 0, 4 }, { -1, -1, -1 }, {2, 0, 1} },
        SliceSpecificParams{ { 2, 12, 100 }, { 0, 0, 4 }, { -5, -1, -1 }, { 1, 2, 1 }, {2, 0, 1} },
        SliceSpecificParams{ { 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, {} },
        SliceSpecificParams{ { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, {} },
        SliceSpecificParams{ { 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 }, {} },
        SliceSpecificParams{ { 2, 2, 4, 2 }, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 }, {} },
        SliceSpecificParams{ { 1, 2, 4, 2 }, { 0, 1, 0, 1 }, { 10, 2, 4, 2 }, { 1, 1, 2, 1 }, {} },
        SliceSpecificParams{ { 10, 2, 4, 2 }, { 9, 1, 3, 0 }, { 0, 0, 0, 1 }, { -1, -1, -1, 1 }, {} },
        SliceSpecificParams{ { 10, 2, 4, 2 }, { 19, 1, -1, 0 }, { -10, 0, 0, -1 }, { -1, -1, -1, 1 }, {} },
        SliceSpecificParams{ { 3, 2, 4, 200 }, { 0, 1, -1, -1 }, { 3, 2, 0, 0 }, { 1, 1, -2, -1 }, {} },
        SliceSpecificParams{ { 2, 4, 5, 5, 68 }, { 0, 1, 0, 0, 0 }, {
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max() }, { 1, 1, 1, 1, 16 }, {} },
        SliceSpecificParams{ { 10, 12 }, { -1, 1 }, { -9999, 10 }, { -1, 1 }, {} },
        SliceSpecificParams{ { 5, 5, 5, 5 }, { -1, 0, -1, 0 }, { -50, -1, -60, -1 }, { -1, 1, -1, 1 }, {} },
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I64
};

INSTANTIATE_TEST_SUITE_P(
        smoke_BasicSlice, SliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(shapes),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::Values(std::map<std::string, std::string>())),
        SliceLayerTest::getTestCaseName);

}  // namespace
