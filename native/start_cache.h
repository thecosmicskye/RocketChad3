#pragma once

#include "rocketgpu/rocketgpu.h"

#include <cstddef>
#include <cstdint>

namespace start_cache {

constexpr int kCars = 6;
constexpr int kPads = 34;
constexpr size_t kMaxRows = 131072;

constexpr int kOverrideCar = 14;
constexpr int kOverrideFloats = kCars * kOverrideCar + 6;
constexpr size_t kMaxArenas = 20000;

struct Row {
    rocketgpu::CarState cars[kCars];
    rocketgpu::CarConfig configs[kCars];
    rocketgpu::BallState ball;
    rocketgpu::BoostPadState pads[kPads];
    int32_t cars_n;
};

void launch_apply(rocketgpu::DeviceView destination, size_t arena_count,
                  const Row* rows, size_t row_count, const int64_t* indices,
                  const uint8_t* reset_mask, rocketgpu::CarConfig default_config,
                  uint32_t* errors, cudaStream_t stream,
                  const float* overrides = nullptr);

}
