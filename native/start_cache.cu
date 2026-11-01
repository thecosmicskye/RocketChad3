#include "start_cache.h"
#include "rocketgpu/detail/math.hpp"

namespace start_cache {
namespace {

__device__ inline float Canon(float v) { return (v * (1.0f / 50.0f)) * 50.0f; }
__device__ inline float CanonBt(float v) { return Canon(v) * (1.0f / 50.0f); }
__device__ inline rocketgpu::Vec3 CanonBt3(float x, float y, float z) {
    return rocketgpu::Vec3(CanonBt(x), CanonBt(y), CanonBt(z));
}
__device__ inline rocketgpu::Vec3 Uu(const rocketgpu::Vec3& bt) {
    return rocketgpu::Vec3(bt.x * 50.0f, bt.y * 50.0f, bt.z * 50.0f);
}

__device__ inline bool LayoutOk(const rocketgpu::DeviceView& view, size_t arena, int live) {
    if (live != 2 && live != 4 && live != kCars) return false;
    for (int car = 0; car < kCars; ++car) {
        if (view.teams[arena * kCars + car] !=
            (car < live / 2 ? rocketgpu::Team::Blue : rocketgpu::Team::Orange)) return false;
    }
    return true;
}

__global__ void Validate(rocketgpu::DeviceView view, size_t arena_count,
                         const Row* rows, size_t row_count, const int64_t* indices,
                         const uint8_t* reset_mask, uint32_t* errors) {
    const size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (arena >= arena_count) return;
    uint32_t status = 0;
    const int64_t index = indices[arena];
    const uint8_t reset = reset_mask ? reset_mask[arena] : 0;
    if (reset > 1) status |= 8;
    if (index >= 0 && reset_mask && reset != 1) status |= 4;
    if (index != -1 &&
        (index < 0 || static_cast<uint64_t>(index) >= row_count)) status |= 1;
    const int live = static_cast<int>(view.env_car_count[arena]);
    if (index == -1 && reset == 1 && !LayoutOk(view, arena, live)) status |= 2;
    if (index >= 0 && static_cast<uint64_t>(index) < row_count &&
        (!LayoutOk(view, arena, live) || rows[index].cars_n != live)) status |= 2;

    if (status) atomicOr(errors, status);
}

__global__ void Scatter(rocketgpu::DeviceView view, size_t arena_count,
                        const Row* rows, const int64_t* indices,
                        const uint8_t* reset_mask, rocketgpu::CarConfig default_config,
                        const uint32_t* errors, const float* overrides) {
    const size_t arena = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (arena >= arena_count || *errors) return;
    const int64_t index = indices[arena];
    if (index == -1) {
        if (reset_mask && reset_mask[arena] == 1) {
            for (int car = 0; car < kCars; ++car)
                view.car_configs[arena * kCars + car] = default_config;
        }
        return;
    }
    const Row& row = rows[index];
    const float* o = overrides ? overrides + arena * kOverrideFloats : nullptr;
    for (int car = 0; car < row.cars_n; ++car) {
        const size_t slot = arena * kCars + car;
        rocketgpu::CarState state = row.cars[car];
        if (o) {
            const float* c = o + car * kOverrideCar;
            state.pos_bt = CanonBt3(c[0], c[1], c[2]);
            state.pos = Uu(state.pos_bt);
            state.vel_bt = CanonBt3(c[3], c[4], c[5]);
            state.vel = rocketgpu::Vec3(Canon(c[3]), Canon(c[4]), Canon(c[5]));
            state.ang_vel = rocketgpu::Vec3(c[6], c[7], c[8]);
            state.rot = rocketgpu::Quat(c[9], c[10], c[11], c[12]);
            state.basis_authoritative = 0;
            rocketgpu::detail::CanonicalizeImportedCarOrientation(state);
            state.boost = c[13];
        }
        state.is_supersonic = rocketgpu::detail::LengthSq(state.vel) >= 2200.0f * 2200.0f;
        view.car_states[slot] = state;
        view.car_configs[slot] = row.configs[car];
        view.controls[slot] = {};
    }
    rocketgpu::BallState ball = row.ball;
    if (o) {
        const float* b = o + kCars * kOverrideCar;
        ball.pos_bt = CanonBt3(b[0], b[1], b[2]);
        ball.vel_bt = CanonBt3(b[3], b[4], b[5]);
        ball.pos = Uu(ball.pos_bt);
        ball.vel = Uu(ball.vel_bt);
    }
    view.ball_states[arena] = ball;
    for (int pad = 0; pad < kPads; ++pad)
        view.boost_pad_states[arena * kPads + pad] = row.pads[pad];
}

}

void launch_apply(rocketgpu::DeviceView destination, size_t arena_count,
                  const Row* rows, size_t row_count, const int64_t* indices,
                  const uint8_t* reset_mask, rocketgpu::CarConfig default_config,
                  uint32_t* errors, cudaStream_t stream, const float* overrides) {
    Validate<<<unsigned((arena_count + 127) / 128), 128, 0, stream>>>(
        destination, arena_count, rows, row_count, indices, reset_mask, errors);
    rocketgpu::CheckCuda(cudaPeekAtLastError(), "start cache validation launch");
    Scatter<<<unsigned((arena_count + 127) / 128), 128, 0, stream>>>(
        destination, arena_count, rows, indices, reset_mask, default_config, errors, overrides);
    rocketgpu::CheckCuda(cudaPeekAtLastError(), "start cache scatter launch");
}

}
