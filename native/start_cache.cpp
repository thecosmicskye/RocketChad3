#include "start_cache.h"

#include <pybind11/pybind11.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rocketgpu/detail/step.hpp"

namespace py = pybind11;

namespace {

void Check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

bool Finite(rocketgpu::Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
bool Finite(rocketgpu::Quat q) {
    return std::isfinite(q.w) && std::isfinite(q.x) &&
           std::isfinite(q.y) && std::isfinite(q.z) &&
           q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z > 1e-12f;
}
bool Valid(const rocketgpu::CarState& s) {
    return Finite(s.pos) && Finite(s.rot) && Finite(s.vel) && Finite(s.ang_vel) &&
           Finite(s.flip_rel_torque) && std::isfinite(s.jump_time) &&
           std::isfinite(s.flip_time) && std::isfinite(s.air_time) &&
           std::isfinite(s.air_time_since_jump) && std::isfinite(s.boost) &&
           std::isfinite(s.time_since_boosted) && std::isfinite(s.boosting_time) &&
           std::isfinite(s.supersonic_time) && std::isfinite(s.handbrake_val) &&
           std::isfinite(s.auto_flip_timer) && std::isfinite(s.auto_flip_torque_scale) &&
           std::isfinite(s.demo_respawn_timer) && Finite(s.world_contact.contact_normal) &&
           std::isfinite(s.car_contact.cooldown_timer) && Finite(s.vel_impulse_cache);
}
bool Valid(const rocketgpu::BallState& s) {
    return Finite(s.pos) && Finite(s.rot) && Finite(s.vel) && Finite(s.ang_vel) &&
           Finite(s.vel_impulse_cache) && std::isfinite(s.hs_info.y_target_dir) &&
           std::isfinite(s.hs_info.cur_target_speed) &&
           std::isfinite(s.hs_info.time_since_hit) &&
           std::isfinite(s.ds_info.accumulated_hit_force) &&
           std::isfinite(s.ds_info.y_target_dir);
}
bool Valid(const rocketgpu::CarConfig& c) {
    return Finite(c.hitbox_size) && c.hitbox_size.x > 0 && c.hitbox_size.y > 0 &&
           c.hitbox_size.z > 0 && Finite(c.hitbox_offset) &&
           Finite(c.front_wheels_offset) && Finite(c.back_wheels_offset) &&
           std::isfinite(c.front_wheel_radius) && c.front_wheel_radius > 0 &&
           std::isfinite(c.back_wheel_radius) && c.back_wheel_radius > 0 &&
           std::isfinite(c.front_suspension_rest) && c.front_suspension_rest >= 0 &&
           std::isfinite(c.back_suspension_rest) && c.back_suspension_rest >= 0 &&
           std::isfinite(c.dodge_deadzone) && c.dodge_deadzone >= 0 &&
           c.dodge_deadzone <= 1 && c.three_wheels <= 1;
}
bool Valid(const rocketgpu::BoostPadState& p) {
    return std::isfinite(p.cooldown) && p.cooldown >= 0 &&
           p.is_active <= 1 && p.is_big <= 1;
}

py::sequence Sequence(py::handle value, size_t expected, const char* name) {
    if (!py::isinstance<py::sequence>(value))
        throw py::value_error("");
    py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
    if (static_cast<size_t>(py::len(seq)) != expected)
        throw py::value_error("");
    return seq;
}

void CheckPointer(const void* pointer, int device, const char* name) {
    cudaPointerAttributes attributes{};
    Check(cudaPointerGetAttributes(&attributes, pointer), name);
    if (attributes.device != device ||
        (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged))
        throw py::value_error("");
}

class StartCache {
public:
    StartCache(py::sequence cars, py::sequence balls, py::sequence configs,
               py::sequence pads, uintptr_t stream_address, const rocketgpu::SimConfig* sim_config)
        : stream_(reinterpret_cast<cudaStream_t>(stream_address)),
          default_config_(rocketgpu::OctaneConfig()) {
        const size_t count = static_cast<size_t>(py::len(cars));
        if (!count || count > start_cache::kMaxRows)
            throw py::value_error("");
        Sequence(balls, count, "balls");
        Sequence(configs, count, "configs");
        Sequence(pads, count, "pads");
        std::vector<start_cache::Row> host(count);
        std::optional<rocketgpu::MutatorConfig> host_mutator;
        if (sim_config) {
            host_mutator = sim_config->mutator;
            host_mutator->native_sphere_rsqrt_table = nullptr;
        }
        for (size_t row = 0; row < count; ++row) {
            const size_t row_n = static_cast<size_t>(py::len(cars[row]));
            if (row_n != 2 && row_n != 4 && row_n != static_cast<size_t>(start_cache::kCars))
                throw py::value_error("");
            const auto row_cars = Sequence(cars[row], row_n, "cars[row]");
            const auto row_configs = Sequence(configs[row], row_n, "configs[row]");
            const auto row_pads = Sequence(pads[row], start_cache::kPads, "pads[row]");
            host[row].cars_n = static_cast<int32_t>(row_n);
            for (int car = 0; car < static_cast<int>(row_n); ++car) {
                auto state = py::cast<rocketgpu::CarState>(row_cars[car]);
                auto config = py::cast<rocketgpu::CarConfig>(row_configs[car]);
                if (!Valid(state) || !Valid(config))
                    throw py::value_error("");
                rocketgpu::CanonicalizeImportedCarState(state);
                if (sim_config) {
                    auto reported = state;
                    rocketgpu::detail::RestoreStepCarFields(reported);
                    rocketgpu::detail::CalInternalTimersFromSetState(reported, *host_mutator, sim_config->tick_dt);
                    rocketgpu::detail::ReportRLBotCarFields(reported, nullptr, *host_mutator, sim_config->tick_dt);
                    state.air_state = reported.air_state;
                    state.dodge_timeout = reported.dodge_timeout;
                    state.dodge_elapsed = reported.dodge_elapsed;
                    state.dodge_dir_x = reported.dodge_dir_x;
                    state.dodge_dir_y = reported.dodge_dir_y;
                }
                host[row].cars[car] = state;
                host[row].configs[car] = config;
            }
            auto ball = py::cast<rocketgpu::BallState>(balls[row]);
            if (!Valid(ball)) throw py::value_error("");
            rocketgpu::CanonicalizeImportedBallState(ball);
            host[row].ball = ball;
            for (int pad = 0; pad < start_cache::kPads; ++pad) {
                auto state = py::cast<rocketgpu::BoostPadState>(row_pads[pad]);
                if (!Valid(state)) throw py::value_error("");
                host[row].pads[pad] = state;
            }
        }
        Check(cudaGetDevice(&device_), "cudaGetDevice");
        unsigned flags = 0;
        Check(cudaStreamGetFlags(stream_, &flags), "cudaStreamGetFlags");
        try {
            Check(cudaMalloc(reinterpret_cast<void**>(&rows_), count * sizeof(start_cache::Row)), "cudaMalloc(rows)");
            Check(cudaMalloc(reinterpret_cast<void**>(&errors_), sizeof(uint32_t)), "cudaMalloc(errors)");
            Check(cudaMemcpyAsync(rows_, host.data(), count * sizeof(start_cache::Row),
                                  cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync(rows)");
            Check(cudaMemsetAsync(errors_, 0, sizeof(uint32_t), stream_), "cudaMemsetAsync(errors)");

            Check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(constructor)");
        } catch (...) {
            if (errors_) cudaFree(errors_);
            if (rows_) cudaFree(rows_);
            throw;
        }
        count_ = count;
    }

    ~StartCache() {

        cudaSetDevice(device_);
        cudaStreamSynchronize(stream_);
        if (errors_) cudaFree(errors_);
        if (rows_) cudaFree(rows_);
    }

    StartCache(const StartCache&) = delete;
    StartCache& operator=(const StartCache&) = delete;

    void Apply(rocketgpu::BatchSim& destination, uintptr_t indices_address,
               size_t count, uintptr_t stream_address, uintptr_t reset_mask_address,
               uintptr_t overrides_address) {
        if (stream_address != reinterpret_cast<uintptr_t>(stream_))
            throw py::value_error("");
        int current = -1;
        Check(cudaGetDevice(&current), "cudaGetDevice");
        if (current != device_) throw py::value_error("");
        if (!indices_address || count != destination.env_count() || !count ||
            count > start_cache::kMaxArenas ||
            destination.car_capacity() != count * start_cache::kCars ||
            destination.boost_pad_capacity() != count * start_cache::kPads ||
            destination.config().mode != rocketgpu::Mode::Soccar ||
            destination.config().max_cars_per_env != start_cache::kCars ||
            destination.config().boost_pad_count != start_cache::kPads)
            throw py::value_error("");
        const auto view = destination.device_view();
        if (!view.car_states || !view.car_configs || !view.controls || !view.teams ||
            !view.ball_states || !view.boost_pad_states || !view.env_car_count)
            throw py::value_error("");
        CheckPointer(reinterpret_cast<const void*>(indices_address), device_, "indices_ptr");
        if (reset_mask_address)
            CheckPointer(reinterpret_cast<const void*>(reset_mask_address), device_,
                         "reset_mask_ptr");
        if (overrides_address)
            CheckPointer(reinterpret_cast<const void*>(overrides_address), device_, "overrides_ptr");
        CheckPointer(view.car_states, device_, "destination.car_states");
        start_cache::launch_apply(view, count, rows_, count_,
                                  reinterpret_cast<const int64_t*>(indices_address),
                                  reinterpret_cast<const uint8_t*>(reset_mask_address),
                                  default_config_, errors_, stream_,
                                  reinterpret_cast<const float*>(overrides_address));
        Check(cudaGetLastError(), "start cache kernel launch");
        inspected_ = false;
    }

    void Synchronize() const {
        Check(cudaSetDevice(device_), "cudaSetDevice");
        Check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(start cache)");
    }

    uint32_t CheckErrors() {
        Synchronize();
        uint32_t status = 0;
        Check(cudaMemcpy(&status, errors_, sizeof(status), cudaMemcpyDeviceToHost),
              "cudaMemcpy(start cache errors)");
        inspected_ = true;
        return status;
    }

    void ClearErrors() {
        if (!inspected_)
            throw py::value_error("");
        Check(cudaSetDevice(device_), "cudaSetDevice");
        Check(cudaMemsetAsync(errors_, 0, sizeof(uint32_t), stream_),
              "cudaMemsetAsync(clear start cache errors)");
        inspected_ = false;
    }

    size_t Size() const { return count_; }

private:
    start_cache::Row* rows_ = nullptr;
    uint32_t* errors_ = nullptr;
    size_t count_ = 0;
    int device_ = -1;
    cudaStream_t stream_ = nullptr;
    bool inspected_ = false;
    rocketgpu::CarConfig default_config_;
};

}

PYBIND11_MODULE(_rocketgpu_start_cache, module) {
    py::class_<StartCache>(module, "StartCache")
        .def(py::init<py::sequence, py::sequence, py::sequence, py::sequence, uintptr_t, const rocketgpu::SimConfig*>(),
             py::arg("cars"), py::arg("balls"), py::arg("configs"), py::arg("pads"),
             py::arg("stream") = 0, py::arg("sim_config") = nullptr)
        .def("apply", &StartCache::Apply, py::arg("destination"), py::arg("indices_ptr"),
             py::arg("count"), py::arg("stream"), py::arg("reset_mask_ptr") = 0,
             py::arg("overrides_ptr") = 0)
        .def("synchronize", &StartCache::Synchronize)
        .def("check_errors", &StartCache::CheckErrors)
        .def("clear_errors", &StartCache::ClearErrors)
        .def_property_readonly("size", &StartCache::Size);
    module.attr("ERROR_INVALID_INDEX") = 1;
    module.attr("ERROR_WRONG_SHAPE") = 2;
    module.attr("ERROR_NONRESET_IMPORT") = 4;
    module.attr("ERROR_INVALID_RESET_MASK") = 8;
    module.attr("MAX_ROWS") = start_cache::kMaxRows;
    module.attr("OVERRIDE_FLOATS") = start_cache::kOverrideFloats;
    module.attr("TEAM_SIZES") = 1;
}
