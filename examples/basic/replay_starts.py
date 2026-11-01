"""Example replay starts from states already converted to RocketGPU snapshots."""

# Decode .replay files separately; a parser is not included. Upload one frame
# per arena, with each arena's blue cars first and orange cars second:
# env.sim.upload_car_states(car_states, env.stream)
# env.sim.upload_ball_states(ball_states, env.stream)
# env.sim.upload_boost_pad_states(boost_pad_states, env.stream)
# torch.save(capture_states(env), 'replay_states.pt')
# Capture and train with the same RocketGPU build, car configuration and slot
# capacity. Mixed-mode capture stores team sizes to match starts to their mode.
# To enable replay starts, uncomment the example in config.py.
# For RocketChad2's native replay cache, build CMakeLists.txt with
# ROCKETCHAD_BUILD_NATIVE=ON and add the build directory and its rocketgpu/python
# directory to PYTHONPATH. NativeReplayStarts requires six car slots per arena.
# For each replay frame provide its blue-first cars, ball, car configurations
# and 34 boost pads as RocketGPU state objects, then use them in a state setter:
# from rocketchad.replay import NativeReplayStarts
# def make_state_setter(env):
#     return NativeReplayStarts(env, cars, balls, configs, pads, fraction=0.5)

import rocketgpu_py_cuda as views
import torch


def _state_arrays(env):
    arrays = env.sim.device_arrays()
    result = {}
    for name in ('car_states', 'ball_states', 'boost_pad_states'):
        entry = arrays[name]
        ptr = entry['__cuda_array_interface__']['data'][0]
        size = entry['itemsize']
        view = views.DeviceFieldArray(name, 'bytes', '|u1', ptr, entry['count'], size, 0, (size,))
        result[name] = torch.as_tensor(view, device=env.device).reshape(env.arenas, -1)
    return result


def capture_states(env):
    """Capture one converted RocketGPU frame per arena for torch.save."""
    states = {name: values.cpu() for name, values in _state_arrays(env).items()}
    if env.team_layout is not None:
        states['team_sizes'] = env.team_layout.sizes.cpu()
    return states


class ReplayStarts:
    def __init__(self, env, path, fraction=0.5):
        self.arenas, self.device, self.fraction = env.arenas, env.device, fraction
        self.arrays = _state_arrays(env)
        self.frames = torch.load(path, map_location=env.device, weights_only=True)
        self.count = len(self.frames['ball_states'])
        self.team_sizes = (env.team_layout.sizes if env.team_layout is not None else
                           torch.full((self.arenas,), env.players // 2, device=self.device)
                           if 'team_sizes' in self.frames else None)
        self.mode_rows = None if self.team_sizes is None else {
            mode: (self.frames['team_sizes'] == mode).nonzero().flatten()
            for mode in (1, 2, 3)
        }

    def __call__(self, reset):
        selected = reset & (torch.rand(self.arenas, device=self.device) < self.fraction)
        if self.mode_rows is None:
            rows = torch.randint(self.count, (self.arenas,), device=self.device)
        else:
            rows = torch.zeros(self.arenas, dtype=torch.int64, device=self.device)
            available = torch.zeros_like(selected)
            for mode, choices in self.mode_rows.items():
                if choices.numel():
                    match = self.team_sizes == mode
                    pick = choices[torch.randint(choices.numel(), (self.arenas,), device=self.device)]
                    rows = torch.where(match, pick, rows)
                    available |= match
            selected &= available
        for name, values in self.arrays.items():
            values.copy_(torch.where(selected[:, None], self.frames[name][rows], values))
