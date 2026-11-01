"""Example replay starts from states already converted to RocketGPU snapshots."""

# Decode .replay files separately; a parser is not included. Upload one frame
# per arena, with each arena's blue cars first and orange cars second:
# env.sim.upload_car_states(car_states, env.stream)
# env.sim.upload_ball_states(ball_states, env.stream)
# env.sim.upload_boost_pad_states(boost_pad_states, env.stream)
# torch.save(capture_states(env), 'replay_states.pt')
# Capture and train with the same RocketGPU build, team size and car configuration.
# To enable replay starts, uncomment the example in config.py.

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
    return {name: values.cpu() for name, values in _state_arrays(env).items()}


class ReplayStarts:
    def __init__(self, env, path, fraction=0.5):
        self.arenas, self.device, self.fraction = env.arenas, env.device, fraction
        self.arrays = _state_arrays(env)
        self.frames = torch.load(path, map_location=env.device, weights_only=True)
        self.count = len(self.frames['ball_states'])

    def __call__(self, reset):
        rows = torch.randint(self.count, (self.arenas,), device=self.device)
        selected = reset & (torch.rand(self.arenas, device=self.device) < self.fraction)
        for name, values in self.arrays.items():
            values.copy_(torch.where(selected[:, None], self.frames[name][rows], values))
