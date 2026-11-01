import torch


class NativeReplayStarts:
    def __init__(self, env, cars, balls, configs, pads, fraction=0.5):
        import rocketgpu_py
        import rocketgpu_py_cuda as views
        from _rocketgpu_start_cache import StartCache

        self.env = env
        self.fraction = fraction
        rows = [i for i, row in enumerate(cars) if len(row) == env.players]
        self.cache = StartCache(*([values[i] for i in rows] for values in (cars, balls, configs, pads)),
                                env.stream, env.sim.config)
        self.indices = torch.empty(env.arenas, dtype=torch.int64, device=env.device)
        self.selected = torch.empty(env.arenas, dtype=torch.bool, device=env.device)
        entry = env.sim.device_arrays()['car_configs']
        ptr = entry['__cuda_array_interface__']['data'][0]
        size = entry['itemsize']
        view = views.DeviceFieldArray('car_configs', 'bytes', '|u1', ptr, entry['count'], size, 0, (size,))
        self.configs = torch.as_tensor(view, device=env.device).reshape(env.arenas, -1)
        self.kickoff_configs = self.configs.clone()

    def __call__(self, reset):
        rows = torch.randint(self.cache.size, (self.env.arenas,), device=self.env.device)
        self.selected.copy_(reset & (torch.rand(self.env.arenas, device=self.env.device) < self.fraction))
        self.indices.copy_(torch.where(self.selected, rows, -1))
        self.cache.apply(self.env.sim, self.indices.data_ptr(), self.env.arenas,
                         self.env.stream, self.selected.data_ptr())
        self.configs.copy_(torch.where((reset & ~self.selected)[:, None], self.kickoff_configs, self.configs))
