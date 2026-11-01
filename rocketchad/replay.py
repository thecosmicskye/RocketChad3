import torch


class NativeReplayStarts:
    def __init__(self, env, cars, balls, configs, pads, fraction=0.5, *, jitter_rows=None, jitter_eligible=None):
        import rocketgpu_py
        import rocketgpu_py_cuda as views
        from _rocketgpu_start_cache import StartCache

        self.env = env
        self.fraction = fraction
        rows = [i for i, row in enumerate(cars) if len(row) == env.players or
                (env.team_layout is not None and len(row) in (2, 4, 6))]
        self.cache = StartCache(*([values[i] for i in rows] for values in (cars, balls, configs, pads)),
                                env.stream, env.sim.config)
        self.mode_rows = None
        if env.team_layout is not None:
            self.mode_rows = {}
            for mode in (1, 2, 3):
                indices = [j for j, row in enumerate(rows) if len(cars[row]) == 2 * mode]
                if indices:
                    self.mode_rows[mode] = torch.tensor(indices, dtype=torch.int64, device=env.device)
        self.jitter = self.jitter_eligible = None
        if jitter_rows is not None:
            import numpy as np
            from .replay_jitter import Jitter
            eligible = np.ones(len(cars), dtype=bool) if jitter_eligible is None else np.asarray(jitter_eligible, dtype=bool)
            eligible = eligible[rows]
            if eligible.any():
                groups = []
                for row in rows:
                    value = np.asarray(jitter_rows[row])
                    if not groups or groups[-1][0].shape != value.shape:
                        groups.append([])
                    groups[-1].append(value)
                self.jitter = Jitter([np.stack(group) for group in groups], env.seed, env.device)
                self.jitter_eligible = torch.as_tensor(eligible, device=env.device)
        self.indices = torch.empty(env.arenas, dtype=torch.int64, device=env.device)
        self.selected = torch.empty(env.arenas, dtype=torch.bool, device=env.device)
        if self.jitter is not None:
            self.plain_indices = torch.empty_like(self.indices)
            self.jitter_indices = torch.empty_like(self.indices)
            self.plain_selected = torch.empty_like(self.selected)
            self.jitter_selected = torch.empty_like(self.selected)
        entry = env.sim.device_arrays()['car_configs']
        ptr = entry['__cuda_array_interface__']['data'][0]
        size = entry['itemsize']
        view = views.DeviceFieldArray('car_configs', 'bytes', '|u1', ptr, entry['count'], size, 0, (size,))
        self.configs = torch.as_tensor(view, device=env.device).reshape(env.arenas, -1)
        self.kickoff_configs = self.configs.clone()

    def __call__(self, reset):
        self.selected.copy_(reset & (torch.rand(self.env.arenas, device=self.env.device) < self.fraction))
        if self.mode_rows is None:
            rows = torch.randint(self.cache.size, (self.env.arenas,), device=self.env.device)
            self.indices.copy_(torch.where(self.selected, rows, -1))
        else:
            self.indices.fill_(-1)
            for mode, choices in self.mode_rows.items():
                pick = choices[torch.randint(choices.numel(), (self.env.arenas,), device=self.env.device)]
                self.indices.copy_(torch.where(self.selected & (self.env.team_layout.sizes == mode), pick, self.indices))
            self.selected.logical_and_(self.indices >= 0)
        if self.jitter is None:
            self.cache.apply(self.env.sim, self.indices.data_ptr(), self.env.arenas,
                             self.env.stream, self.selected.data_ptr())
        else:
            eligible = self.selected & self.jitter_eligible[self.indices.clamp(min=0)]
            self.jitter_selected.copy_(self.jitter.choose(eligible))
            self.plain_selected.copy_(self.selected & ~self.jitter_selected)
            self.plain_indices.copy_(torch.where(self.plain_selected, self.indices, -1))
            self.jitter_indices.copy_(torch.where(self.jitter_selected, self.indices, -1))
            overrides = self.jitter.overrides(self.indices, self.jitter_selected)
            self.cache.apply(self.env.sim, self.plain_indices.data_ptr(), self.env.arenas,
                             self.env.stream, self.plain_selected.data_ptr())
            self.cache.apply(self.env.sim, self.jitter_indices.data_ptr(), self.env.arenas,
                             self.env.stream, self.jitter_selected.data_ptr(), overrides.data_ptr())
        self.configs.copy_(torch.where((reset & ~self.selected)[:, None], self.kickoff_configs, self.configs))
