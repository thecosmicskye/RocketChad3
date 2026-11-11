import inspect
import rocketgpu_py as rg
import rocketgpu_py_cuda as views
import torch


class _Bytes:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = dict(data=(ptr, False), shape=(nbytes,), typestr='|u1', version=3)


class Environment:
    def __init__(self, config, example, seed, device, *, initialize=True, generation=0):
        self.config = config
        self.arenas = config['arenas']
        self.tick_skip, self.device = config['tick_skip'], torch.device(device)
        self.seed = seed if not generation else (seed + generation * 1000000007) & ((1 << 63) - 1)
        self.team_layout = None
        fractions = config.get('team_size_fractions')
        if fractions is not None:
            from .team_layout import TeamLayout, mixed_counts
            fractions = {int(key): value for key, value in fractions.items()}
            generator = torch.Generator().manual_seed(seed)
            sizes = torch.full((self.arenas,), 3, dtype=torch.int64)
            default_fraction = 0.25 if config.get('opponents') else 0
            league_count = 2 * round(self.arenas * config.get('league_fraction', default_fraction) / 2)
            for start, count in ((0, self.arenas - league_count), (self.arenas - league_count, league_count)):
                if count:
                    counts = mixed_counts(count, fractions)
                    pairs = torch.randperm(count // 2, generator=generator)
                    offset = 0
                    for mode in (1, 2):
                        length = counts[mode] // 2
                        selected = start + (pairs[offset:offset + length, None] * 2 + torch.arange(2)).flatten()
                        sizes[selected] = mode
                        offset += length
            self.team_layout = TeamLayout(sizes.to(self.device))
        self.players = 6 if self.team_layout is not None else 2 * config['team_size']
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        settings = rg.SimConfig()
        settings.env_count, settings.max_cars_per_env = self.arenas, self.players
        settings.enable_boost_pads = settings.enable_car_car_collisions = settings.enable_demos = settings.enable_scoring = 1
        settings.tick_dt = 1 / 120
        self.goal_threshold = settings.mutator.goal_base_threshold_y + settings.mutator.ball_radius
        self.sim = rg.BatchSim(settings)
        if self.team_layout is None:
            self.sim.set_env_car_count([self.players] * self.arenas, self.stream)
            self.sim.set_teams(([rg.Team.Blue] * (self.players // 2) + [rg.Team.Orange] * (self.players // 2)) * self.arenas, self.stream)
        else:
            team_sizes = self.team_layout.sizes.cpu().tolist()
            self.sim.set_env_car_count([2 * size for size in team_sizes], self.stream)
            self.sim.set_teams([team for size in team_sizes for team in [rg.Team.Blue] * size + [rg.Team.Orange] * (6 - size)], self.stream)
            states = [rg.EnvState() for _ in range(self.arenas)]
            for state in states:
                state.car_visit_order_mode = 0
            self.sim.upload_env_states(states, self.stream)
        self.sim.reset_to_kickoff(self.seed, self.stream)

        def fields(name):
            # Each field views one tensor over the whole struct array, so its storage offset carries its alignment:
            # compiled code takes a zero-offset input to be 16-byte aligned and otherwise copies the array every call.
            entry = self.sim.device_arrays()[name]
            raw = torch.as_tensor(_Bytes(entry['__cuda_array_interface__']['data'][0], entry['itemsize'] * entry['count']),
                                  device=self.device)
            out = {}
            for key, value in views.field_arrays(self.sim, name).items():
                field = torch.as_tensor(value, device=self.device)
                out[key] = raw.view(field.dtype).as_strided(field.shape, field.stride(),
                                                            (field.data_ptr() - raw.data_ptr()) // field.element_size())
            return out

        self.car, self.ball = fields('car_states'), fields('ball_states')
        self.pads, self.state, self.controls = fields('boost_pad_states'), fields('env_states'), fields('controls')
        self.goal_telemetry = None
        self.goal_events = None
        if config.get('goal_telemetry', False):
            from .goal_telemetry import GoalTelemetry
            self.goal_telemetry = GoalTelemetry(self.sim, self.device, self.tick_skip)
        self.actions = example['action_table'](self.device)
        self.observation = example['Observation'](self)
        arenas_param = inspect.signature(self.observation).parameters.get('arenas')
        self.subset_observe = (arenas_param is not None and arenas_param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
        self.reward = example['reward']
        self.terminal_condition = example['terminal_condition']
        self.last_action = torch.zeros((self.arenas, self.players, 8), device=self.device)
        self.episode_steps = round(config['episode_seconds'] * 120 / self.tick_skip)
        self.age = torch.zeros(self.arenas, dtype=torch.int64, device=self.device)
        self.goal_inside = torch.zeros(self.arenas, dtype=torch.bool, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.timeouts = torch.randint(1, self.episode_steps + 1, (self.arenas,), generator=generator, device=self.device)
        self.reset_count = torch.zeros_like(self.age)
        self.env_ids = torch.arange(self.arenas, device=self.device, dtype=torch.int64)
        self.commands = torch.zeros((self.arenas, 2), dtype=torch.int64, device=self.device)
        self.opponents = None
        self.fixed_timeouts = True
        self.set_state = example['make_state_setter'](self)
        if initialize:
            self.initialize()

    def initialize(self):
        age_ref, timeouts_ref = self.age, self.timeouts
        age_version, timeouts_version = self.age._version, self.timeouts._version
        with torch.no_grad():
            self.set_state(torch.ones(self.arenas, dtype=torch.bool, device=self.device))
        self.fixed_timeouts = (self.age is age_ref and self.timeouts is timeouts_ref and
                               self.age._version == age_version and self.timeouts._version == timeouts_version)

    def timeout_schedule(self, steps):
        if not self.subset_observe or not self.fixed_timeouts or self.episode_steps < 1:
            return None
        window = min(steps, 32, self.episode_steps)
        bucket = (self.timeouts - self.age - 1).clamp(0, window)
        order = bucket.argsort(stable=True)
        counts = torch.zeros(window + 1, dtype=torch.int64, device=self.device)
        counts.scatter_add_(0, bucket, torch.ones_like(bucket))
        sizes = counts[:window].tolist()
        return order[:sum(sizes)].split(sizes)

    def observe(self):
        return self.observation(self)

    def logical_car_field(self, name):
        value = self.car[name]
        if self.team_layout is None:
            return value
        shaped = value.reshape(self.arenas, 6, *value.shape[1:])
        signed = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}.get(value.dtype)
        if signed is not None:
            shaped = shaped.view(signed)
        result = self.team_layout.to_logical(shaped).reshape(value.shape)
        return result if signed is None else result.view(value.dtype)

    def step(self, actions, phase, *, capture_transition=False, capture_info=False):
        previous_blue = self.state['blue_score'].to(torch.int64)
        previous_orange = self.state['orange_score'].to(torch.int64)
        controls = self.actions[actions]
        self.last_action.copy_(controls.reshape(self.arenas, self.players, 8))
        if self.team_layout is not None:
            controls = self.team_layout.to_physical(self.last_action).reshape(-1, 8)
        for index, name in enumerate(('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake')):
            self.controls[name].copy_(controls[:, index])
        if self.goal_telemetry is not None:
            self.goal_telemetry.begin()
        self.sim.step(self.tick_skip, self.stream)
        blue_score = self.state['blue_score'].to(torch.int64)
        orange_score = self.state['orange_score'].to(torch.int64)
        blue = (blue_score > previous_blue) & ~self.goal_inside
        orange = (orange_score > previous_orange) & ~self.goal_inside
        if self.goal_telemetry is not None:
            self.goal_events = self.goal_telemetry.read(blue | orange)
        reward = self.reward(self, blue, orange, phase)
        self.age.add_(1)
        terminal = self.terminal_condition(self, blue, orange)
        timeout = (self.age >= self.timeouts) & ~terminal
        done = terminal | timeout
        self.goal_inside.copy_((self.ball['pos'][:, 1].abs() > self.goal_threshold) & ~done)
        want_transition = isinstance(capture_transition, torch.Tensor) or bool(capture_transition)
        if want_transition or capture_info:
            info = dict(timeout=timeout, blue_goals=blue, orange_goals=orange)
            if isinstance(capture_transition, torch.Tensor):
                info['timeout_arenas'] = capture_transition
                info['transition_observation'] = (self.observation(self, arenas=capture_transition)
                                                  if capture_transition.numel() else None)
            elif want_transition:
                info['transition_observation'] = self.observe().clone()
        self.reset_count.add_(done)
        self.commands[:, 0] = done.to(torch.int64) * ((1 << 0) | (1 << 8) | (1 << 16))
        self.commands[:, 1] = (self.seed + self.env_ids * 1000003 + self.reset_count * 9176) & 9223372036854775807
        self.sim.apply_env_commands_device(self.commands.data_ptr(), self.arenas, self.stream)
        self.age.masked_fill_(done, 0)
        self.timeouts.masked_fill_(done, self.episode_steps)
        self.last_action.masked_fill_(done[:, None, None], 0)
        age_ref, timeouts_ref = self.age, self.timeouts
        age_version, timeouts_version = self.age._version, self.timeouts._version
        self.set_state(done)
        if not (self.age is age_ref and self.timeouts is timeouts_ref and
                self.age._version == age_version and self.timeouts._version == timeouts_version):
            self.fixed_timeouts = False
        result = (self.observe(), reward, done.repeat_interleave(self.players))
        return (*result, info) if want_transition or capture_info else result

    def close(self):
        torch.cuda.synchronize(self.device)
        self.set_state = None
        self.opponents = None
        for fields in (self.car, self.ball, self.pads, self.state, self.controls):
            fields.clear()
        self.goal_events = self.goal_telemetry = None
        self.sim = None
