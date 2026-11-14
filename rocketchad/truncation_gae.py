import torch


def full_precision_forward(model):
    def forward(observations):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            if observations.is_cuda:
                torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type=observations.device.type, enabled=False):
                return model(observations.float())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
    return forward


def gae(rewards, values, next_value, terminal, truncated=None, truncated_value=None, gamma=0.99, lam=0.95):
    if truncated is None:
        done = terminal
    else:
        if truncated_value is None or truncated_value.shape != values.shape:
            raise ValueError('truncated steps need successor values with the shape of values')
        truncated = truncated & ~terminal.bool()
        done = terminal.bool() | truncated
    alive = 1 - done.to(values.dtype)
    next_values = torch.cat((values[1:rewards.shape[0]], next_value.unsqueeze(0)), 0)[:rewards.shape[0]]
    boot = next_values * alive if truncated is None else torch.where(truncated, truncated_value, next_values * alive)
    delta = rewards + gamma * boot - values
    decay = gamma * lam * alive
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        running = delta[step] + decay[step] * running
        advantages[step] = running
    return advantages, advantages + values


class TimeoutValues:
    def __init__(self, model, horizon, rows, batch_size, device, owned=None, *,
                 obs_size=None, players=6, forward=None, episode_steps=None):
        self.forward = full_precision_forward(model) if forward is None else forward
        self.obs_size = model.obs_size if obs_size is None else obs_size
        self.players = players
        self.horizon = horizon
        self.batch_size, self.rows, self.owned = batch_size, rows, owned
        self.owned_mask = None
        if owned is not None:
            self.owned_mask = torch.zeros(rows, dtype=torch.bool, device=device)
            self.owned_mask.index_fill_(0, owned, True)
        self.player_offsets = torch.arange(players, device=device)
        self.values = torch.zeros((horizon, rows), dtype=torch.float32, device=device)
        self.truncated = torch.zeros((horizon, rows), dtype=torch.bool, device=device)
        self.block_rows = rows if owned is None else owned.numel()
        self.single_slot = None
        self.block_capacity = 0
        self.block_observations = None
        self.block_mask = None
        self.block_step = None
        self._configure(episode_steps)
        self.begin()

    def _configure(self, episode_steps):
        single = episode_steps is not None and episode_steps >= 1
        capacity = min(self.horizon, 32, episode_steps) if single else min(self.horizon, 32)
        if self.single_slot == single and self.block_capacity == capacity:
            return
        self.single_slot = single
        self.block_capacity = capacity
        self.block_observations = None
        device = self.values.device
        if single:
            self.block_step = torch.full((self.block_rows,), -1, dtype=torch.int32, device=device)
            self.block_mask = None
        else:
            self.block_mask = torch.empty((capacity, self.block_rows), dtype=torch.bool, device=device)
            self.block_step = None

    def begin(self, *, episode_steps=False):
        if episode_steps is not False:
            self._configure(episode_steps)
        self.values.zero_()
        self.truncated.zero_()
        if self.block_step is not None:
            self.block_step.fill_(-1)
        self.candidate_observations, self.candidate_positions = [], []
        self.candidate_steps = 0
        self.observations, self.positions = [], []
        self.pending = self.total_rows = self.forward_batches = self.peak_rows = 0
        self.block_count = 0

    def capture(self, step, observation, info, *, episode_steps=False):
        mask = info['timeout'].repeat_interleave(self.players)
        self.truncated[step].copy_(mask)
        timeout_arenas = info.get('timeout_arenas')
        if timeout_arenas is not None:
            if self.candidate_steps == self.block_capacity:
                self.collect_block()
            self.candidate_steps += 1
            if not timeout_arenas.numel():
                return
            observation = observation[..., :self.obs_size]
            indices = (timeout_arenas[:, None] * self.players + self.player_offsets).reshape(-1)
            self.candidate_observations.append(observation)
            self.candidate_positions.append(step * self.rows + indices)
            return
        if episode_steps is not False and (episode_steps is None or episode_steps < 1) and self.single_slot:
            self.collect_block()
            self._configure(None)
        if self.block_count == self.block_capacity:
            self.collect_block()
        observation = observation[..., :self.obs_size]
        if self.single_slot:
            if self.owned is not None:
                observation = observation.index_select(0, self.owned)
                mask = mask.index_select(0, self.owned)
            if self.block_observations is None:
                self.block_observations = observation.new_zeros((self.block_rows, *observation.shape[1:]))
            torch.where(mask.reshape(self.block_rows, *([1] * (observation.ndim - 1))),
                        observation, self.block_observations, out=self.block_observations)
            self.block_step.masked_fill_(mask, step)
            self.block_count += 1
        else:
            if self.block_observations is None:
                self.block_observations = observation.new_empty((self.block_capacity, self.block_rows, *observation.shape[1:]))
            if not self.block_count:
                self.block_start = step
            if self.owned is None:
                self.block_observations[self.block_count].copy_(observation)
                self.block_mask[self.block_count].copy_(mask)
            else:
                torch.index_select(observation, 0, self.owned, out=self.block_observations[self.block_count])
                torch.index_select(mask, 0, self.owned, out=self.block_mask[self.block_count])
            self.block_count += 1

    def _append(self, observations, positions):
        count = positions.numel()
        self.observations.append(observations)
        self.positions.append(positions)
        self.pending += count
        self.total_rows += count
        self.peak_rows = max(self.peak_rows, self.pending)
        if self.pending >= self.batch_size:
            self.flush_pending()

    def collect_block(self):
        if self.candidate_positions:
            positions = torch.cat(self.candidate_positions)
            keep = self.truncated.view(-1).index_select(0, positions)
            if self.owned_mask is not None:
                keep = keep & self.owned_mask.index_select(0, positions.remainder(self.rows))
            selected = keep.nonzero().flatten()
            if selected.numel():
                self._append(torch.cat(self.candidate_observations).index_select(0, selected),
                             positions.index_select(0, selected))
            self.candidate_observations.clear()
            self.candidate_positions.clear()
        self.candidate_steps = 0
        if not self.block_count:
            return
        if self.single_slot:
            selected = (self.block_step >= 0).nonzero().flatten()
            if selected.numel():
                steps = self.block_step.index_select(0, selected).to(torch.int64)
                indices = selected if self.owned is None else self.owned.index_select(0, selected)
                self._append(self.block_observations.index_select(0, selected), steps * self.rows + indices)
            self.block_step.fill_(-1)
        else:
            masks = self.block_mask[:self.block_count]
            selected = masks.flatten().nonzero().flatten()
            if selected.numel():
                offsets = selected.remainder(self.block_rows)
                indices = offsets if self.owned is None else self.owned.index_select(0, offsets)
                positions = (selected.div(self.block_rows, rounding_mode='floor') + self.block_start) * self.rows + indices
                observations = self.block_observations[:self.block_count].flatten(0, 1).index_select(0, selected)
                self._append(observations, positions)
        self.block_count = 0

    def flush(self):
        self.collect_block()
        self.flush_pending()

    def flush_pending(self):
        if not self.pending:
            return
        observations = torch.cat(self.observations)
        values = []
        for chunk in observations.split(self.batch_size):
            _, value = self.forward(chunk)
            values.append(value.float())
            self.forward_batches += 1
        self.values.view(-1)[torch.cat(self.positions)] = torch.cat(values)
        self.observations.clear()
        self.positions.clear()
        self.pending = 0
