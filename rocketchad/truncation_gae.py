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
    def __init__(self, model, horizon, rows, batch_size, device, owned=None, *, obs_size=None, players=6):
        self.forward = full_precision_forward(model)
        self.obs_size = model.obs_size if obs_size is None else obs_size
        self.players = players
        self.batch_size, self.rows, self.owned = batch_size, rows, owned
        self.values = torch.zeros((horizon, rows), dtype=torch.float32, device=device)
        self.truncated = torch.zeros((horizon, rows), dtype=torch.bool, device=device)
        self.block_capacity = min(horizon, 32)
        self.block_rows = rows if owned is None else owned.numel()
        self.block_mask = torch.empty((self.block_capacity, self.block_rows), dtype=torch.bool, device=device)
        self.block_observations = None
        self.begin()

    def begin(self):
        self.values.zero_()
        self.truncated.zero_()
        self.observations, self.positions = [], []
        self.pending = self.total_rows = self.forward_batches = self.peak_rows = 0
        self.block_count = 0

    def capture(self, step, observation, info):
        observation = observation[..., :self.obs_size]
        mask = info['timeout'].repeat_interleave(self.players)
        self.truncated[step].copy_(mask)
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
        if self.block_count == self.block_capacity:
            self.collect_block()

    def collect_block(self):
        if not self.block_count:
            return
        masks = self.block_mask[:self.block_count]
        selected = masks.flatten().nonzero().flatten()
        if selected.numel():
            offsets = selected.remainder(self.block_rows)
            indices = offsets if self.owned is None else self.owned.index_select(0, offsets)
            positions = (selected.div(self.block_rows, rounding_mode='floor') + self.block_start) * self.rows + indices
            observations = self.block_observations[:self.block_count].flatten(0, 1).index_select(0, selected)
            self.observations.append(observations)
            self.positions.append(positions)
            self.pending += selected.numel()
            self.total_rows += selected.numel()
            self.peak_rows = max(self.peak_rows, self.pending)
            if self.pending >= self.batch_size:
                self.flush_pending()
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
