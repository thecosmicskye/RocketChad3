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
    def __init__(self, model, horizon, rows, batch_size, device, owned=None, *, obs_size=None, players=6, forward=None):
        self.forward = full_precision_forward(model) if forward is None else forward
        self.obs_size = model.obs_size if obs_size is None else obs_size
        self.players = players
        self.batch_size, self.rows, self.owned = batch_size, rows, owned
        self.owned_mask = None
        if owned is not None:
            self.owned_mask = torch.zeros(rows, dtype=torch.bool, device=device)
            self.owned_mask.index_fill_(0, owned, True)
        self.player_offsets = torch.arange(players, device=device)
        self.values = torch.zeros((horizon, rows), dtype=torch.float32, device=device)
        self.truncated = torch.zeros((horizon, rows), dtype=torch.bool, device=device)
        self.begin()

    def begin(self):
        self.values.zero_()
        self.truncated.zero_()
        self.observations, self.positions = [], []
        self.pending = self.total_rows = self.forward_batches = self.peak_rows = 0

    def capture(self, step, observation, info):
        observation = observation[..., :self.obs_size]
        mask = info['timeout'].repeat_interleave(self.players)
        self.truncated[step].copy_(mask)
        timeout_arenas = info.get('timeout_arenas')
        if timeout_arenas is not None:
            if not observation.shape[0]:
                return
            indices = (timeout_arenas[:, None] * self.players + self.player_offsets).reshape(-1)
            if self.owned_mask is not None:
                keep = self.owned_mask.index_select(0, indices)
                observation = observation[keep]
                indices = indices[keep]
        else:
            if self.owned_mask is not None:
                mask = mask & self.owned_mask
            indices = mask.nonzero().flatten()
            if not indices.numel():
                return
            observation = observation.index_select(0, indices)
        count = indices.numel()
        if not count:
            return
        self.observations.append(observation)
        self.positions.append(step * self.rows + indices)
        self.pending += count
        self.total_rows += count
        self.peak_rows = max(self.peak_rows, self.pending)
        if self.pending >= self.batch_size:
            self.flush_pending()

    def flush(self):
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
