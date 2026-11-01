import torch


class EntropyBand:
    def __init__(self, config, device):
        band = config['entropy_band']
        self.low, self.high = float(band['low']), float(band['high'])
        self.up, self.down = float(band['up']), float(band['down'])
        self.ema_updates = int(band['ema_updates'])
        self.w0 = float(config['entropy'])
        clamp = float(band['clamp'])
        self.wmin, self.wmax = self.w0 / clamp, self.w0 * clamp
        self.weight, self.ema, self.updates, self.skipped = self.w0, None, 0, False
        self.tensor = torch.tensor(self.w0, dtype=torch.float32, device=device)

    def update(self, entropy):
        if not self.skipped:
            self.skipped = True
            return self.weight
        self.ema = entropy if self.ema is None else self.ema + (entropy - self.ema) / self.ema_updates
        if self.ema < self.low:
            self.weight *= self.up
        elif self.ema > self.high:
            self.weight *= self.down
        self.weight = min(self.wmax, max(self.wmin, self.weight))
        self.updates += 1
        self.tensor.fill_(self.weight)
        torch.cuda.synchronize(self.tensor.device)
        return self.weight

    def state_dict(self):
        return dict(weight=self.weight, ema=self.ema, updates=self.updates, skipped=self.skipped)

    def load_state_dict(self, state):
        self.weight, self.ema = state['weight'], state['ema']
        self.updates, self.skipped = state['updates'], state['skipped']
        self.tensor.fill_(self.weight)
