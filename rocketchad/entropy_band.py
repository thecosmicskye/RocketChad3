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

    def update(self, entropy, *, sync=True):
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
        if sync and self.tensor.is_cuda:
            torch.cuda.synchronize(self.tensor.device)
        return self.weight

    def state_dict(self):
        return dict(weight=self.weight, ema=self.ema, updates=self.updates, skipped=self.skipped)

    def load_state_dict(self, state):
        if 'modes' in state:
            state = state['modes'].get(3) or state['modes'].get('3')
        self.weight, self.ema = state['weight'], state['ema']
        self.updates, self.skipped = state['updates'], state['skipped']
        self.tensor.fill_(self.weight)


class MultiModeEntropyBand:
    def __init__(self, config, device):
        self.bands = {mode: EntropyBand(config, device) for mode in (1, 2, 3)}
        self.tensor = torch.full((3,), float(config['entropy']), dtype=torch.float32, device=device)

    @property
    def weight(self):
        return {mode: band.weight for mode, band in self.bands.items()}

    @property
    def ema(self):
        return {mode: band.ema for mode, band in self.bands.items()}

    def _sync_tensor(self):
        self.tensor.copy_(self.tensor.new_tensor([self.bands[mode].weight for mode in (1, 2, 3)]))
        if self.tensor.is_cuda:
            torch.cuda.synchronize(self.tensor.device)

    def update(self, entropies):
        for mode, band in self.bands.items():
            value = entropies.get(mode)
            if value is not None:
                band.update(value, sync=False)
        self._sync_tensor()
        return self.weight

    def state_dict(self):
        return dict(modes={mode: band.state_dict() for mode, band in self.bands.items()})

    def load_state_dict(self, state):
        modes = state.get('modes')
        for mode, band in self.bands.items():
            item = state if modes is None else (modes.get(mode) or modes.get(str(mode)))
            if item is not None:
                band.load_state_dict(item)
        self._sync_tensor()


def make_entropy_band(config, device):
    if not config.get('entropy_band'):
        return None
    if config.get('team_size_fractions') and config['entropy_band'].get('per_mode', True):
        return MultiModeEntropyBand(config, device)
    return EntropyBand(config, device)
