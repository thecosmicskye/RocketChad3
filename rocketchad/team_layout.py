import math
import torch

def realized_count(arenas, fraction, quantum=16):
    count = int(arenas * fraction) // quantum * quantum
    if fraction and (not count):
        raise ValueError
    return count

def rollout_quantum(owned_rows, minibatch):
    return math.lcm(4, minibatch // math.gcd(owned_rows, minibatch))

def mixed_counts(arenas, fractions):
    one = float(fractions.get(1, 0))
    two = float(fractions.get(2, 0))
    if math.isclose(one, 1 / 3, abs_tol=1e-8) and math.isclose(two, 1 / 3, abs_tol=1e-8):
        count = 2 * round(arenas / 6)
        counts = {1: count, 2: arenas - 2 * count, 3: count}
    else:
        counts = {1: 2 * round(arenas * one / 2), 2: 2 * round(arenas * two / 2)}
        counts[3] = arenas - sum(counts.values())
    if arenas % 2 or any(count < 0 or count % 2 for count in counts.values()) or (one > 0 and counts[1] == 0) or (two > 0 and counts[2] == 0):
        raise ValueError('mixed modes require paired nonempty requested partitions')
    return counts

def mode_sizes(eligible, fractions, seed):
    counts = {mode: realized_count(len(eligible), fractions.get(mode, 0)) for mode in (1, 2)}
    available = eligible.nonzero().flatten()
    if sum(counts.values()) > len(available):
        raise ValueError
    sizes = torch.full_like(eligible, 3, dtype=torch.int64)
    if not sum(counts.values()):
        return sizes
    generator = torch.Generator(device=eligible.device)
    generator.manual_seed((int(seed) + 6072331472944585285) % (2 ** 63 - 1))
    selected = available[torch.randperm(len(available), device=eligible.device, generator=generator)]
    offset = 0
    for mode in (1, 2):
        sizes[selected[offset:offset + counts[mode]]] = mode
        offset += counts[mode]
    return sizes

class TeamLayout:

    def __init__(self, sizes):
        self.sizes = sizes.clone()
        self.arenas = len(sizes)
        slot = torch.arange(6, device=sizes.device)[None, :]
        team_slot = slot.remainder(3)
        orange = slot >= 3
        self.present = team_slot < sizes[:, None]
        self.absent = ~self.present
        self.logical_to_physical = torch.where(orange, sizes[:, None] + team_slot, team_slot)
        self.logical_to_physical = self.logical_to_physical.clamp_max(5)
        self.physical_present = slot < 2 * sizes[:, None]
        self.physical_absent = ~self.physical_present
        self.physical_to_logical = torch.where(slot < sizes[:, None], slot, 3 + slot - sizes[:, None])
        self.physical_to_logical = self.physical_to_logical.clamp(0, 5)
        self.environment_index = torch.arange(self.arenas, device=sizes.device)[:, None]

    def _transform(self, values, indices, absent, arenas=None):
        if arenas is None:
            selected = values[self.environment_index, indices]
            mask = absent
        else:
            idx = indices.index_select(0, arenas).reshape(len(arenas), 6, *([1] * (values.ndim - 2)))
            selected = values.gather(1, idx.expand(len(arenas), 6, *values.shape[2:]))
            mask = absent.index_select(0, arenas)
        return selected.masked_fill_(mask.reshape(len(selected), 6, *[1] * (values.ndim - 2)), 0)

    def to_logical(self, physical_values, arenas=None):
        return self._transform(physical_values, self.logical_to_physical, self.absent, arenas)

    def to_physical(self, logical_values):
        return self._transform(logical_values, self.physical_to_logical, self.physical_absent)

    def physical_config_values(self, logical_values):
        return logical_values[self.environment_index, self.physical_to_logical]

    def physical_config_specs(self, logical_specs):
        indices = self.physical_to_logical.cpu().tolist()
        return [logical_specs[arena * 6 + slot] for arena, row in enumerate(indices) for slot in row]

    def learner_indices(self, owned):
        return (owned & self.present).flatten().nonzero().flatten()

    def ordered_presence(self, order):
        return self.present[:, order]

class LogicalCarFields:

    def __init__(self, env):
        self.env = env

    def __getitem__(self, name):
        return self.env.logical_car_field(name)
