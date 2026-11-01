import torch


class Observation:
    size = 168

    def __init__(self, env):
        device = env.device
        self.team_size = env.players // 2
        self.layout = env.team_layout
        self.player_slots = torch.tensor(list(range(self.team_size)) + list(range(3, 3 + self.team_size)), device=device)
        present = torch.zeros(6, dtype=torch.bool, device=device)
        present[self.player_slots] = True
        self.rotation = torch.tensor([[1, 1, 1]] * 3 + [[-1, -1, 1]] * 3, device=device, dtype=torch.float32)
        order = []
        for i in range(6):
            base = 0 if i < 3 else 3
            order.append([i, base + (i + 1) % 3, base + (i + 2) % 3] + list(range(3 - base, 6 - base)))
        self.order = torch.tensor(order, device=device)
        self.present = present[self.order] if self.layout is None else self.layout.ordered_presence(self.order)
        self.pad_order = None
        if env.config.get('team_pad_frame', False):
            half_turn = list(range(27, -1, -1)) + [29, 28, 33, 32, 31, 30]
            self.pad_order = torch.tensor([list(range(34))] * 3 + [half_turn] * 3, device=device)

    def pad(self, values):
        if self.layout is not None:
            return self.layout.to_logical(values)
        if self.team_size == 3:
            return values
        missing = values.new_zeros((values.shape[0], 3 - self.team_size, *values.shape[2:]))
        return torch.cat((values[:, :self.team_size], missing, values[:, self.team_size:], missing), dim=1)

    def __call__(self, env):
        n, c, b = env.arenas, env.car, env.ball
        pos, vel, forward, up, angular = (self.pad(c[name].reshape(n, env.players, 3))
                                          for name in ('pos', 'vel', 'basis_forward', 'basis_up', 'ang_vel'))
        scalars = torch.stack([c['boost'].float() / 100, c['is_on_ground'].float(),
                               (1 - c['has_double_jumped'].float()) * (1 - c['has_flipped'].float()),
                               c['is_demoed'].float()], -1).reshape(n, env.players, 4)
        scalars = self.pad(scalars)
        rot = self.rotation[None, :, None, :]
        cars = torch.cat(((pos[:, self.order] - pos[:, :, None]) * rot / 4096,
                          vel[:, self.order] * rot / 2300, forward[:, self.order] * rot,
                          up[:, self.order] * rot, angular[:, self.order] * rot / 5.5,
                          scalars[:, self.order]), -1)
        if self.layout is not None:
            cars = torch.where(self.present[..., None], cars, torch.zeros_like(cars))
        elif self.team_size != 3:
            cars = torch.where(self.present[None, :, :, None], cars, torch.zeros_like(cars))
        cars = cars.reshape(n, 6, 114)
        own = pos * self.rotation[None] / 4096
        ball_pos = (b['pos'][:, None] - pos) * self.rotation[None] / 4096
        ball_vel = b['vel'][:, None] * self.rotation[None] / 6000
        ball_ang = b['ang_vel'][:, None] * self.rotation[None] / 6
        pads = env.pads['is_active'].reshape(n, 34).float()[:, None].expand(-1, 6, -1)
        if self.pad_order is not None:
            pads = pads.gather(-1, self.pad_order[None].expand(n, -1, -1))
        actions = env.last_action if self.layout is not None else self.pad(env.last_action)
        observation = torch.cat((own, ball_pos, ball_vel, ball_ang, cars, actions, pads), -1)
        if self.team_size != 3:
            observation = observation.index_select(1, self.player_slots)
        return observation.reshape(n * env.players, self.size)
