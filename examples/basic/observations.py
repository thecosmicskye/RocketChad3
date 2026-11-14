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
        if env.config.get('team_pad_frame', True):
            half_turn = list(range(27, -1, -1)) + [29, 28, 33, 32, 31, 30]
            self.pad_order = torch.tensor([list(range(34))] * 3 + [half_turn] * 3, device=device)

    def pad(self, values, arenas=None):
        if self.layout is not None:
            return self.layout.to_logical(values, arenas)
        if self.team_size == 3:
            return values
        missing = values.new_zeros((values.shape[0], 3 - self.team_size, *values.shape[2:]))
        return torch.cat((values[:, :self.team_size], missing, values[:, self.team_size:], missing), dim=1)

    def __call__(self, env, arenas=None):
        total, p, c, b = env.arenas, env.players, env.car, env.ball
        if arenas is None:
            n = total
            pos, vel, forward, up, angular = (self.pad(c[name].reshape(total, p, 3))
                                              for name in ('pos', 'vel', 'basis_forward', 'basis_up', 'ang_vel'))
            on_ground = c['is_on_ground']
            has_flip = (on_ground != 0) | ((c['has_flipped'] == 0) & (c['has_double_jumped'] == 0) & (c['air_time_since_jump'] < 1.25))
            scalars = torch.stack([c['boost'].float() / 100, on_ground.float(),
                                   has_flip.float(), c['is_demoed'].float()], -1).reshape(total, p, 4)
            scalars = self.pad(scalars)
            present = self.present
            b_pos, b_vel, b_ang = b['pos'], b['vel'], b['ang_vel']
            raw_pads = env.pads['is_active'].reshape(total, 34)
            last_action = env.last_action
        else:
            n = arenas.numel()
            if not n:
                return env.last_action.new_empty((0, self.size))
            vectors = torch.stack([c[name].reshape(total, p, 3).index_select(0, arenas)
                                   for name in ('pos', 'vel', 'basis_forward', 'basis_up', 'ang_vel')], 2)
            pos, vel, forward, up, angular = self.pad(vectors, arenas).unbind(2)
            sub = lambda name: c[name].reshape(total, p).index_select(0, arenas)
            on_ground = sub('is_on_ground')
            has_flip = (on_ground != 0) | ((sub('has_flipped') == 0) & (sub('has_double_jumped') == 0) & (sub('air_time_since_jump') < 1.25))
            scalars = torch.stack([sub('boost').float() / 100, on_ground.float(),
                                   has_flip.float(), sub('is_demoed').float()], -1)
            scalars = self.pad(scalars, arenas)
            present = self.present if self.layout is None else self.present.index_select(0, arenas)
            b_pos, b_vel, b_ang = (b[name].index_select(0, arenas) for name in ('pos', 'vel', 'ang_vel'))
            raw_pads = env.pads['is_active'].reshape(total, 34).index_select(0, arenas)
            last_action = env.last_action.index_select(0, arenas)
        rot = self.rotation[None, :, None, :]
        cars = torch.cat(((pos[:, self.order] - pos[:, :, None]) * rot / 4096,
                          vel[:, self.order] * rot / 2300, forward[:, self.order] * rot,
                          up[:, self.order] * rot, angular[:, self.order] * rot / 5.5,
                          scalars[:, self.order]), -1)
        if self.layout is not None:
            cars = torch.where(present[..., None], cars, torch.zeros_like(cars))
        elif self.team_size != 3:
            cars = torch.where(present[None, :, :, None], cars, torch.zeros_like(cars))
        cars = cars.reshape(n, 6, 114)
        own = pos * self.rotation[None] / 4096
        ball_pos = (b_pos[:, None] - pos) * self.rotation[None] / 4096
        ball_vel = b_vel[:, None] * self.rotation[None] / 6000
        ball_ang = b_ang[:, None] * self.rotation[None] / 6
        pads = raw_pads.float()[:, None].expand(-1, 6, -1)
        if self.pad_order is not None:
            pads = pads.gather(-1, self.pad_order[None].expand(n, -1, -1))
        actions = last_action if self.layout is not None else self.pad(last_action, arenas)
        observation = torch.cat((own, ball_pos, ball_vel, ball_ang, cars, actions, pads), -1)
        if self.team_size != 3:
            observation = observation.index_select(1, self.player_slots)
        return observation.reshape(n * p, self.size)
