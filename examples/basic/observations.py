import torch


class Observation:
    size = 168

    def __init__(self, env):
        device = env.device
        self.rotation = torch.tensor([[1, 1, 1]] * 3 + [[-1, -1, 1]] * 3, device=device, dtype=torch.float32)
        order = []
        for i in range(6):
            base = 0 if i < 3 else 3
            order.append([i, base + (i + 1) % 3, base + (i + 2) % 3] + list(range(3 - base, 6 - base)))
        self.order = torch.tensor(order, device=device)

    def __call__(self, env):
        n, c, b = env.arenas, env.car, env.ball
        pos, vel, forward, up, angular = (c[name].reshape(n, 6, 3) for name in ('pos', 'vel', 'basis_forward', 'basis_up', 'ang_vel'))
        scalars = torch.stack([c['boost'].float() / 100, c['is_on_ground'].float(),
                               (1 - c['has_double_jumped'].float()) * (1 - c['has_flipped'].float()),
                               c['is_demoed'].float()], -1).reshape(n, 6, 4)
        rot = self.rotation[None, :, None, :]
        cars = torch.cat(((pos[:, self.order] - pos[:, :, None]) * rot / 4096,
                          vel[:, self.order] * rot / 2300, forward[:, self.order] * rot,
                          up[:, self.order] * rot, angular[:, self.order] * rot / 5.5,
                          scalars[:, self.order]), -1).reshape(n, 6, 114)
        own = pos * self.rotation[None] / 4096
        ball_pos = (b['pos'][:, None] - pos) * self.rotation[None] / 4096
        ball_vel = b['vel'][:, None] * self.rotation[None] / 6000
        ball_ang = b['ang_vel'][:, None] * self.rotation[None] / 6
        pads = env.pads['is_active'].reshape(n, 34).float()[:, None].expand(-1, 6, -1)
        return torch.cat((own, ball_pos, ball_vel, ball_ang, cars, env.last_action, pads), -1).reshape(n * 6, self.size)
