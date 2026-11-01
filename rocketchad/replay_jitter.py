import math
import numpy as np
import torch
from .replay_cache import CAR, PADS, PHYS, rot_matrix
JITTER_SEED_OFFSET = 29047
POSITION_UU, YAW_DEGREES, SPEED_FRACTION, BOOST = (50.0, 5.0, 0.05, 10.0)
CAR_RADIUS, BALL_RADIUS, BALL_FLOOR_BAND = (17.0, 92.75, 5.0)
CAR_SPEED_MAX, BALL_SPEED_MAX = (2300.0, 6000.0)
CAR_PAIR_MIN, CAR_BALL_MIN = (110.0, 135.0)
SIDE_X, BACK_Y, GOAL_BACK_Y, CEILING_Z = (4096.0, 5120.0, 6000.0, 2044.0)
GOAL_HALF_WIDTH, GOAL_HEIGHT = (892.755, 642.775)
CORNER_X, CORNER_Y, CORNER_R = (3072.0, 4096.0, 1024.0)
CAR0 = 3 + PHYS + PADS

def base_arrays(rows):
    rows = np.asarray(rows, dtype=np.float64)
    count = len(rows)
    pos, vel, ang, rot, boost, grounded = (np.zeros((count, 6, 3)), np.zeros((count, 6, 3)), np.zeros((count, 6, 3)), np.tile(np.eye(3), (count, 6, 1, 1)), np.zeros((count, 6)), np.zeros((count, 6), bool))
    present = np.zeros((count, 6), bool)
    for r, row in enumerate(rows):
        players = (len(row) - CAR0) // CAR
        cars = row[CAR0:CAR0 + players * CAR].reshape(players, CAR)
        order = [i for team in (0.0, 1.0) for i in range(players) if cars[i, 0] == team]
        for slot, i in enumerate(order):
            present[r, slot] = True
            c = cars[i]
            pos[r, slot], vel[r, slot], ang[r, slot] = (c[24:27], c[27:30], c[30:33])
            rot[r, slot] = rot_matrix(c[33:42]) if np.abs(c[24:42]).any() else np.eye(3)
            boost[r, slot] = np.clip(c[2], 0, 100)
            grounded[r, slot] = c[20:24].any()
    return dict(car_pos=pos, car_vel=vel, car_ang_vel=ang, car_rot=rot, car_boost=boost, car_grounded=grounded, car_present=present, ball_pos=rows[:, 3:6].copy(), ball_vel=rows[:, 6:9].copy())

def quat_wxyz(m):
    r = m.double()
    r00, r01, r02 = (r[..., 0, 0], r[..., 0, 1], r[..., 0, 2])
    r10, r11, r12 = (r[..., 1, 0], r[..., 1, 1], r[..., 1, 2])
    r20, r21, r22 = (r[..., 2, 0], r[..., 2, 1], r[..., 2, 2])
    t = r00 + r11 + r22
    s0 = torch.sqrt((t + 1).clamp(min=1e-12)) * 2
    s1 = torch.sqrt((1 + r00 - r11 - r22).clamp(min=1e-12)) * 2
    s2 = torch.sqrt((1 + r11 - r00 - r22).clamp(min=1e-12)) * 2
    s3 = torch.sqrt((1 + r22 - r00 - r11).clamp(min=1e-12)) * 2
    q0 = torch.stack((0.25 * s0, (r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0), -1)
    q1 = torch.stack(((r21 - r12) / s1, 0.25 * s1, (r01 + r10) / s1, (r02 + r20) / s1), -1)
    q2 = torch.stack(((r02 - r20) / s2, (r01 + r10) / s2, 0.25 * s2, (r12 + r21) / s2), -1)
    q3 = torch.stack(((r10 - r01) / s3, (r02 + r20) / s3, (r12 + r21) / s3, 0.25 * s3), -1)
    c0 = (t > 0)[..., None]
    c1 = ((r00 > r11) & (r00 > r22))[..., None]
    c2 = (r11 > r22)[..., None]
    q = torch.where(c0, q0, torch.where(c1, q1, torch.where(c2, q2, q3)))
    return q / q.norm(dim=-1, keepdim=True)

def axis_rotation(axis, theta):
    x, y, z = axis.unbind(-1)
    c, s = (torch.cos(theta), torch.sin(theta))
    C = 1 - c
    return torch.stack((torch.stack((c + x * x * C, x * y * C - z * s, x * z * C + y * s), -1), torch.stack((y * x * C + z * s, c + y * y * C, y * z * C - x * s), -1), torch.stack((z * x * C - y * s, z * y * C + x * s, c + z * z * C), -1)), -2)
RELATIVE_TOLERANCE = 1.0

def slacks(pos, radius):
    x, y = (pos[..., 0].abs(), pos[..., 1].abs())
    z = pos[..., 2]
    mouth = (x < GOAL_HALF_WIDTH) & (z < GOAL_HEIGHT)
    back = torch.where(mouth, torch.full_like(y, GOAL_BACK_Y), torch.full_like(y, BACK_Y))
    corner = (x > CORNER_X) & (y > CORNER_Y)
    arc = torch.where(corner, CORNER_R - radius - torch.hypot(x - CORNER_X, y - CORNER_Y), torch.full_like(x, math.inf))
    return torch.stack((z - radius, CEILING_Z - radius - z, SIDE_X - radius - x, back - radius - y, arc), -1)

def legal(pos, radius, reference=None):
    new = slacks(pos, radius)
    floor = torch.zeros_like(new) if reference is None else torch.clamp(slacks(reference, radius), max=0) - RELATIVE_TOLERANCE
    return (new >= floor).all(-1)

def clamp_speed(v, limit):
    norm = v.norm(dim=-1, keepdim=True)
    return v * torch.where(norm > limit, limit / norm.clamp(min=1e-09), torch.ones_like(norm))

class Jitter:
    FLOATS = 6 * 14 + 6

    def __init__(self, row_groups, seed, device):
        groups = [base_arrays(rows) for rows in row_groups]
        base = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
        self.base = {k: torch.as_tensor(v, device=device, dtype=torch.bool if v.dtype == bool else torch.float64) for k, v in base.items()}
        self.generator = torch.Generator(device=device).manual_seed(seed + JITTER_SEED_OFFSET)
        self.choice_generator = torch.Generator(device=device).manual_seed(seed + JITTER_SEED_OFFSET + 1)
        self.device = device
        self.pair_i, self.pair_j = torch.triu_indices(6, 6, 1, device=device)
        self.buffer = None

    def compile(self, n):
        self.overrides = torch.no_grad()(torch.compile(self.overrides, dynamic=False, options={"triton.cudagraphs": False}))
        state, live, self.buffer = (self.generator.get_state(), self.buffer, None)
        rows, selected = (torch.zeros(n, dtype=torch.int64, device=self.device), torch.zeros(n, dtype=torch.bool, device=self.device))
        for _ in range(2):
            self.overrides(rows, selected)
        self.generator.set_state(state)
        self.buffer = live

    def choose(self, eligible):
        return eligible & (torch.rand(eligible.shape, generator=self.choice_generator, device=self.device) < 0.5)

    def overrides(self, rows, selected):
        n = rows.shape[0]
        draws = torch.rand(n * 40, generator=self.generator, device=self.device, dtype=torch.float64).mul_(2).sub_(1)

        def uniform(start, end, shape, bound):
            return draws[n * start:n * end].view(shape) * bound

        b = {k: v[rows.clamp(min=0)] for k, v in self.base.items()}
        pos, vel, ang, rot = (b['car_pos'], b['car_vel'], b['car_ang_vel'], b['car_rot'])
        grounded, present = b['car_grounded'], b['car_present']
        up = rot[..., :, 2]
        d = uniform(0, 18, (n, 6, 3), POSITION_UU)
        d = torch.where(grounded[..., None], d - (d * up).sum(-1, keepdim=True) * up, d)
        z = torch.zeros_like(up)
        z[..., 2] = 1
        axis = torch.where(grounded[..., None], up, z)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-09)
        R = axis_rotation(axis, uniform(18, 24, (n, 6), math.radians(YAW_DEGREES)))
        new_rot = torch.where(present[..., None, None], R @ rot, rot)
        new_vel = clamp_speed((R @ vel[..., None])[..., 0] * (1 + uniform(24, 30, (n, 6, 1), SPEED_FRACTION)), CAR_SPEED_MAX)
        new_vel = torch.where(present[..., None], new_vel, vel)
        new_ang = torch.where(present[..., None], (R @ ang[..., None])[..., 0], ang)
        boost = torch.where(present, (b['car_boost'] + uniform(30, 36, (n, 6), BOOST)).clamp(0, 100), b['car_boost'])
        ball_pos, ball_vel = (b['ball_pos'], b['ball_vel'])
        db = uniform(36, 39, (n, 3), POSITION_UU)
        db[:, 2] = torch.where(ball_pos[:, 2] < BALL_RADIUS + BALL_FLOOR_BAND, torch.zeros_like(db[:, 2]), db[:, 2])
        new_ball_vel = clamp_speed(ball_vel * (1 + uniform(39, 40, (n, 1), SPEED_FRACTION)), BALL_SPEED_MAX)
        car_ok = legal(pos + d, CAR_RADIUS, pos)
        ball_ok = legal(ball_pos + db, BALL_RADIUS, ball_pos)
        d = torch.where(car_ok[..., None], d, torch.zeros_like(d))
        db = torch.where(ball_ok[:, None], db, torch.zeros_like(db))
        d = torch.where(present[..., None], d, 0)
        new_pos, new_ball = (pos + d, ball_pos + db)

        def close(p, q, limit):
            return (p - q).norm(dim=-1) < limit
        i, j = self.pair_i, self.pair_j
        crowded = (close(new_pos[:, i], new_pos[:, j], CAR_PAIR_MIN) & ~close(pos[:, i], pos[:, j], CAR_PAIR_MIN) & present[:, i] & present[:, j]).any(1)
        crowded |= (close(new_pos, new_ball[:, None], CAR_BALL_MIN) & ~close(pos, ball_pos[:, None], CAR_BALL_MIN) & present).any(1)
        new_pos = torch.where(crowded[:, None, None], pos, new_pos)
        new_ball = torch.where(crowded[:, None], ball_pos, new_ball)
        cars = torch.cat((new_pos, new_vel, new_ang, quat_wxyz(new_rot), boost[..., None]), -1)
        out = torch.cat((cars.reshape(n, 6 * 14), new_ball, new_ball_vel), -1).float().contiguous()
        if self.buffer is None:
            self.buffer = out
        else:
            self.buffer.copy_(out)
        return self.buffer
