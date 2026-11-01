import json
import math
from pathlib import Path
import struct
import numpy as np
PHYS = 18
PADS = 34
CAR = 42
HEADER = struct.Struct('<8sIIQQ')
CACHE_PAD_XY = ((0, -4240), (-1792, -4184), (1792, -4184), (-3072, -4096), (3072, -4096), (-940, -3308), (940, -3308), (0, -2816), (-3584, -2484), (3584, -2484), (-1788, -2300), (1788, -2300), (-2048, -1036), (0, -1024), (2048, -1036), (-3584, 0), (-1024, 0), (1024, 0), (3584, 0), (-2048, 1036), (0, 1024), (2048, 1036), (-1788, 2300), (1788, 2300), (-3584, 2484), (3584, 2484), (0, 2816), (-940, 3310), (940, 3308), (-3072, 4096), (3072, 4096), (-1792, 4184), (1792, 4184), (0, 4240))
ROCKETGPU_PAD_XY = ((0, -4240), (-1792, -4184), (1792, -4184), (-940, -3308), (940, -3308), (0, -2816), (-3584, -2484), (3584, -2484), (-1788, -2300), (1788, -2300), (-2048, -1036), (0, -1024), (2048, -1036), (-1024, 0), (1024, 0), (-2048, 1036), (0, 1024), (2048, 1036), (-1788, 2300), (1788, 2300), (-3584, 2484), (3584, 2484), (0, 2816), (-940, 3308), (940, 3308), (-1792, 4184), (1792, 4184), (0, 4240), (-3584, 0), (3584, 0), (-3072, 4096), (3072, 4096), (-3072, -4096), (3072, -4096))
PAD_MATCH_DIST2 = 10.0
CACHE_FROM_ROCKETGPU = np.array([next((i for i, (cx, cy) in enumerate(CACHE_PAD_XY) if (cx - rx) ** 2 + (cy - ry) ** 2 < PAD_MATCH_DIST2)) for rx, ry in ROCKETGPU_PAD_XY], dtype=np.int64)

def read_cache(path):
    with open(path, 'rb') as f:
        magic, version, players, count, floats = HEADER.unpack(f.read(HEADER.size))
    return (dict(version=version, players=players, count=count, floats=floats), np.memmap(path, dtype='<f4', mode='r', offset=HEADER.size, shape=(count, floats)))

def rot_matrix(m9):
    m = np.asarray(m9, dtype=np.float64)
    return np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]])

def quat_wxyz(r):
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return np.array(q)

def decode(state):
    s = np.asarray(state, dtype=np.float64)
    c = 3

    def phys(v):
        return dict(pos=v[0:3], vel=v[3:6], ang_vel=v[6:9], rot=rot_matrix(v[9:18]))
    ball = phys(s[c:c + PHYS])
    c += PHYS
    pads = s[c:c + PADS]
    c += PADS
    cars = []
    for i in range((len(s) - c) // CAR):
        v = s[c + i * CAR:c + (i + 1) * CAR]
        cars.append(dict(team=v[0], hitbox=v[1], boost=v[2], boost_active_time=v[3], handbrake=v[4], supersonic_time=v[5], demo_respawn_timer=v[6], has_jumped=v[7], is_jumping=v[8], jump_time=v[9], has_flipped=v[10], has_double_jumped=v[11], air_time_since_jump=v[12], flip_time=v[13], flip_torque=v[14:17], is_autoflipping=v[17], autoflip_timer=v[18], autoflip_direction=v[19], wheels=v[20:24], physics=v[24:42], **phys(v[24:42])))
    return dict(tick=s[0], blue_score=s[1], orange_score=s[2], ball=ball, pads=pads, cars=cars)


def choose_ranges(sources, rows, rng, target_kickoff=None):
    lengths = np.array([s['end_row'] - s['first_row'] for s in sources])
    mass = lengths * np.array([s['sampling_weight'] for s in sources], dtype=float)
    if target_kickoff is not None:
        kickoff = np.array([s['row_kind'] == 'kickoff' for s in sources])
        for mask, fraction in ((kickoff, target_kickoff), (~kickoff, 1 - target_kickoff)):
            total = mass[mask].sum()
            if total:
                mass[mask] *= fraction / total
    count = min(rows, int(lengths.sum()))
    quotas = np.minimum((count * mass / mass.sum()).astype(int), lengths)
    while quotas.sum() < count:
        available = lengths > quotas
        probability = mass * available
        if not probability.sum():
            probability = available.astype(float)
        quotas[rng.choice(len(sources), p=probability / probability.sum())] += 1
    chosen, eligible = [], []
    for source, quota, length in zip(sources, quotas, lengths):
        chosen.append(source['first_row'] + rng.choice(int(length), int(quota), replace=False))
        eligible.append(np.full(int(quota), source['row_kind'] == 'open_play' and source['jitter_eligible']))
    chosen, eligible = np.concatenate(chosen), np.concatenate(eligible)
    order = rng.permutation(count)
    return chosen[order], eligible[order]



def sample_cache(path, count, rng, *, return_eligibility=False):
    _, data = read_cache(path)
    sidecar = Path(str(path) + '.receipt.json')
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if metadata.get('kind') == 'merged_replay_start_cache_v2':
        indices, eligible = choose_ranges(metadata['sources'], count, rng,
                                          metadata.get('sampling_mix', {}).get('target_1hz_kickoff_fraction'))
    else:
        indices = rng.choice(len(data), min(count, len(data)), replace=False)
        eligible = np.zeros(len(indices), dtype=bool)
    return (data[indices], eligible) if return_eligibility else data[indices]

def public_states(rows):
    import rocketgpu_py as rs
    cars, balls, configs, pads = ([], [], [], [])
    config_factories = [getattr(rs, name + 'Config') for name in ('Octane', 'Dominus', 'Plank', 'Breakout', 'Hybrid', 'Merc')]

    def vec(v):
        return rs.Vec3(*(float(x) for x in v))

    def physics(target, source):
        target.pos, target.vel, target.ang_vel = [vec(source[k]) for k in ('pos', 'vel', 'ang_vel')]
        target.rot = rs.Quat(*(float(x) for x in quat_wxyz(source['rot'])))
    for row in rows:
        d = decode(row)
        ball = rs.BallState()
        physics(ball, d['ball'])
        balls.append(ball)
        players, bodies = ([], [])
        ordered = [c for team in (0, 1) for c in d['cars'] if c['team'] == team]
        for item in ordered:
            car = rs.CarState()
            if item['demo_respawn_timer'] > 0 and (not np.abs(item['physics']).any()):
                item = dict(item, rot=np.eye(3))
            physics(car, item)
            car.boost = float(np.clip(item['boost'], 0, 100))
            for source, target in (('boost_active_time', 'boosting_time'), ('handbrake', 'handbrake_val'), ('supersonic_time', 'supersonic_time'), ('demo_respawn_timer', 'demo_respawn_timer'), ('jump_time', 'jump_time'), ('air_time_since_jump', 'air_time_since_jump'), ('flip_time', 'flip_time'), ('autoflip_timer', 'auto_flip_timer'), ('autoflip_direction', 'auto_flip_torque_scale')):
                setattr(car, target, float(item[source]))
            for source, target in (('has_jumped', 'has_jumped'), ('is_jumping', 'is_jumping'), ('has_flipped', 'has_flipped'), ('has_double_jumped', 'has_double_jumped'), ('is_autoflipping', 'is_auto_flipping')):
                setattr(car, target, bool(item[source]))
            car.flip_rel_torque = vec(item['flip_torque'])
            car.is_flipping = bool(item['has_flipped'] and item['flip_time'] <= 0.65)
            car.is_demoed = item['demo_respawn_timer'] > 0
            car.is_boosting = item['boost_active_time'] > 0
            car.is_supersonic = item['supersonic_time'] > 0
            car.wheels_with_contact = [int(x) for x in item['wheels']]
            car.is_on_ground = sum(car.wheels_with_contact) >= 3
            car.air_time = 0.0 if car.is_on_ground else max(float(item['air_time_since_jump']), 1 / 120)
            players.append(car)
            bodies.append(config_factories[int(item['hitbox'])]())
        cars.append(players)
        configs.append(bodies)
        arena_pads = []
        for destination, source in enumerate(CACHE_FROM_ROCKETGPU):
            pad = rs.BoostPadState()
            pad.is_big = destination >= 28
            pad.cooldown = max(0.0, min(10.0 if pad.is_big else 4.0, float(d['pads'][source])))
            pad.is_active = pad.cooldown <= 0.0
            arena_pads.append(pad)
        pads.append(arena_pads)
    return (cars, balls, configs, pads)

