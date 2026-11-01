import itertools
import torch

def action_table(device):
    actions = []
    for throttle, steer, boost, handbrake in itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)):
        if boost and throttle != 1:
            continue
        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch, yaw, roll, jump, boost in itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)):
        if jump and yaw != 0:
            continue
        if pitch == roll == jump == 0:
            continue
        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, int(jump and (pitch != 0 or yaw != 0 or roll != 0))])
    return torch.tensor(actions, device=device, dtype=torch.float32)
