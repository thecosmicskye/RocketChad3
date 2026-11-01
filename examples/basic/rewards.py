import torch


def reward(env, blue_goals, orange_goals, phase):
    if phase == 'velocity':
        goals = env.ball['pos'].new_tensor([[0.0, 6000.0, 0.0], [0.0, -6000.0, 0.0]])
        delta = goals[None] - env.ball['pos'][:, None, :]
        direction = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        score = (env.ball['vel'][:, None, :] * direction).sum(-1) / 6000
        return score.repeat_interleave(env.players // 2, dim=1).reshape(-1)
    score = blue_goals.float() - orange_goals.float()
    return torch.stack((score, -score), 1).repeat_interleave(env.players // 2, dim=1).reshape(-1)
