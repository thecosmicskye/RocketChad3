import torch
import torch.distributed as dist
from .policy_terms import policy_terms

def normalize_advantages(advantages, metrics=None, mode_columns=None):
    values = advantages.double()
    if mode_columns is not None:
        rows = []
        for cols in mode_columns:
            part = values[:, cols]
            rows.append(torch.stack((part.sum(), part.square().sum(),
                                     torch.tensor(part.numel(), device=values.device, dtype=torch.float64))))
        moments = torch.stack(rows)
        if dist.is_initialized():
            dist.all_reduce(moments)
        out = torch.empty_like(advantages)
        for cols, row in zip(mode_columns, moments):
            if not cols.numel():
                continue
            total, squares, count = row.unbind()
            mean = total / count.clamp_min(1)
            variance = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0)
            std = variance.sqrt()
            out[:, cols] = ((values[:, cols] - mean) / std.clamp_min(1e-08)).to(advantages.dtype)
        if metrics is not None:
            total, squares, count = moments.sum(0).unbind()
            mean = total / count.clamp_min(1)
            std = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0).sqrt()
            metrics.update(raw_advantage_mean=mean, raw_advantage_std=std)
        return out
    moments = torch.stack((values.sum(), values.square().sum(), torch.tensor(values.numel(), device=values.device, dtype=torch.float64)))
    if dist.is_initialized():
        dist.all_reduce(moments)
    total, squares, count = moments.unbind()
    mean = total / count
    variance = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0)
    std = variance.sqrt()
    if metrics is not None:
        metrics.update(raw_advantage_mean=mean, raw_advantage_std=std)
    return ((values - mean) / std.clamp_min(1e-08)).to(advantages.dtype)

def objective(logits, values, actions, old_log, advantages, returns, clip, entropy_weight, action_floor=0.0, modes=None):
    no_entropy_grad = isinstance(entropy_weight, (int, float)) and entropy_weight == 0
    taken, plogp = policy_terms(logits.float(), actions, entropy_grad=not no_entropy_grad, action_floor=action_floor)
    logratio = taken - old_log
    ratio = logratio.exp()
    policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
    value = 0.5 * (values.float() - returns).square().mean()
    entropy = -plogp.mean()
    base = torch.stack((policy.detach(), value.detach(), entropy.detach(), (ratio - 1 - logratio).mean().detach()))
    if modes is None:
        return (policy + value - entropy_weight * entropy, base)
    loss = policy + value + (entropy_weight[modes] * plogp).mean()
    detached = -plogp.detach()
    masks = (modes[:, None] == torch.arange(3, device=modes.device)).to(detached.dtype)
    return (loss, torch.cat((base, (detached[:, None] * masks).mean(0), masks.mean(0))))
