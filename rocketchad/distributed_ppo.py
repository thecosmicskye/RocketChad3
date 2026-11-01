import torch
import torch.distributed as dist
from .policy_terms import policy_terms

def normalize_advantages(advantages, metrics=None):
    values = advantages.double()
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

def objective(logits, values, actions, old_log, advantages, returns, clip, entropy_weight):
    no_entropy_grad = isinstance(entropy_weight, (int, float)) and entropy_weight == 0
    taken, plogp = policy_terms(logits.float(), actions, entropy_grad=not no_entropy_grad)
    logratio = taken - old_log
    ratio = logratio.exp()
    policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
    value = 0.5 * (values.float() - returns).square().mean()
    entropy = -plogp.mean()
    loss = policy + value - entropy_weight * entropy
    stats = torch.stack((policy.detach(), value.detach(), entropy.detach(), (ratio - 1 - logratio).mean().detach()))
    return (loss, stats)
