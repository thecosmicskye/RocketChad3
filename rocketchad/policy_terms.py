import math
import torch


def action_log_probs(logits, action_floor=0.0):
    logs = logits.log_softmax(-1)
    if not action_floor:
        return logs
    width = logits.shape[-1]
    if not 0.0 < action_floor * width < 1.0:
        raise ValueError(f'action_floor {action_floor} must satisfy 0 < floor * K < 1 (K={width})')
    return torch.logaddexp(logs + math.log1p(-width * action_floor), logs.new_full((), math.log(action_floor)))


class _PolicyTerms(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, actions, entropy_grad):
        logs = logits.log_softmax(-1)
        taken = logs.gather(-1, actions[:, None]).squeeze(-1)
        probabilities = logs.exp()
        plogp = (probabilities * logs).sum(-1)
        ctx.entropy_grad = entropy_grad
        if entropy_grad:
            ctx.save_for_backward(logs, probabilities, actions)
        else:
            ctx.save_for_backward(logs, actions)
            ctx.mark_non_differentiable(plogp)
        return (taken, plogp)

    @staticmethod
    def backward(ctx, grad_taken, grad_plogp):
        grad = None
        if ctx.entropy_grad:
            logs, probabilities, actions = ctx.saved_tensors
        else:
            (logs, actions), grad_plogp = (ctx.saved_tensors, None)
        if grad_plogp is not None:
            g = grad_plogp[:, None].expand_as(logs)
            grad = g * probabilities
            grad = grad + g * logs * probabilities
        if grad_taken is not None:
            if grad is None:
                grad = torch.zeros_like(logs)
            grad.scatter_add_(-1, actions[:, None], grad_taken[:, None])
        if grad is None:
            return (None, None, None)
        return (torch._log_softmax_backward_data(grad, logs, -1, logs.dtype), None, None)

def policy_terms(logits, actions, entropy_grad=True, action_floor=0.0):
    if action_floor or not (torch.is_grad_enabled() and logits.requires_grad) or logits.dim() != 2:
        logs = action_log_probs(logits, action_floor)
        detached = logs if entropy_grad else logs.detach()
        return (logs.gather(-1, actions[:, None]).squeeze(-1), (detached.exp() * detached).sum(-1))
    return _PolicyTerms.apply(logits, actions, entropy_grad)
