from collections import defaultdict
from contextlib import nullcontext
import math
import torch

def orthogonalize_batch(matrices, coefficients, steps, eps, dtype):
    a, b, c = coefficients
    x = matrices.to(dtype=dtype)
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(-2, -1))
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    return x.transpose(-2, -1) if transpose else x

class BatchedMuon(torch.optim.Muon):

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            for p, grad in zip(params, grads):
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
            bufs = [self.state[p]['momentum_buffer'] for p in params]
            torch._foreach_lerp_(bufs, grads, 1 - group['momentum'])
            updates = torch._foreach_lerp(grads, bufs, group['momentum']) if group['nesterov'] else bufs
            if group['weight_decay']:
                torch._foreach_mul_(params, 1 - lr * group['weight_decay'])
            batches = defaultdict(list)
            for p, update in zip(params, updates):
                update = update.mT if update.shape[0] > update.shape[1] else update
                batches[tuple(update.shape), p.device].append((p, update))
            streams = self.__dict__.setdefault('streams', {})
            ambient = torch.cuda.current_stream() if params and params[0].is_cuda else None
            for key, entries in batches.items():
                shape, device = key
                if ambient is not None and key not in streams:
                    streams[key] = torch.cuda.Stream(device)
                stream = None if ambient is None else streams[key]
                if stream is not None:
                    stream.wait_stream(ambient)
                with nullcontext() if stream is None else torch.cuda.stream(stream):
                    updates = orthogonalize_batch(torch.stack([g for _, g in entries]), group['ns_coefficients'], group['ns_steps'], group['eps'], self.compute_dtype)
                    scale = 0.2 * math.sqrt(max(shape))
                    for (p, _), update in zip(entries, updates.unbind()):
                        p.add_(update if update.shape == p.shape else update.mT, alpha=-lr * scale)
            if ambient is not None:
                for key in batches:
                    ambient.wait_stream(streams[key])
