import torch
import torch.distributed as dist

class ManualReducer:

    def __init__(self, parameters):
        self.params = list(parameters)
        # Hold each parameter's AccumulateGrad node on the capture stream so CUDA graph capture sees no stale stream-0 reference.
        self.accumulators = [p.view_as(p).grad_fn.next_functions[0][0] for p in self.params]
        self.world = dist.get_world_size()
        lengths = [p.numel() for p in self.params]
        self.lengths = lengths
        self.numel = sum(lengths)
        self.flat = torch.zeros(-(-self.numel // 8) * 8, device=self.params[0].device)
        self.views = list(self.flat[:self.numel].split(self.lengths))

    def __call__(self):
        grads = [p.grad.reshape(-1)[:n] for p, n in zip(self.params, self.lengths)]
        torch._foreach_copy_(self.views, grads)
        self.flat.div_(self.world)
        dist.all_reduce(self.flat)
        torch._foreach_copy_(grads, self.views)

    def broadcast_parameters(self):
        with torch.no_grad():
            for p in self.params:
                dist.broadcast(p, 0)
