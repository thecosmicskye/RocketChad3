import gc
from contextlib import contextmanager

import torch
import torch.distributed as dist

from .policy_terms import policy_terms


class Optimizers:
    def __init__(self, *, captured=(), eager=(), graph=True):
        self.captured, self.eager = tuple(captured), tuple(eager)
        self.all = self.captured + self.eager
        self.graph = graph
        self.parameters = list({id(p): p for optimizer in self.all for group in optimizer.param_groups
                                for p in group['params'] if p.requires_grad}.values())

    def zero_grad(self):
        for optimizer in self.all:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.all:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.all]

    def load_state_dict(self, state):
        for optimizer, item in zip(self.all, state):
            optimizer.load_state_dict(item)


class Reducer:
    def __init__(self, model, world):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.accumulators = [p.view_as(p).grad_fn.next_functions[0][0] for p in self.parameters]
        self.world = world
        lengths = [p.numel() for p in self.parameters]
        self.flat = torch.zeros(-(-sum(lengths) // 8) * 8, device=self.parameters[0].device)
        self.views = list(self.flat[:sum(lengths)].split(lengths))

    def __call__(self):
        grads = [p.grad.reshape(-1) for p in self.parameters]
        torch._foreach_copy_(self.views, grads)
        self.flat.div_(self.world)
        dist.all_reduce(self.flat)
        torch._foreach_copy_(grads, self.views)


def forward(model, obs, precision):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        logits, value = model(obs)
    return logits.float(), value.float()


def objective(logits, values, actions, old_log, advantages, returns, clip, entropy):
    taken, plogp = policy_terms(logits.float(), actions, entropy_grad=entropy != 0)
    logratio = taken - old_log
    ratio = logratio.exp()
    policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
    value = 0.5 * (values.float() - returns).square().mean()
    entropy_value = -plogp.mean()
    loss = policy + value - entropy * entropy_value
    stats = torch.stack((policy.detach(), value.detach(), entropy_value.detach(), (ratio - 1 - logratio).mean().detach()))
    return loss, stats


@contextmanager
def capture(graph, stream):
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, stream=stream):
            yield
    finally:
        if enabled:
            gc.enable()


class Learner:
    def __init__(self, model, optimizer, config, critic_parameters, world):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.critics = {id(p) for p in critic_parameters}
        optimized = {id(p) for p in optimizer.parameters}
        self.optimized = [p for p in model.parameters() if p.requires_grad and id(p) in optimized]
        self.unoptimized = [p for p in model.parameters() if p.requires_grad and id(p) not in optimized]
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.reduce = Reducer(model, world) if world > 1 else None
        torch.cuda.current_stream().wait_stream(self.stream)
        self.graph = None
        self.eager_steps = 0

    def _backward(self, inputs):
        obs, actions, old_log, advantages, returns = inputs
        logits, values = forward(self.model, obs, self.config['precision'])
        loss, stats = objective(logits, values, actions, old_log, advantages, returns,
                                self.config['clip'], self.config['entropy'])
        loss.backward()
        if self.reduce is not None:
            self.reduce()
        return stats

    def step(self, sources, indices, burnin=False):
        if indices.numel() != self.config['minibatch'] and self.graph is not None:
            self.graph.reset()
            self.graph = None
            self.eager_steps = 0
        if not self.optimizer.graph or burnin or self.eager_steps < 11 or indices.numel() != self.config['minibatch']:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                inputs = tuple(source[indices] for source in sources)
                self.optimizer.zero_grad()
                for p in self.unoptimized:
                    p.grad = None
                stats = self._backward(inputs)
                if burnin:
                    for p in self.model.parameters():
                        if id(p) not in self.critics:
                            p.grad = None
                torch.nn.utils.clip_grad_norm_(self.optimized, 0.5)
                self.optimizer.step()
            torch.cuda.current_stream().wait_stream(self.stream)
            stats.record_stream(torch.cuda.current_stream())
            if not burnin and indices.numel() == self.config['minibatch']:
                self.eager_steps += 1
            return stats
        if self.graph is None:
            inputs = tuple(source[indices].clone() for source in sources)
            tensors = list(self.model.parameters()) + list(self.model.buffers())
            for component in self.optimizer.captured:
                for state in component.state.values():
                    tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
            torch.cuda.synchronize()
            saved = [tensor.detach().clone() for tensor in tensors]
            self.optimizer.zero_grad()
            self.graph = torch.cuda.CUDAGraph()
            self.inputs = inputs
            with capture(self.graph, self.stream):
                self.optimizer.zero_grad()
                for p in self.unoptimized:
                    p.grad = None
                self.stats = self._backward(self.inputs)
                torch.nn.utils.clip_grad_norm_(self.optimized, 0.5)
                for optimizer in self.optimizer.captured:
                    optimizer.step()
            with torch.no_grad():
                for tensor, value in zip(tensors, saved):
                    tensor.copy_(value)
        for target, source in zip(self.inputs, sources):
            torch.index_select(source, 0, indices, out=target)
        self.graph.replay()
        for optimizer in self.optimizer.eager:
            optimizer.step()
        return self.stats

    def close(self):
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
