import torch

from .graphed_ppo import PPOMinibatchGraph, accumulate
from .manual_reduce import ManualReducer
from .precision import training_forward


class Optimizers:
    def __init__(self, *, captured=(), eager=(), graph=True):
        self.captured, self.eager = tuple(captured), tuple(eager)
        self.all = self.captured + self.eager
        self.graph = graph
        self.parameters = list({id(p): p for optimizer in self.all for group in optimizer.param_groups
                                for p in group['params'] if p.requires_grad}.values())
        self.parameter_ids = {id(p) for p in self.parameters}

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


class Learner:
    def __init__(self, model, optimizer, config, critic_parameters, world, shared=None, stream=None):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.shared = model if shared is None else shared
        self.forward = training_forward(self.shared, config['precision'])
        self.critics = {id(p) for p in critic_parameters}
        optimized = {id(p) for p in optimizer.parameters}
        self.optimized = [p for p in model.parameters() if p.requires_grad and id(p) in optimized]
        self.unoptimized = [p for p in model.parameters() if p.requires_grad and id(p) not in optimized]
        self.stream = torch.cuda.Stream() if stream is None else stream
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.reduce = ManualReducer(self.optimized) if world > 1 and config.get('grad_reduce', 'flat_fp32') != 'ddp' else None
        torch.cuda.current_stream().wait_stream(self.stream)
        self.graph = None
        self.eager_steps = 0

    def _backward(self, inputs):
        stats = accumulate(self.forward, self.config, *inputs)
        if self.reduce is not None:
            self.reduce()
        return stats

    def step(self, sources, indices, burnin=False, obs=None):
        if indices.numel() != self.config['minibatch'] and self.graph is not None:
            self.graph.graph.reset()
            self.graph = None
            self.eager_steps = 0
        if obs is None and sources[0].dtype == torch.bfloat16:
            obs = sources[0].index_select(0, indices).float()
        if not self.optimizer.graph or burnin or self.eager_steps < 11 or indices.numel() != self.config['minibatch']:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                inputs = (sources[0][indices] if obs is None else obs.float(),
                          *(source[indices] for source in sources[1:]))
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
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                inputs = (sources[0][indices] if obs is None else obs.float(),
                          *(source[indices] for source in sources[1:]))
                self.graph = PPOMinibatchGraph(self.shared, self.model, self.optimizer, self.config,
                                              inputs, self.stream, reduce_gradients=self.reduce)
            torch.cuda.current_stream().wait_stream(self.stream)
        return self.graph.step_indexed(sources, indices, None if obs is None else obs.float())

    def close(self):
        if self.graph is not None:
            self.graph.graph.reset()
            self.graph = None
