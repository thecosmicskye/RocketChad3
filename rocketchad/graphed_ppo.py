import torch
from .distributed_ppo import objective
from .precision import training_forward
from .rollout_graph import capture
compiled_objective = torch.compile(objective, fullgraph=True, dynamic=False)

def state_tensors(model, optimizer):
    values = list(model.parameters()) + list(model.buffers())
    for component in optimizer.captured:
        for state in component.state.values():
            values.extend((v for v in state.values() if isinstance(v, torch.Tensor)))
    return values

def chunks(rows, microbatch):
    if not microbatch or microbatch >= rows:
        return [(0, rows)]
    return [(b, min(b + microbatch, rows)) for b in range(0, rows, microbatch)]

def accumulate(forward, cfg, obs, actions, old_log, advantages, returns):
    objective_fn = compiled_objective if cfg.get('graphed_compile', False) else objective
    parts = chunks(obs.shape[0], cfg.get('microbatch', 0))
    whole = len(parts) == 1
    stats = None
    for b, e in parts:
        o, a, l, v, r = (obs, actions, old_log, advantages, returns) if whole else (obs[b:e], actions[b:e], old_log[b:e], advantages[b:e], returns[b:e])
        logits, values = forward(o)
        loss, part = objective_fn(logits, values, a, l, v, r, cfg['clip'], cfg.get('_entropy_tensor', cfg['entropy']))
        if whole:
            loss.backward()
            return part
        w = (e - b) / obs.shape[0]
        (loss * w).backward()
        stats = part * w if stats is None else stats + part * w
    return stats

class PPOMinibatchGraph:

    def __init__(self, shared, model, optimizer, cfg, inputs, stream, reduce_gradients=None):
        self.reduce_gradients = reduce_gradients
        self.shared = shared
        self.model = model
        self.optimizer = optimizer
        self.optimized = [p for p in model.parameters() if p.requires_grad and id(p) in optimizer.parameter_ids]
        self.cfg = cfg
        self.inputs = tuple((x.clone() for x in inputs))
        self.forward = training_forward(shared, cfg['precision'])
        self.graph = torch.cuda.CUDAGraph()
        self.stream = stream
        owned = state_tensors(model, optimizer)
        saved = [v.detach().clone() for v in owned]
        optimizer.zero_grad()
        with capture(self.graph, stream=stream):
            self.stats = self.eager_step(self.inputs)
        with torch.no_grad():
            for tensor, value in zip(owned, saved):
                tensor.copy_(value)
        self.storage = [x.data_ptr() for x in state_tensors(model, optimizer)]

    def eager_step(self, inputs):
        obs, actions, old_log, advantages, returns = inputs
        self.optimizer.zero_grad()
        for parameter in self.model.parameters():
            if parameter.requires_grad and id(parameter) not in self.optimizer.parameter_ids:
                parameter.grad = None
        stats = accumulate(self.forward, self.cfg, obs, actions, old_log, advantages, returns)
        if self.reduce_gradients is not None:
            self.reduce_gradients()
        self.preclip_norm = torch.nn.utils.clip_grad_norm_(self.optimized, 0.5)
        for component in self.optimizer.captured:
            component.step()
        return stats

    def step_indexed(self, sources, idx, obs=None, timing=None):
        first = int(obs is not None)
        if first:
            self.inputs[0].copy_(obs)
        for target, source in zip(self.inputs[first:], sources[first:]):
            torch.index_select(source, 0, idx, out=target)
        if timing is not None:
            timing[1].record()
        self.graph.replay()
        if timing is not None:
            timing[2].record()
        for component in self.optimizer.eager:
            component.step()
        if timing is not None:
            timing[3].record()
        return self.stats

    def check_storage(self):
        assert self.storage == [x.data_ptr() for x in state_tensors(self.model, self.optimizer)]
