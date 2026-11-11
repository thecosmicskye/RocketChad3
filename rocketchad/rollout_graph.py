import functools
import gc
import json
import sys
import traceback
from contextlib import contextmanager
import torch

_SKIP_MODULES = {'builtins', 'torch', 'numpy', 'pathlib', 'rocketgpu_py', 'rocketgpu_py_cuda', '_rocketgpu_start_cache'}


@contextmanager
def capture(graph, **kwargs):
    """torch.cuda.graph with cyclic GC paused. Torch's static Triton launcher unloads a kernel's module when the
    kernel is collected, and a module unload during capture invalidates the capture (CUDA 900)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, **kwargs):
            yield
    finally:
        if enabled:
            gc.enable()


def _is_project_module(module):
    if not module:
        return False
    if module in ('__main__', '<run_path>') or module.startswith('rocketchad'):
        return True
    root = module.split('.')[0]
    if root.startswith('_') or root in sys.stdlib_module_names or root in _SKIP_MODULES:
        return False
    origin = getattr(getattr(sys.modules.get(module), '__spec__', None), 'origin', '') or ''
    return 'site-packages' not in origin and 'dist-packages' not in origin


def _scan(roots):
    root_ids = {id(obj) for obj in roots if obj is not None}
    seen, found, generators, stack = (set(), [], {}, list(roots))
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen or isinstance(obj, (torch.nn.Module, torch.Tensor, type)):
            continue
        if isinstance(obj, torch.Generator):
            if obj.device.type == 'cuda':
                generators[id(obj)] = obj
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if isinstance(obj, functools.partial):
            stack.append(obj.func)
            stack.extend(obj.args)
            if obj.keywords:
                stack.extend(obj.keywords.values())
            stack.extend(vars(obj).values())
            continue
        wrapped = getattr(obj, '__wrapped__', None)
        if wrapped is not None:
            stack.append(wrapped)
        owner = getattr(obj, '__self__', None)
        if owner is not None:
            stack.append(owner)
        func = getattr(obj, '__func__', None)
        if func is not None:
            stack.append(func)
        if hasattr(obj, '__closure__'):
            if id(obj) in root_ids or _is_project_module(getattr(obj, '__module__', None)):
                for cell in obj.__closure__ or ():
                    try:
                        stack.append(cell.cell_contents)
                    except ValueError:
                        pass
            continue
        if hasattr(obj, '__dict__') and (id(obj) in root_ids or _is_project_module(getattr(type(obj), '__module__', None))):
            found.append(obj)
            stack.extend(vars(obj).values())
    return (found, list(generators.values()))


def _state(objects):
    tensors, scalars = ({}, {})
    for obj in objects:
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                tensors[id(obj), name] = (obj, value)
            elif isinstance(value, (bool, int, float)):
                scalars[id(obj), name] = (type(obj).__name__, type(value), value)
    return (tensors, scalars)


class RolloutGraph:
    """Captures one rollout decision step per rollout and replays it for the remaining steps.

    Rollout objects rebind state tensors every step (self.slot = torch.where(...)), and a replay would keep
    reading the pre-capture tensors, so inside the graph each rebound tensor is copied back into its own buffer.
    Capture runs on the physics stream, which must be the current stream (native start caches pin their stream).
    A step that changes python scalars on those objects, or that cannot be captured, runs eagerly instead."""

    def __init__(self, roots, physics):
        self.roots = roots
        self.stream = torch.cuda.ExternalStream(physics.stream)
        self.scalar_warned = False
        self.begin()

    def begin(self):
        self.graph, self.out, self.eager = (None, None, False)

    def __call__(self, decide, note_exposure=None):
        if self.graph is not None:
            self.graph.replay()
            out = self.out
        elif self.eager:
            out = decide()
        else:
            objects, generators = _scan(self.roots)
            tensors, scalars = _state(objects)
            out = decide()
        if note_exposure is not None:
            note_exposure()
        if self.graph is not None or self.eager:
            return out
        after, after_scalars = _state(objects)
        if after_scalars != scalars:
            if not self.scalar_warned:
                self.scalar_warned = True
                changed = [f'{(after_scalars.get(key) or scalars[key])[0]}.{key[1]}'
                           for key in dict.fromkeys((*scalars, *after_scalars))
                           if scalars.get(key) != after_scalars.get(key)]
                print(json.dumps(dict(rollout_graph='scalar_changed', attributes=changed)), flush=True)
            return out
        rebound = [(obj, key[1]) for key, (obj, value) in after.items() if tensors.get(key, (None, None))[1] is not value]
        for obj, name in rebound:
            setattr(obj, name, getattr(obj, name).clone())
        buffers = [getattr(obj, name) for obj, name in rebound]
        graph = torch.cuda.CUDAGraph()
        for generator in generators:
            graph.register_generator_state(generator)
        try:
            with capture(graph, stream=self.stream, capture_error_mode='thread_local'):
                self.out = decide()
                for (obj, name), buffer in zip(rebound, buffers):
                    if getattr(obj, name) is not buffer:
                        buffer.copy_(getattr(obj, name))
                        setattr(obj, name, buffer)
        except RuntimeError as error:
            if 'captur' not in str(error).lower():
                raise
            for (obj, name), buffer in zip(rebound, buffers):
                setattr(obj, name, buffer)
            self.eager = True
            print(json.dumps(dict(rollout_graph='eager_fallback', error=str(error), traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))), flush=True)
            return out
        self.graph = graph
        return out
