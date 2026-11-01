import concurrent.futures
import time
import torch

class HostObservations:

    def __init__(self, max_h, n, obs_size, learner_indices, device, minibatch):
        self.device, self.dtype = (device, torch.bfloat16)
        self.indices = learner_indices
        self.rows = n if learner_indices is None else int(learner_indices.numel())
        self.obs_size = obs_size
        self.host = torch.empty((max_h, self.rows, obs_size), dtype=self.dtype)
        self.host_bytes = self.host.numel() * self.host.element_size()
        if int(torch.cuda.cudart().cudaHostRegister(self.host.data_ptr(), self.host_bytes, 0)):
            raise RuntimeError
        self.registered = True
        self.side = torch.cuda.Stream(device=device)
        self.stage = [torch.empty((self.rows, obs_size), dtype=self.dtype, device=device) for _ in range(2)]
        self.staged = [torch.cuda.Event() for _ in range(2)]
        self.d2h = [torch.cuda.Event() for _ in range(2)]
        self.d2h_live = [False, False]
        self.minibatch = None
        self._slots(minibatch)
        self.pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='host_obs_gather')

    def footprint(self):
        return dict(host_registered_bytes=self.host_bytes, storage='bf16', rows=self.rows)

    def record(self, step, obs):
        s = step & 1
        current = torch.cuda.current_stream(self.device)
        if self.d2h_live[s]:
            current.wait_event(self.d2h[s])
        self.stage[s].copy_(obs if self.indices is None else obs.index_select(0, self.indices))
        self.staged[s].record(current)
        self.side.wait_event(self.staged[s])
        with torch.cuda.stream(self.side):
            self.host[step].copy_(self.stage[s], non_blocking=True)
        self.d2h[s].record(self.side)
        self.d2h_live[s] = True

    def end_rollout(self, h):
        self.side.synchronize()
        self.d2h_live = [False, False]
        return self.host[:h].flatten(0, 1)

    def _slots(self, minibatch):
        if self.minibatch == minibatch:
            return
        if self.minibatch is not None:
            torch.cuda.current_stream(self.device).synchronize()
            self.side.synchronize()
        self.minibatch = minibatch
        self.pinned = [torch.empty((minibatch, self.obs_size), dtype=self.dtype, pin_memory=True) for _ in range(2)]
        self.gpu = [torch.empty((minibatch, self.obs_size), dtype=self.dtype, device=self.device) for _ in range(2)]
        self.h2d = [torch.cuda.Event() for _ in range(2)]
        self.used = [torch.cuda.Event() for _ in range(2)]
        self.h2d_live, self.used_live = ([False, False], [False, False])

    def _gather(self, flat, cpu_idx, s, wait):
        if wait is not None:
            wait.synchronize()
        torch.index_select(flat, 0, cpu_idx, out=self.pinned[s])
        return s

    def epoch(self, flat, idx_list, timing_iter):
        if not idx_list:
            return
        self._slots(int(idx_list[0].numel()))
        started = time.monotonic()
        cpu_idx = torch.stack(idx_list).cpu()
        self.cpu_wait_seconds += time.monotonic() - started
        futures = {k: self.pool.submit(self._gather, flat, cpu_idx[k], k & 1, self.h2d[k & 1] if self.h2d_live[k & 1] else None) for k in range(min(2, len(idx_list)))}
        current = torch.cuda.current_stream(self.device)
        for k, idx in enumerate(idx_list):
            started = time.monotonic()
            s = futures.pop(k).result()
            self.cpu_wait_seconds += time.monotonic() - started
            if self.used_live[s]:
                self.side.wait_event(self.used[s])
            with torch.cuda.stream(self.side):
                self.gpu[s].copy_(self.pinned[s], non_blocking=True)
            self.h2d[s].record(self.side)
            self.h2d_live[s] = True
            if k + 2 < len(idx_list):
                futures[k + 2] = self.pool.submit(self._gather, flat, cpu_idx[k + 2], s, self.h2d[s])
            timing = next(timing_iter) if timing_iter is not None else None
            if timing is not None:
                timing[0].record(current)
            current.wait_event(self.h2d[s])
            yield (k, idx, self.gpu[s], timing)
            self.used[s].record(current)
            self.used_live[s] = True

    def close(self):
        self.pool.shutdown(wait=True)
        if self.registered:
            torch.cuda.synchronize(self.device)
            torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            self.registered = False
