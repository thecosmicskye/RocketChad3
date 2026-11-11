import hashlib
import io
from pathlib import Path

import torch
import torch.distributed as dist

from rocketchad.league import LeaguePool


class WatchingLeague(LeaguePool):
    def __init__(self, *args, watch, **kwargs):
        self.watch = watch
        self.seen = {}
        super().__init__(*args, **kwargs)

    def poll(self):
        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        limit = self.watch.get('max_members', 20)
        room = limit - len(self.members)
        mode = self.watch.get('team_size', 3)
        starts = self.watch.get('starts', 'mixed')
        replace = self.watch.get('replace_weakest', False) and not any(m.get('departing') for m in self.members)
        known = self.beaten | {m['key'] for m in self.members if m.get('team_size', 3) == mode}
        candidates = [None]
        if rank == 0:
            found = []
            if (room > 0 or replace) and (starts != 'replay' or mode in self.replay_modes):
                try:
                    paths = list(Path(self.watch['directory']).glob(self.watch.get('pattern', '*.pt')))
                except OSError:
                    paths = []
                for path in paths:
                    try:
                        stat = path.stat()
                        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                        if self.seen.get(str(path)) != stamp:
                            found.append((stat.st_mtime_ns, str(path), stamp))
                    except OSError:
                        continue
            candidates[0] = sorted(found, reverse=True)
        if distributed:
            dist.broadcast_object_list(candidates, src=0, device=self.env.device)
        members = []
        for _, path, stamp in candidates[0]:
            if room <= 0 and (not replace or members):
                break
            packet = [None]
            if rank == 0:
                try:
                    data = Path(path).read_bytes()
                    packet[0] = (hashlib.sha256(data).hexdigest(), data)
                except OSError:
                    pass
            if distributed:
                dist.broadcast_object_list(packet, src=0, device=self.env.device)
            if packet[0] is None:
                continue
            key, data = packet[0]
            if key in known:
                self.seen[path] = stamp
                continue
            if room <= 0:
                scores = [((self.ema[i][0] / sum(self.ema[i]) if sum(self.ema[i]) else goals[0] / sum(goals)), i)
                          for i, goals in enumerate(self.totals)
                          if sum(goals) >= self.min_goals and not self.retired[i]
                          and not self.members[i].get('retry', False)
                          and self.members[i].get('team_size', 3) == mode]
                if scores:
                    self.remove_member(max(scores)[1])
                    room = limit - len(self.members)
                if room <= 0:
                    break
            entry = dict(path=path, key=key, name=Path(path).stem,
                         team_size=mode, starts=starts, retry=self.watch.get('retry', False))
            model, error = None, None
            try:
                model = self.model_factory(entry).to(self.env.device).eval().requires_grad_(False)
                checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model'])
                entry.setdefault('action_floor', checkpoint.get('config', {}).get('action_floor', 0.0))
            except Exception as exc:
                error = (f'{type(exc).__name__}: {exc}', isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)))
                del model
                model = None
            errors = [error]
            if distributed:
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, error)
            if any(errors):
                if rank == 0:
                    print(f'Could not load opponent {path}: {next(e[0] for e in errors if e)}', flush=True)
                if not any(e and e[1] for e in errors):
                    self.seen[path] = stamp
                del model
                continue
            self.seen[path] = stamp
            members.append(dict(entry, model=model))
            known.add(key)
            room -= 1
        self.add_members(members)

    def begin_rollout(self, updates):
        if updates % self.watch.get('scan_every', 1) == 0:
            self.poll()
        super().begin_rollout(updates)

    def state_dict(self):
        return dict(super().state_dict(), watch_seen=self.seen)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.seen = state.get('watch_seen', {})
