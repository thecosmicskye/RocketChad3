import torch
import torch.distributed as dist

from .precision import training_forward

STARTS = {'mixed': 0, 'kickoff': 1, 'replay': 2}


def sample(logits):
    probabilities = logits.log_softmax(-1).exp()
    return probabilities.div(torch.empty_like(probabilities).exponential_()).argmax(-1)


class LeaguePool:
    def __init__(self, env, members, fraction, seed, *, precision='bf16',
                 retire_goal_share=None, min_goals=64, resample_updates=8, model_factory=None):
        self.members = []
        seen = set()
        for member in members:
            identity = (member.get('team_size', 3), member['key'])
            if identity not in seen:
                seen.add(identity)
                self.members.append(dict(member))
        self.precision, self.model_factory = precision, model_factory
        self.fraction = fraction
        self.retire_goal_share, self.min_goals = retire_goal_share, min_goals
        self.resample_updates = resample_updates
        self.generator = torch.Generator().manual_seed(seed)
        for member in self.members:
            member['model'].to(env.device).eval().requires_grad_(False)
        self.forward = [training_forward(member['model'], precision) for member in self.members]
        self.retired = [False] * len(self.members)
        self.beaten = set()
        self.tried = [False] * len(self.members)
        self.totals = [[0, 0] for _ in self.members]
        self.stats = torch.zeros((len(self.members), 3), device=env.device)
        self.start_modes = torch.tensor([0] + [STARTS[m.get('starts', 'mixed')] for m in self.members], device=env.device)
        self.active = torch.full((env.arenas,), -1, dtype=torch.int64, device=env.device)
        self.pending = self.active.clone()
        self.rebind(env)
        self.draw()
        self.active.copy_(self.pending)

    def add_member(self, member):
        member = dict(member)
        identity = (member.get('team_size', 3), member['key'])
        if member['key'] in self.beaten or any((item.get('team_size', 3), item['key']) == identity for item in self.members):
            return
        member['model'].to(self.env.device).eval().requires_grad_(False)
        self.members.append(member)
        self.forward.append(training_forward(member['model'], self.precision))
        self.retired.append(False)
        self.tried.append(False)
        self.totals.append([0, 0])
        self.stats = torch.cat((self.stats, self.stats.new_zeros((1, 3))))
        self.start_modes = torch.cat((self.start_modes, self.start_modes.new_tensor([STARTS[member.get('starts', 'mixed')]])))
        self.draw()

    def rebind(self, env):
        self.env = env
        env.opponents = self
        self.active.copy_(self.pending)
        arena = torch.arange(env.arenas, device=env.device)
        slots = torch.arange(env.players, device=env.device)
        count = 2 * round(env.arenas * self.fraction / 2)
        self.league_mask = arena >= env.arenas - count
        self.learner_blue = arena.remainder(2) == 0
        present = (torch.ones((env.arenas, env.players), dtype=torch.bool, device=env.device)
                   if env.team_layout is None else env.team_layout.present)
        side = slots[None] // (env.players // 2)
        frozen = present & self.league_mask[:, None] & (side != arena[:, None].remainder(2))
        self.learner_indices = (present & ~frozen).flatten().nonzero().flatten()
        self.frozen_indices = frozen.flatten().nonzero().flatten()
        self.arena_modes = (torch.full((env.arenas,), env.players // 2, dtype=torch.int64, device=env.device)
                            if env.team_layout is None else env.team_layout.sizes)

    def draw(self):
        modes = self.arena_modes.cpu().tolist()
        league = self.league_mask.cpu().tolist()
        presence = torch.zeros(len(self.members), dtype=torch.int32, device=self.env.device)
        if self.members:
            presence.scatter_add_(0, self.active.clamp(min=0), (self.active >= 0).to(torch.int32))
            if dist.is_initialized():
                dist.all_reduce(presence)
        active = {i for i, count in enumerate(presence.cpu().tolist()) if count}
        keep = [i for i, member in enumerate(self.members) if member['key'] not in self.beaten or i in active]
        if len(keep) != len(self.members):
            remap = self.active.new_full((len(self.members),), -1)
            remap[keep] = torch.arange(len(keep), device=self.env.device)
            self.active.copy_(torch.where(self.active >= 0, remap[self.active.clamp(min=0)], -1))
            self.members = [self.members[i] for i in keep]
            self.forward = [self.forward[i] for i in keep]
            self.retired = [self.retired[i] for i in keep]
            self.tried = [self.tried[i] for i in keep]
            self.totals = [self.totals[i] for i in keep]
            self.stats = self.stats[keep]
            self.start_modes = self.start_modes[[0] + [i + 1 for i in keep]]
            active = {new for new, old in enumerate(keep) if old in active}
        pending = torch.full((self.env.arenas,), -1, dtype=torch.int64)
        for mode in (1, 2, 3):
            arenas = [arena for arena, size in enumerate(modes) if league[arena] and size == mode]
            matching = [i for i, member in enumerate(self.members)
                        if member.get('team_size', 3) == mode and member['key'] not in self.beaten]
            available = [i for i in matching if not self.retired[i] or
                         (self.members[i].get('retry', False) and i not in active)]
            if available and all(self.tried[i] for i in available):
                retry = [i for i in available if self.retired[i]]
                if retry:
                    for i in available:
                        self.tried[i] = False
                    for i in retry:
                        self.retired[i] = False
                        self.totals[i] = [0, 0]
            eligible = [i for i in matching if not self.retired[i]]
            if not arenas or not eligible:
                continue
            weights = torch.tensor([self.members[i].get('weight', 1.0) for i in eligible], dtype=torch.float64)
            draws = torch.multinomial(weights, len(arenas), replacement=True, generator=self.generator).tolist()
            untried = [i for i in eligible if not self.tried[i]]
            for index, member in enumerate(untried[:len(draws)]):
                draws[index] = eligible.index(member)
            pending[arenas] = torch.tensor([eligible[i] for i in draws], dtype=torch.int64)
        self.pending.copy_(pending.to(self.env.device))

    def begin_rollout(self, updates):
        self.updates = updates + 1
        frozen = self.frozen_indices.cpu()
        arenas = frozen.div(self.env.players, rounding_mode='floor')
        active, pending = self.active.cpu()[arenas], self.pending.cpu()[arenas]
        self.groups = []
        for member in range(-1, len(self.members)):
            selected = (active == member) | (pending == member)
            if selected.any():
                self.groups.append((member, frozen[selected].to(self.env.device), arenas[selected].to(self.env.device)))

    def override_actions(self, observations, actions, current_forward):
        for member, rows, arenas in self.groups:
            forward = current_forward if member < 0 else self.forward[member]
            logits, _ = forward(observations.index_select(0, rows))
            chosen = sample(logits)
            previous = actions.index_select(0, rows)
            actions.index_copy_(0, rows, torch.where(self.active[arenas] == member, chosen, previous))
        return actions

    def replay_starts(self, reset, sampled):
        modes = self.start_modes[self.pending + 1]
        return reset & ((modes == 2) | ((modes == 0) & sampled))

    def on_step(self, done, info):
        if self.members:
            mask = self.league_mask & (self.active >= 0)
            indices = self.active.clamp(min=0)
            blue, orange = info['blue_goals'].float(), info['orange_goals'].float()
            ours = torch.where(self.learner_blue, blue, orange) * mask
            theirs = torch.where(self.learner_blue, orange, blue) * mask
            self.stats[:, 0].scatter_add_(0, indices, ours)
            self.stats[:, 1].scatter_add_(0, indices, theirs)
            self.stats[:, 2].scatter_add_(0, indices, mask.float())
        self.active.copy_(torch.where(done & self.league_mask, self.pending, self.active))

    def end_rollout(self, world):
        if world > 1 and self.members:
            dist.all_reduce(self.stats)
        summary = self.stats.cpu().tolist()
        self.stats.zero_()
        changed = False
        for i, (goals_for, goals_against, played) in enumerate(summary):
            self.tried[i] |= played > 0
            self.totals[i][0] += int(goals_for)
            self.totals[i][1] += int(goals_against)
            total = sum(self.totals[i])
            share = self.totals[i][0] / total if total else None
            if (share is not None and not self.retired[i] and self.retire_goal_share is not None and total >= self.min_goals
                    and share >= self.retire_goal_share):
                self.retired[i] = changed = True
                if not self.members[i].get('retry', False):
                    key = self.members[i]['key']
                    self.beaten.add(key)
                    for j, member in enumerate(self.members):
                        if member['key'] == key:
                            self.retired[j] = True
        result = [dict(name=member.get('name', str(i)), goals_for=self.totals[i][0],
                       goals_against=self.totals[i][1],
                       goal_share=self.totals[i][0] / sum(self.totals[i]) if sum(self.totals[i]) else None,
                       retired=self.retired[i]) for i, member in enumerate(self.members)]
        waiting = any(self.retired[i] and member.get('retry', False) and member['key'] not in self.beaten
                      for i, member in enumerate(self.members))
        waiting |= not self.resample_updates and any(member['key'] in self.beaten for member in self.members)
        if changed or waiting or (self.resample_updates and self.updates % self.resample_updates == 0):
            self.draw()
        return result

    def state_dict(self):
        return dict(pending=self.pending.cpu(), retired=self.retired,
                    beaten=list(self.beaten), tried=self.tried, totals=self.totals,
                    rng=self.generator.get_state(), fraction=self.fraction,
                    members=[{key: value for key, value in member.items() if key != 'model'}
                             for member in self.members])

    def model_state_dict(self):
        return [{key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                 for key, value in member['model'].state_dict().items()} for member in self.members]

    def load_state_dict(self, state):
        existing = {(m.get('team_size', 3), m['key']): m['model'] for m in self.members}
        self.members = []
        for member in state['members']:
            model = existing.get((member.get('team_size', 3), member['key']))
            if model is None:
                model = self.model_factory(member)
            self.members.append(dict(member, model=model))
            model.to(self.env.device).eval().requires_grad_(False)
        self.forward = [training_forward(member['model'], self.precision) for member in self.members]
        self.stats = torch.zeros((len(self.members), 3), device=self.env.device)
        self.start_modes = torch.tensor([0] + [STARTS[m.get('starts', 'mixed')] for m in self.members], device=self.env.device)
        self.pending.copy_(state['pending'].to(self.env.device))
        self.active.copy_(self.pending)
        self.retired, self.tried, self.totals = state['retired'], state['tried'], state['totals']
        self.beaten = set(state['beaten'])
        self.generator.set_state(state['rng'])

    def load_model_state_dict(self, states):
        for member, weights in zip(self.members, states):
            member['model'].load_state_dict(weights)
