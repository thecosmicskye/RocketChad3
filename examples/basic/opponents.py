import copy
import hashlib
import io
from pathlib import Path

import torch

from rocketchad.league import LeaguePool


# Entries and frozen weights must match across ranks. Each entry targets one
# team_size; list a model separately for each mode. retry=True permits a beaten
# member back after the pool is tried and its existing games finish.
# Each member's key is its checkpoint file's SHA-256; use the same key for the
# same file even when adding it under a different name or path.
# With a replay state setter enabled, set starts to 'kickoff' or 'replay' to
# override its replay fraction for that opponent. Replay-only opponents need
# frames for their mode; without them those slots use the current policy.
# Use open-play frames to exclude kickoffs. The default is 'mixed'. A custom
# state setter must expose replay_modes as a set of supported team sizes.
# After changing modes, call pool.draw() on all ranks between rollouts.
# Reserved arenas train one side, including when they fall back to self-play.
# Add the same members on all ranks between rollouts with add_member; metadata
# must use basic Python types. model_factory recreates added models on resume.
# A custom pool supplies learner_indices, rebind, begin_rollout, override_actions,
# on_step, end_rollout, state_dict/load_state_dict and
# model_state_dict/load_model_state_dict.
# Mixed-mode pools should use CONFIG's league_fraction; every rank needs the
# same number of learner_indices.
def make_opponents(env, model, config, saved_state=None):
    entries = config.get('opponents', ()) if saved_state is None else saved_state['members']
    if not entries and saved_state is None:
        return None
    members = []
    for entry in entries:
        frozen = copy.deepcopy(model)
        if saved_state is None:
            data = Path(entry['path']).read_bytes()
            entry = dict(entry, key=hashlib.sha256(data).hexdigest())
            checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
            frozen.load_state_dict(checkpoint['model'])
        name = entry.get('name') or (Path(entry['path']).stem if 'path' in entry else str(len(members)))
        members.append(dict(entry, name=name, model=frozen))
    fraction = config.get('league_fraction', 0.25) if saved_state is None else saved_state['fraction']
    return LeaguePool(env, members, fraction, env.seed,
                      precision=config['precision'],
                      model_factory=lambda _: copy.deepcopy(model),
                      retire_goal_share=config.get('league_retire_goal_share', 0.5),
                      min_goals=config.get('league_min_goals', 64),
                      resample_updates=config.get('league_resample_updates', 8))
