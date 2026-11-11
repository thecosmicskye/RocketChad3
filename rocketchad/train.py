import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

if os.environ.get('ROCKETCHAD_RANK_CPU_PINNING') and os.environ.get('ROCKETCHAD_RANK_CPUS_INHERIT') != '1':
    from .rank_cpus import rank_cpus
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('WORLD_SIZE', rank + 1)))
    os.sched_setaffinity(0, rank_cpus(rank, local_world))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .environment import Environment
from .distributed_ppo import normalize_advantages
from .entropy_band import MultiModeEntropyBand, make_entropy_band
from .host_rollout import HostObservations
from .inference import chunked_rollout_forward
from .learner import Learner, Optimizers
from .policy_terms import action_log_probs
from .precision import training_forward
from .rollout_graph import RolloutGraph
from .truncation_gae import TimeoutValues, gae as truncation_gae

FUSED_AUTOTUNE = dict(max_autotune=True, coordinate_descent_tuning=True, benchmark_epilogue_fusion=False)


def checkpoint(path, model, optimizer, config, phase_index, phase_updates, phase_seconds, updates, steps, elapsed, device, world, entropy_band=None, opponents=None):
    state = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device).cpu()}
    if opponents is not None:
        state['opponents'] = opponents.state_dict()
        state['opponent_fraction'] = config.get('league_fraction', 0.25 if config.get('opponents') else 0)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if int(os.environ.get('RANK', '0')) == 0:
        temporary = path.with_suffix('.tmp')
        saved_config = {key: value for key, value in config.items() if key != '_entropy_tensor'}
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=saved_config,
                        phase_index=phase_index, phase_updates=phase_updates, phase_seconds=phase_seconds,
                        updates=updates, player_steps=steps, elapsed_seconds=elapsed, rng=states,
                        entropy_band=None if entropy_band is None else entropy_band.state_dict(),
                        opponent_models=None if opponents is None else opponents.model_state_dict()), temporary)
        os.replace(temporary, path)


def train(config, output, *, make_model, action_table, Observation, reward, make_state_setter, terminal_condition,
          make_optimizer, critic_parameters, make_opponents=None, resume=None):
    config = dict(config)
    example = dict(action_table=action_table, Observation=Observation, reward=reward,
                   make_state_setter=make_state_setter, terminal_condition=terminal_condition)
    output = Path(output)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    original_stream = torch.cuda.current_stream(device)
    group_initialized = False
    try:
        if config.get('rollout_graph', False):
            torch.cuda.set_stream(torch.cuda.Stream(device=device))
        if world > 1:
            os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
            dist.init_process_group('nccl', device_id=device)
            group_initialized = True
        return _train(config, output, example, rank, world, device, make_model, make_optimizer,
                      critic_parameters, make_opponents, resume)
    finally:
        try:
            if group_initialized:
                dist.destroy_process_group()
        finally:
            if config.get('rollout_graph', False):
                torch.cuda.set_stream(original_stream)


def _train(config, output, example, rank, world, device, make_model, make_optimizer, critic_parameters, make_opponents, resume):
    entropy_band = make_entropy_band(config, device)
    if entropy_band is not None:
        config['_entropy_tensor'] = entropy_band.tensor
    action_floor = float(config.get('action_floor', 0.0))
    torch.manual_seed(config['seed'])
    model = make_model().to(device)
    if world > 1:
        with torch.no_grad():
            for tensor in (*model.parameters(), *model.buffers()):
                dist.broadcast(tensor, src=0)
    def build_optimizer():
        optimizer = make_optimizer(model, config)
        return optimizer if isinstance(optimizer, Optimizers) else Optimizers(eager=(optimizer,), graph=False)

    optimizer = build_optimizer()
    phase_index = phase_updates = updates = steps = 0
    previous_seconds = phase_previous_seconds = 0.0
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        phase_index, phase_updates = state['phase_index'], state['phase_updates']
        phase_previous_seconds, previous_seconds = state['phase_seconds'], state['elapsed_seconds']
        updates, steps = state['updates'], state['player_steps']
        torch.set_rng_state(state['rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['rng'][rank]['cuda'], device)
        if entropy_band is not None and state.get('entropy_band') is not None:
            entropy_band.load_state_dict(state['entropy_band'])
    else:
        torch.manual_seed(config['seed'] + rank)
    shared = model
    if config.get('graphed_compile', False):
        import torch._dynamo as dynamo
        dynamo.config.optimize_ddp = False
        options = dict(options=FUSED_AUTOTUNE) if config.get('rollout_compile', False) else dict(mode='default')
        shared = torch.compile(model, fullgraph=True, dynamic=False, **options)
    learner_stream = None
    if world > 1 and config.get('grad_reduce', 'flat_fp32') == 'ddp':
        learner_stream = torch.cuda.Stream(device=device)
        learner_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(learner_stream):
            shared = DistributedDataParallel(shared, device_ids=[device.index], broadcast_buffers=False,
                                             gradient_as_bucket_view=True, static_graph=True)
        torch.cuda.current_stream(device).wait_stream(learner_stream)
    learner = Learner(model, optimizer, config, tuple(critic_parameters(model)), world, shared=shared, stream=learner_stream)
    timeout_forward = training_forward(model, config['precision'])
    rollout_forward = timeout_forward
    if config.get('rollout_compile', False):
        rollout_forward = torch.compile(rollout_forward, dynamic=True, options=FUSED_AUTOTUNE)
    rollout_forward = chunked_rollout_forward(rollout_forward, config.get('rollout_forward_rows', 0))

    def make_environment(*, initialize=True):
        env = Environment(config, example, config['seed'] + rank * 1000003, device,
                          initialize=initialize, generation=updates)
        if config.get('compile_obs', False):
            env.observe = torch.no_grad()(torch.compile(env.observe, dynamic=False,
                                                       options={'triton.cudagraphs': False}))
            jitter = getattr(env.set_state, 'jitter', None)
            if jitter is not None:
                jitter.compile(env.arenas)
        return env

    def player_rows(env, opponents):
        full = env.arenas * env.players
        active = opponents.learner_indices if opponents is not None else (
            None if env.team_layout is None else env.team_layout.present.flatten().nonzero().flatten())
        rows = full if active is None else active.numel()
        learner_modes = mode_columns = None
        if env.team_layout is not None:
            slot_modes = env.team_layout.sizes.repeat_interleave(env.players)
            learner_modes = slot_modes if active is None else slot_modes.index_select(0, active)
            if config.get('per_mode_advantage_norm', True):
                mode_columns = tuple((learner_modes == mode).nonzero().flatten() for mode in (1, 2, 3))
        return full, active, rows, learner_modes, mode_columns

    saved_opponents = state['rng'][rank].get('opponents') if resume else None
    if saved_opponents is not None:
        config['league_fraction'] = state['rng'][rank]['opponent_fraction']
    env = make_environment(initialize=False)
    opponents = None if make_opponents is None else make_opponents(env, model, config, saved_opponents)
    if saved_opponents is not None and opponents is not None:
        opponents.load_state_dict(saved_opponents)
        opponents.load_model_state_dict(state['opponent_models'])
    env.opponents = opponents
    env.initialize()
    rollout_graph = None
    host_observations = None
    try:
        rollout_graph = RolloutGraph((env, opponents), env) if config.get('rollout_graph', False) else None
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        started = phase_started = time.monotonic()
        next_save = config['checkpoint_seconds']
        obs = env.observe()
        full_rows, active, rows, learner_modes, mode_columns = player_rows(env, opponents)
        timeout_values = None
        while True:
            now = time.monotonic()
            elapsed = now - started
            phase_elapsed = phase_previous_seconds + now - phase_started
            phase = config['phases'][phase_index]
            decision = torch.zeros(3, dtype=torch.int32, device=device)
            if rank == 0:
                decision[0] = int(phase['seconds'] > 0 and phase_elapsed >= phase['seconds'] and phase_index + 1 < len(config['phases']))
                decision[1] = int(config['checkpoint_seconds'] > 0 and elapsed >= next_save)
                decision[2] = int(config['seconds'] > 0 and previous_seconds + elapsed >= config['seconds'])
            if world > 1:
                dist.broadcast(decision, src=0)
            transition, save, stop = decision.tolist()
            if transition:
                learner.close()
                rollout_graph = None
                if host_observations is not None:
                    host_observations.close()
                    host_observations = None
                if opponents is None:
                    env.close()
                phase_index += 1
                phase_updates = 0
                phase_previous_seconds = 0.0
                phase_started = time.monotonic()
                phase = config['phases'][phase_index]
                critics = tuple(critic_parameters(model))
                if phase.get('reset_critic', False):
                    with torch.no_grad():
                        for parameter in critics:
                            parameter.zero_()
                if phase.get('reset_optimizer', True):
                    optimizer = build_optimizer()
                elif phase.get('reset_critic', False):
                    for component in optimizer.all:
                        state_map = getattr(component, 'state', None)
                        if isinstance(state_map, dict):
                            for parameter in critics:
                                state_map.pop(parameter, None)
                learner = Learner(model, optimizer, config, critics, world, shared=shared, stream=learner_stream)
                if opponents is None:
                    env = make_environment()
                    obs = env.observe()
                rollout_graph = RolloutGraph((env, opponents), env) if config.get('rollout_graph', False) else None
                full_rows, active, rows, learner_modes, mode_columns = player_rows(env, opponents)
                timeout_values = None
                phase_elapsed = 0.0
            if save or stop:
                checkpoint(output / f'checkpoint_{updates}.pt', model, optimizer, config, phase_index, phase_updates,
                           phase_elapsed, updates, steps, previous_seconds + elapsed, device, world, entropy_band, opponents)
                next_save = elapsed + config['checkpoint_seconds']
            if stop:
                break
            horizon = phase['horizon']
            if opponents is not None:
                opponents.begin_rollout(updates)
            if config.get('rollout_host_obs', False):
                if host_observations is None:
                    host_observations = HostObservations(horizon, rows, obs.shape[-1], None, device, config['minibatch'])
                observations = None
            else:
                storage_dtype = torch.bfloat16 if config.get('rollout_obs_storage') == 'bf16' else torch.get_default_dtype()
                observations = torch.empty((horizon, rows, obs.shape[-1]), dtype=storage_dtype, device=device)
            actions = torch.empty((horizon, rows), dtype=torch.int64, device=device)
            log_probs = torch.empty((horizon, rows), device=device)
            values, rewards = torch.empty_like(log_probs), torch.empty_like(log_probs)
            dones = torch.empty_like(log_probs, dtype=torch.bool)
            if phase.get('timeout_bootstrap', config.get('timeout_bootstrap', False)):
                if timeout_values is None:
                    timeout_values = TimeoutValues(model, horizon, full_rows, config['minibatch'], device,
                                                   owned=active, obs_size=obs.shape[-1], players=env.players,
                                                   forward=timeout_forward)
                timeout_values.begin()
            else:
                timeout_values = None
            iteration_started = time.monotonic()
            with torch.no_grad():
                if rollout_graph is not None:
                    rollout_graph.begin()

                def decide():
                    nonlocal obs
                    learner_obs = obs if active is None else obs.index_select(0, active)
                    logits, value = rollout_forward(learner_obs)
                    log_all = action_log_probs(logits, action_floor)
                    probabilities = log_all.exp()
                    action = probabilities.div(torch.empty_like(probabilities).exponential_()).argmax(-1)
                    applied = action if active is None else action.new_zeros(full_rows).index_copy_(0, active, action)
                    if opponents is not None:
                        applied = opponents.override_actions(obs, applied, rollout_forward)
                    if timeout_values is None and opponents is None:
                        following, reward_value, done = env.step(applied, phase['name'])
                        info = None
                    else:
                        capture_mode = (('full' if rollout_graph is not None and not rollout_graph.eager else 'timeout')
                                        if timeout_values is not None else False)
                        following, reward_value, done, info = env.step(applied, phase['name'],
                            capture_transition=capture_mode, capture_info=opponents is not None)
                    if opponents is not None:
                        opponents.on_step(done.reshape(env.arenas, env.players)[:, 0], info)
                    if active is not None:
                        reward_value, done = reward_value.index_select(0, active), done.index_select(0, active)
                    if rollout_graph is None:
                        obs = following
                    else:
                        obs.copy_(following)
                    return action, log_all.gather(-1, action[:, None]).squeeze(-1), value, reward_value, done, info

                for index in range(horizon):
                    learner_obs = obs if active is None else obs.index_select(0, active)
                    if host_observations is None:
                        observations[index].copy_(learner_obs)
                    else:
                        host_observations.record(index, learner_obs)
                    action, log_prob, value, reward_value, done, info = decide() if rollout_graph is None else rollout_graph(decide)
                    actions[index] = action
                    log_probs[index] = log_prob
                    values[index], rewards[index], dones[index] = value, reward_value, done
                    if timeout_values is not None:
                        timeout_values.capture(index, info['transition_observation'], info)
                learner_obs = obs if active is None else obs.index_select(0, active)
                _, next_value = rollout_forward(learner_obs)
                if timeout_values is not None:
                    timeout_values.flush()
                    truncated = timeout_values.truncated if active is None else timeout_values.truncated.index_select(1, active)
                    timeout_bootstrap = timeout_values.values if active is None else timeout_values.values.index_select(1, active)
                    advantages, returns = truncation_gae(rewards, values, next_value, dones & ~truncated,
                                                         truncated, timeout_bootstrap,
                                                         phase['gamma'], phase['gae_lambda'])
                else:
                    advantages = torch.empty_like(rewards)
                    running = torch.zeros(rows, device=device)
                    for index in reversed(range(horizon)):
                        alive = (~dones[index]).float()
                        delta = rewards[index] + phase['gamma'] * next_value * alive - values[index]
                        running = delta + phase['gamma'] * phase['gae_lambda'] * alive * running
                        advantages[index] = running
                        next_value = values[index]
                    returns = advantages + values
                advantages = normalize_advantages(advantages, mode_columns=mode_columns)
            torch.cuda.synchronize(device)
            collection_end = time.monotonic()
            flat_obs = observations.flatten(0, 1) if host_observations is None else host_observations.end_rollout(horizon)
            per_mode_entropy = isinstance(entropy_band, MultiModeEntropyBand) and learner_modes is not None
            flat_modes = (learner_modes - 1).repeat(horizon) if per_mode_entropy else None
            sources = (flat_obs, actions.flatten(), log_probs.flatten(), advantages.flatten(), returns.flatten(),
                       *((flat_modes,) if flat_modes is not None else ()))
            metrics = torch.zeros(10 if flat_modes is not None else 4, device=device)
            count = 0
            burnin = phase_updates < phase['critic_burnin']
            if host_observations is not None:
                host_observations.cpu_wait_seconds = 0.0
            for _ in range(config['epochs']):
                permutation = torch.randperm(horizon * rows, device=device)
                if host_observations is None:
                    for indices in permutation.split(config['minibatch']):
                        metrics += learner.step(sources, indices, burnin=burnin)
                        count += 1
                else:
                    batches = list(permutation.split(config['minibatch']))
                    groups = (batches[:-1], batches[-1:]) if len(batches[-1]) != config['minibatch'] else (batches,)
                    for group in groups:
                        for _, indices, obs_rows, _ in host_observations.epoch(flat_obs, group, None):
                            metrics += learner.step(sources, indices, burnin=burnin, obs=obs_rows)
                            count += 1
            phase_updates += 1
            updates += 1
            steps += horizon * rows * world
            if world > 1:
                dist.all_reduce(metrics)
            torch.cuda.synchronize(device)
            mean_metrics = metrics / (count * world)
            entropy_weight = entropy_band.weight if entropy_band is not None else None
            mode_entropy = None
            if flat_modes is not None:
                mode_entropy = {mode + 1: (mean_metrics[4 + mode] / mean_metrics[7 + mode]).item()
                                if mean_metrics[7 + mode] > 0 else None for mode in range(3)}
            if entropy_band is not None and not burnin:
                entropy_band.update(mode_entropy if flat_modes is not None else mean_metrics[2].item())
            opponent_metrics = None if opponents is None else opponents.end_rollout(world)
            if rank == 0:
                total_seconds = time.monotonic() - iteration_started
                policy_loss, value_loss, entropy, kl = mean_metrics[:4].tolist()
                metrics_row = dict(phase=phase['name'], update=updates, player_steps=steps,
                                   elapsed_seconds=previous_seconds + time.monotonic() - started,
                                   player_steps_per_second=horizon * rows * world / total_seconds,
                                   collection_seconds=collection_end - iteration_started,
                                   policy_loss=policy_loss, value_loss=value_loss, entropy=entropy, approx_kl=kl)
                if entropy_band is not None:
                    metrics_row.update(entropy_weight=entropy_weight, entropy_weight_next=entropy_band.weight,
                                       entropy_ema=entropy_band.ema)
                if mode_entropy is not None:
                    metrics_row['mode_entropy'] = {mode: value for mode, value in mode_entropy.items() if value is not None}
                if opponent_metrics is not None:
                    metrics_row['opponents'] = opponent_metrics
                print(json.dumps(metrics_row), flush=True)
    finally:
        learner.close()
        rollout_graph = None
        if host_observations is not None:
            host_observations.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.config.resolve().parent))
    example = runpy.run_path(str(args.config.resolve()))
    keys = ('make_model', 'action_table', 'Observation', 'reward', 'make_state_setter', 'terminal_condition',
            'make_optimizer', 'critic_parameters')
    train(example['CONFIG'], args.out, **{key: example[key] for key in keys},
          make_opponents=example.get('make_opponents'), resume=args.resume)


if __name__ == '__main__':
    main()
