import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .environment import Environment
from .learner import Learner, Optimizers, forward


def normalize(advantages, world):
    values = advantages.double()
    moments = torch.stack((values.sum(), values.square().sum(), values.new_tensor(values.numel())))
    if world > 1:
        dist.all_reduce(moments)
    total, squares, count = moments.unbind()
    mean = total / count
    variance = ((squares - total * mean) / (count - 1).clamp_min(1)).clamp_min(0)
    return ((values - mean) / variance.sqrt().clamp_min(1e-8)).float()


def checkpoint(path, model, optimizer, config, phase_index, phase_updates, phase_seconds, updates, steps, elapsed, device, world):
    state = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device).cpu()}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if int(os.environ.get('RANK', '0')) == 0:
        temporary = path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                        phase_index=phase_index, phase_updates=phase_updates, phase_seconds=phase_seconds,
                        updates=updates, player_steps=steps, elapsed_seconds=elapsed, rng=states), temporary)
        os.replace(temporary, path)


def train(config, output, *, make_model, action_table, Observation, reward, make_state_setter, terminal_condition,
          make_optimizer, critic_parameters, resume=None):
    example = dict(action_table=action_table, Observation=Observation, reward=reward,
                   make_state_setter=make_state_setter, terminal_condition=terminal_condition)
    output = Path(output)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'
        dist.init_process_group('nccl', device_id=device)
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
    else:
        torch.manual_seed(config['seed'] + rank)
    learner = Learner(model, optimizer, config, critic_parameters(model), world)
    env = Environment(config, example, config['seed'] + rank * 1000003, device)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    started = phase_started = time.monotonic()
    next_save = config['checkpoint_seconds']
    obs = env.observe()
    rows = config['arenas'] * config['team_size'] * 2
    try:
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
                env.close()
                phase_index += 1
                phase_updates = 0
                phase_previous_seconds = 0.0
                phase_started = time.monotonic()
                phase = config['phases'][phase_index]
                if phase.get('reset_critic', False):
                    with torch.no_grad():
                        for parameter in critic_parameters(model):
                            parameter.zero_()
                optimizer = build_optimizer()
                learner = Learner(model, optimizer, config, critic_parameters(model), world)
                env = Environment(config, example, config['seed'] + rank * 1000003, device)
                obs = env.observe()
                phase_elapsed = 0.0
            if save or stop:
                checkpoint(output / f'checkpoint_{updates}.pt', model, optimizer, config, phase_index, phase_updates,
                           phase_elapsed, updates, steps, previous_seconds + elapsed, device, world)
                next_save = elapsed + config['checkpoint_seconds']
            if stop:
                break
            horizon = phase['horizon']
            observations = torch.empty((horizon, rows, obs.shape[-1]), device=device)
            actions = torch.empty((horizon, rows), dtype=torch.int64, device=device)
            log_probs = torch.empty((horizon, rows), device=device)
            values, rewards = torch.empty_like(log_probs), torch.empty_like(log_probs)
            dones = torch.empty_like(log_probs, dtype=torch.bool)
            iteration_started = time.monotonic()
            with torch.no_grad():
                for index in range(horizon):
                    observations[index].copy_(obs)
                    logits, value = forward(model, obs, config['precision'])
                    log_all = logits.log_softmax(-1)
                    probabilities = log_all.exp()
                    action = probabilities.div(torch.empty_like(probabilities).exponential_()).argmax(-1)
                    obs, reward_value, done = env.step(action, phase['name'])
                    actions[index] = action
                    log_probs[index] = log_all.gather(-1, action[:, None]).squeeze(-1)
                    values[index], rewards[index], dones[index] = value, reward_value, done
                _, next_value = forward(model, obs, config['precision'])
                advantages = torch.empty_like(rewards)
                running = torch.zeros(rows, device=device)
                for index in reversed(range(horizon)):
                    alive = (~dones[index]).float()
                    delta = rewards[index] + phase['gamma'] * next_value * alive - values[index]
                    running = delta + phase['gamma'] * phase['gae_lambda'] * alive * running
                    advantages[index] = running
                    next_value = values[index]
                returns = advantages + values
                advantages = normalize(advantages, world)
            torch.cuda.synchronize(device)
            collection_end = time.monotonic()
            sources = (observations.flatten(0, 1), actions.flatten(), log_probs.flatten(),
                       advantages.flatten(), returns.flatten())
            metrics = torch.zeros(4, device=device)
            count = 0
            for _ in range(config['epochs']):
                permutation = torch.randperm(horizon * rows, device=device)
                for start in range(0, len(permutation), config['minibatch']):
                    indices = permutation[start:start + config['minibatch']]
                    metrics += learner.step(sources, indices, burnin=phase_updates < phase['critic_burnin'])
                    count += 1
            phase_updates += 1
            updates += 1
            steps += horizon * rows * world
            if world > 1:
                dist.all_reduce(metrics)
            torch.cuda.synchronize(device)
            if rank == 0:
                total_seconds = time.monotonic() - iteration_started
                policy_loss, value_loss, entropy, kl = (metrics / (count * world)).tolist()
                print(json.dumps(dict(phase=phase['name'], update=updates, player_steps=steps,
                                      elapsed_seconds=previous_seconds + time.monotonic() - started,
                                      player_steps_per_second=horizon * rows * world / total_seconds,
                                      collection_seconds=collection_end - iteration_started,
                                      policy_loss=policy_loss, value_loss=value_loss, entropy=entropy, approx_kl=kl)), flush=True)
    finally:
        learner.close()
        env.close()
        if world > 1:
            dist.destroy_process_group()


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
    train(example['CONFIG'], args.out, **{key: example[key] for key in keys}, resume=args.resume)


if __name__ == '__main__':
    main()
