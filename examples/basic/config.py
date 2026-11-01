from actions import action_table
from episode import make_state_setter, terminal_condition
from model import make_model, critic_parameters
from observations import Observation
from optimizer import make_optimizer
from rewards import reward

# Optional: start with the ball 600 units above the field.
# from episode import high_ball_start as make_state_setter
# Optional: also end an episode if the ball goes above 1,500 units.
# from episode import end_when_ball_high as terminal_condition

# Or start half of episodes from converted replay states; see replay_starts.py:
# from replay_starts import ReplayStarts
# def make_state_setter(env):
#     return ReplayStarts(env, 'replay_states.pt', fraction=0.5)

# Arena and minibatch sizes are per GPU. Rollout graphs replay GPU operations,
# so custom callbacks used with them should not depend on Python side effects.
# For mixed modes, add 'team_size_fractions': {1: 0.25, 2: 0.25}; the rest
# are 3v3 arenas. Custom observations and rewards should return six rows per
# arena; env.team_layout identifies active slots and env.logical_car_field
# maps car data into blue-three, orange-three order.
# Multi-GPU models must use every optimized parameter on every rank.
# To use standard DDP, set 'grad_reduce': 'ddp', leave microbatch at 0, and use
# every trainable parameter on every rank.
# rollout_host_obs stores rollout observations in BF16 host memory.
# Set 'goal_telemetry': True to expose env.goal_events in rewards. Event
# validity assumes a goal ends the episode.
# Optional entropy control: set a positive entropy and add
# 'entropy_band': {'low': 1, 'high': 2, 'up': 1.0001, 'down': 0.99995,
#                  'ema_updates': 20, 'clamp': 4} to CONFIG.
CONFIG = {
    'arenas': 8192, 'team_size': 3, 'tick_skip': 8, 'episode_seconds': 30,
    'minibatch': 8192, 'microbatch': 0, 'epochs': 1, 'clip': 0.2, 'entropy': 0.0,
    'graphed_compile': False, 'rollout_compile': False, 'rollout_graph': False,
    'rollout_forward_rows': 0, 'rollout_host_obs': False, 'compile_obs': False, 'timeout_bootstrap': False,
    'optimizer': 'muon', 'adam_lr': 0.0003, 'adamw_lr': 0.0003, 'muon_lr': 0.001, 'optimizer_eps': 1e-5,
    'muon_momentum': 0.95, 'muon_ns_steps': 5, 'weight_decay': 0.0,
    'precision': 'bf16', 'seed': 1, 'checkpoint_seconds': 600, 'seconds': 0,
    # Train fresh weights with ball-velocity rewards for 60 seconds, then goals/concedes.
    'phases': [
        {'name': 'velocity', 'seconds': 60, 'horizon': 4, 'gamma': 0.99, 'gae_lambda': 0.95, 'critic_burnin': 0},
        {'name': 'goals', 'seconds': 0, 'horizon': 32, 'gamma': 0.999, 'gae_lambda': 0.99, 'critic_burnin': 16, 'reset_critic': True},
    ],
}
