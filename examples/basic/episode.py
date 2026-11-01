# The setter runs at startup and every step. Arenas marked in its GPU boolean
# mask have been reset to kickoff; the mask may contain no true values.
def make_state_setter(env):
    return lambda reset: None


# Goal arguments mark new crossings in the current decision, not total scores.
# Return one GPU boolean per arena for true terminal states. The trainer handles
# timeouts separately, and bootstraps them if enabled in config.py.
def terminal_condition(env, blue_goals, orange_goals):
    return blue_goals | orange_goals


# Uncomment one or both examples in config.py to replace the defaults.
def high_ball_start(env):
    def set_state(reset):
        # RocketGPU stores positions in both game units and physics units.
        env.ball['pos'][:, 2].masked_fill_(reset, 600.0)
        env.ball['pos_bt'][:, 2].masked_fill_(reset, 600.0 / 50.0)

    return set_state


def end_when_ball_high(env, blue_goals, orange_goals):
    return terminal_condition(env, blue_goals, orange_goals) | (env.ball['pos'][:, 2] > 1500)
