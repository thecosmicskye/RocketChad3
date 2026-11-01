# The setter runs at startup and every step. Arenas marked in its GPU boolean
# mask have been reset to kickoff; the mask may contain no true values.
def make_state_setter(env):
    return lambda reset: None


# Return one GPU boolean per arena. PPO treats all endings, including timeouts, as terminal.
def terminal_condition(env, blue_goals, orange_goals):
    return blue_goals | orange_goals | (env.age >= env.timeouts)


# Uncomment one or both examples in config.py to replace the defaults.
def high_ball_start(env):
    def set_state(reset):
        # RocketGPU stores positions in both game units and physics units.
        env.ball['pos'][:, 2].masked_fill_(reset, 600.0)
        env.ball['pos_bt'][:, 2].masked_fill_(reset, 600.0 / 50.0)

    return set_state


def end_when_ball_high(env, blue_goals, orange_goals):
    return terminal_condition(env, blue_goals, orange_goals) | (env.ball['pos'][:, 2] > 1500)
