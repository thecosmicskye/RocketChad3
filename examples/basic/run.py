from config import (CONFIG, make_model, action_table, Observation, reward, make_state_setter,
                    terminal_condition, make_optimizer, critic_parameters)
from rocketchad import train

# Pass your own example components to train, or use --config with the training command.
train(CONFIG, output='runs/example', make_model=make_model, action_table=action_table,
      Observation=Observation, reward=reward, make_state_setter=make_state_setter,
      terminal_condition=terminal_condition, make_optimizer=make_optimizer, critic_parameters=critic_parameters)
