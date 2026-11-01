# On Linux, optional per-GPU NUMA pinning can be enabled before launch with
# ROCKETCHAD_RANK_CPU_PINNING=numa_local and ROCKETCHAD_CPUS_PER_RANK=8.
# Use CUDA_DEVICE_ORDER=PCI_BUS_ID and integer CUDA_VISIBLE_DEVICES; set
# ROCKETCHAD_RANK_CPU_SMT=1 to include sibling threads allowed by your cpuset.
# The system PCI list must match CUDA's GPU ordinals, including in containers.
# If your launcher already pins each rank, set ROCKETCHAD_RANK_CPUS_INHERIT=1.
from rocketchad import train
from config import (CONFIG, make_model, action_table, Observation, reward, make_state_setter,
                    terminal_condition, make_optimizer, critic_parameters)

# Pass your own example components to train, or use --config with the training command.
train(CONFIG, output='runs/example', make_model=make_model, action_table=action_table,
      Observation=Observation, reward=reward, make_state_setter=make_state_setter,
      terminal_condition=terminal_condition, make_optimizer=make_optimizer, critic_parameters=critic_parameters)
