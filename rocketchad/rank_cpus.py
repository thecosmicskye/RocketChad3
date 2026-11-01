import os
from pathlib import Path
ENV_MODE = 'ROCKETCHAD_RANK_CPU_PINNING'
ENV_PER_RANK = 'ROCKETCHAD_CPUS_PER_RANK'
ENV_SMT = 'ROCKETCHAD_RANK_CPU_SMT'
NUMA_LOCAL = 'numa_local'
DEFAULT_PER_RANK = 6

def mode(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get(ENV_MODE)
    if not value:
        return None
    if value != NUMA_LOCAL:
        raise ValueError
    return value

def per_rank(environ=None):
    environ = os.environ if environ is None else environ
    return int(environ.get(ENV_PER_RANK, DEFAULT_PER_RANK))

def smt(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get(ENV_SMT)
    if not value:
        return False
    if value != '1':
        raise ValueError
    return True

def parse_cpulist(text):
    cpus = []
    for part in text.strip().split(','):
        if not part:
            continue
        if '-' in part:
            low, high = part.split('-')
            cpus.extend(range(int(low), int(high) + 1))
        else:
            cpus.append(int(part))
    return cpus

def nvidia_gpus(sysfs='/sys'):
    found = []
    for device in sorted(Path(sysfs, 'bus/pci/devices').iterdir(), key=lambda p: p.name.lower()):
        try:
            vendor = (device / 'vendor').read_text().strip().lower()
            klass = (device / 'class').read_text().strip().lower()
        except OSError:
            continue
        if vendor == '0x10de' and klass.startswith('0x03'):
            found.append(device.name.lower())
    return found

def visible_gpus(environ=None, sysfs='/sys'):
    environ = os.environ if environ is None else environ
    if environ.get('CUDA_DEVICE_ORDER') not in (None, '', 'PCI_BUS_ID'):
        raise ValueError
    gpus = nvidia_gpus(sysfs)
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return gpus
    return [gpus[int(item)] for item in visible.split(',')]

def physical_primaries(node, sysfs='/sys'):
    cpus = parse_cpulist(Path(sysfs, 'devices/system/node', f'node{node}', 'cpulist').read_text())
    allowed = os.sched_getaffinity(0)
    result, seen = [], set()
    for cpu in cpus:
        if cpu in allowed:
            siblings = parse_cpulist(Path(sysfs, f'devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text())
            core = min(siblings)
            if core not in seen:
                seen.add(core)
                result.append(cpu)
    return result

def numa_plan(world, cpus_per_rank=DEFAULT_PER_RANK, environ=None, sysfs='/sys'):
    gpus = visible_gpus(environ, sysfs)
    plan, used = ([], {})
    for rank in range(world):
        bus = gpus[rank]
        node = int(Path(sysfs, 'bus/pci/devices', bus, 'numa_node').read_text().strip())
        ordinal = used.get(node, 0)
        used[node] = ordinal + 1
        block = physical_primaries(node, sysfs)[ordinal * cpus_per_rank:(ordinal + 1) * cpus_per_rank]
        if len(block) != cpus_per_rank:
            raise ValueError
        if smt(environ):
            block = sorted({sibling for cpu in block for sibling in parse_cpulist(Path(sysfs, f'devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text())} & os.sched_getaffinity(0))
        plan.append((bus, node, block))
    return plan

def rank_cpus(rank, world, environ=None, sysfs='/sys'):
    if mode(environ) is None:
        return set(range(rank * 6, (rank + 1) * 6))
    return set(numa_plan(world, per_rank(environ), environ, sysfs)[rank][2])
