import os
import jax
import psutil
from src.utils import util
from pynvml.smi import nvidia_smi

__UNIT_MAP = {
    'B': 1,
    'KiB': 2 ** 10,
    'MiB': 2 ** 20,
    'GiB': 2 ** 30
}


def gpu_split(arr, over=0, use_gpu=True):
    """Splits the first axis of `arr` evenly across the number of devices."""
    num_devices = jax.device_count('gpu' if use_gpu else 'cpu')
    if num_devices > arr.shape[over]:
        return arr
    if over == 0:
        return arr.reshape(num_devices, arr.shape[over] // num_devices, *arr.shape[1:])
    else:
        return arr.reshape(num_devices, *arr.shape[:over], arr.shape[over] // num_devices, *arr.shape[over + 1:])


def gpu_expand(arr, on=jax.local_devices()):
    return jax.device_put_replicated(arr, on)


def get_min_gpu_memory_in_bytes():
    instance = nvidia_smi.getInstance()
    total_memories = instance.DeviceQuery('memory.total')['gpu']
    return min(map(lambda x: x['fb_memory_usage']['total'] * __UNIT_MAP[x['fb_memory_usage']['unit']], total_memories))


def get_free_gpu_memory_in_bytes():
    instance = nvidia_smi.getInstance()
    total_memories = instance.DeviceQuery('memory.free')['gpu']
    return min(map(lambda x: x['fb_memory_usage']['free'] * __UNIT_MAP[x['fb_memory_usage']['unit']], total_memories))


def jax_preallocate_gpu():
    return 'XLA_PYTHON_CLIENT_PREALLOCATE' in os.environ and \
           bool(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']) and os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] != 'false'


def jax_mem_fraction():
    if jax_preallocate_gpu() and 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
        return float(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
    else:
        return 1.


def get_float_size():
    return 8 if jax.config.jax_enable_x64 is not None and jax.config.jax_enable_x64 else 4


def get_usable_cpu_memory_in_bytes():
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')
    return available


def get_usable_gpu_memory_in_bytes():  # Todo: this will not work for preallocate mode, fix this!
    if jax_preallocate_gpu():
        return jax_mem_fraction() * get_min_gpu_memory_in_bytes() * .95
    else:
        return get_free_gpu_memory_in_bytes() * .95
