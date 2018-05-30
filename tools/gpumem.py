import subprocess
from time import time
import os
import torch


def get_gpu_memory_map(used=True):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=memory.' + ('used' if used else 'free'),
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


# https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
class Occupier(object):
    def __init__(self):
        self.gpu = os.environ['CUDA_VISIBLE_DEVICES']
        if len(self.gpu) == 0:
            raise RuntimeError('CUDA_VISIBLE_DEVICES is not set?')
        else:
            self.gpu = int(self.gpu)

    def occupy(self):
        if hasattr(self, 'memory'):
            print('Already busy')
        else:
            t = time()
            free = get_gpu_memory_map(False)
            used = get_gpu_memory_map(True)
            total = free[self.gpu] + used[self.gpu]
            print("Occupying memory: currently at {}/{}".format(
                used[self.gpu], total))
            self.memory = torch.CharTensor(int(free[self.gpu] * 1e6 * .9)).cuda()
            used_after = get_gpu_memory_map(True)
            print("Done. Memory: {0:d}/{1:d} [{2:.1f} s.]".format(
                used_after[self.gpu],
                total,
                time() - t))

    def free(self):
        if hasattr(self, 'memory'):
            print('Deleting occupied memory')
            del self.memory

    def is_busy(self):
        return (True if hasattr(self, 'memory') else False)
