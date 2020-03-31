import random

from Config import Config
from Utils import to_tensor


class MemoryReplay:
    def __init__(self, config: Config):
        self.memory = []
        self.index = 0
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.device = config.device

    def __len__(self):
        return len(self.memory)

    def append(self, elem):
        if len(self) < self.capacity:
            self.memory.append(elem)
        else:
            self.memory[self.index] = elem
            self.index = (self.index + 1) % self.capacity

    def sample(self):
        return [to_tensor(e, device=self.device) for e in zip(*random.sample(self.memory, self.batch_size))]