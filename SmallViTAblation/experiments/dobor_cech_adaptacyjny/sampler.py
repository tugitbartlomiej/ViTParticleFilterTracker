"""Mutable indices, immutable dataset and epoch length; safe with persistent workers."""
import math
import numpy as np


class AdaptiveSampler:
    def __init__(self, indices, batch_size, seed):
        self.budget = len(indices)
        if self.budget < 1 or batch_size < 1:
            raise ValueError('positive budget and batch size required')
        self.length = math.ceil(self.budget / batch_size) * batch_size
        self.seed = seed
        self.update(indices, epoch=0)

    def update(self, indices, epoch):
        if len(indices) != self.budget or len(set(indices)) != self.budget or min(indices) < 0:
            raise ValueError('indices must be unique, nonnegative and preserve the budget')
        self.indices, self.epoch = list(map(int, indices)), int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        values = np.array(self.indices)
        rng.shuffle(values)
        if len(values) < self.length:
            values = np.concatenate([values, rng.choice(values, self.length-len(values), replace=True)])
        return iter(values.tolist())

    def __len__(self):
        return self.length
