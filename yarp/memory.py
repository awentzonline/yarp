import random

import numpy as np


class Memory(object):
    def __init__(self, max_items):
        self._last_index = 0
        self.data = []
        self.max_items = max_items

    def add(self, item):
        if self._last_index < self.max_items:
            self.data.append(item)
        else:
            self.data[self.mask(self._last_index)] = item
        self._last_index += 1

    def mask(self, index):
        return index % len(self.data)

    def sample(self, size):
        data = random.sample(self.data, size)
        return data

    def sample_batch(self, size):
        samples = random.sample(self.data, size)
        s = np.concatenate([_[0] for _ in samples])
        a = np.stack([_[1] for _ in samples])
        r = np.array([_[2] for _ in samples])
        s2 = np.concatenate([_[3] for _ in samples])
        t = np.array([_[4] for _ in samples])
        # for x in s, a, r, s2, t:
        #     print x.shape
        return s, a, r, s2, t

    @property
    def size(self):
        return min(len(self.data), self._last_index)
