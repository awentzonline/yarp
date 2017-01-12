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

    @property
    def size(self):
        return min(len(self.data), self._last_index)
