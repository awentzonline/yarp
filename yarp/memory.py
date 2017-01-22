import random

import numpy as np


class Memory(object):
    def __init__(self, state_shape, action_shape, max_items, sample_length=1):
        self._next_index = 0
        self.data = []
        self.states = np.zeros((max_items,) + state_shape).astype(np.float32)
        self.actions = np.zeros((max_items,) + action_shape).astype(np.int8)
        self.rewards = np.zeros((max_items,)).astype(np.float32)
        self.terminals = np.zeros((max_items,)).astype(np.bool)
        self.max_items = max_items
        self.sample_length = sample_length
        self.terminal_map = {}

    def add(self, state, action, reward, terminal, next_state):
        index = self.mask(self._next_index)
        index_p1 = self.mask(self._next_index + 1)
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index_p1] = terminal
        self.states[index_p1] = next_state
        was_terminal = index_p1 in self.terminal_map
        if was_terminal and not terminal:
            del self.terminal_map[index_p1]
        elif terminal and not was_terminal:
            self.terminal_map[index_p1] = True
        self._next_index += 1

    def mask(self, index):
        return index % self.max_items

    def sample_batch(self, size):
        indexes = np.array(
            [self.batch_safe_index(i) for i in self.random_indexes(size)])
        if self.sample_length == 1:
            return self._one_step_batch(indexes)
        else:
            return self._multi_step_batch(indexes)

    def _one_step_batch(self, indexes):
        s = self.states[indexes]
        a = self.actions[indexes]
        r = self.rewards[indexes]
        s2 = self.states[indexes + 1]
        t = self.terminals[indexes + 1]
        return s, a, r, s2, t

    def _multi_step_batch(self, indexes):
        s = []
        s2 = []
        for index in indexes:
            item_indexes = np.arange(index, index + self.sample_length + 1)
            all_states = self.states[item_indexes]
            s_i = all_states[:-1]
            s2_i = all_states[1:]
            s.append(s_i)
            s2.append(s2_i)
        s = np.stack(s).astype(np.float32)
        s2 = np.stack(s2).astype(np.float32)
        a = self.actions[indexes + self.sample_length - 1]
        r = self.rewards[indexes + self.sample_length - 1]
        t = self.terminals[indexes + self.sample_length]
        # for x in s, a, r, s2, t:
        #     print x.shape
        return s, a, r, s2, t

    def batch_safe_index(self, start_index):
        tries_remaining = 3
        while tries_remaining > 0:
            # this range is every step *except* the last
            indexes = range(start_index, start_index + self.sample_length)
            terminal_index = None
            for index in indexes:
                if index in self.terminal_map:
                    terminal_index = index
                    break  # needs another loop to reconcile this
            if terminal_index:
                start_index = terminal_index - self.sample_length
                if start_index < 0:
                    start_index = self.random_indexes(1)[0]
            else:
                break
            tries_remaining -= 1
        # if tries_remaining == 0:
        #     print 'failed to find an ideal index'
        return start_index


    def random_indexes(self, size):
        return np.random.randint(0, self.size - self.sample_length, (size,))

    @property
    def size(self):
        return min(self.max_items, self._next_index)

    @property
    def state_shape(self):
        return self.states.shape[1:]

    @property
    def state_sample_shape(self):
        return (self.sample_length,) + self.state_shape
