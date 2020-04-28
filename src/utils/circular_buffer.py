import numpy as np

__author__ = 'Otilia Stretcu'


class CircularBuffer(object):
    def __init__(self, capacity, dtype=np.float32):
        assert 0 < capacity < np.inf
        self.capacity = capacity
        self.buffer = np.zeros((capacity,), dtype=dtype)
        self.index_next = 0
        self.is_full = False

    def empty(self):
        return not self.is_full and self.index_next == 0

    def add(self, elem):
        self.buffer[self.index_next] = elem
        self.index_next = (self.index_next + 1) % self.capacity
        self.is_full = self.is_full or self.index_next == 0

    def average(self):
        if self.is_full:
            return np.mean(self.buffer)
        elif self.empty():
            raise AssertionError('The buffer is empty! Cannot average.')
        return np.mean(self.buffer[:self.index_next])

    def reset(self):
        self.index_next = 0
        self.is_full = False

    def peek(self):
        """Peek at the oldest element."""
        if self.is_full:
            return self.buffer[self.index_next]
        if self.empty():
            return None
        return self.buffer[0]
