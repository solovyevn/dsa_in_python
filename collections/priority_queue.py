#!/usr/bin/env python
# _*_ encoding: utf-8 _*_


"""Implements Priority Queue."""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cs.collections.heap import Heap


class MaxPQ():

    def __init__(self, *, items=None):
        self._heap = Heap()
        if items:
            pass

    def insert(self, key):
        # O(log(n))
        self._heap.insert(key)

    def del_max(self):
        # O(log(n))
        return self._heap.del_max()

    def max(self):
        pass

    def is_empty(self):
        return self._heap.size == 0

    def size(self):
        return self._heap.size


if __name__ == "__main__":
    items = ['P', 'Q', 'E', 'del',
             'X', 'A', 'M', 'del',
             'P', 'L', 'E', 'del']
    expected = [
        [0, 'P'],
        [0, 'Q', 'P'],
        [0, 'Q', 'P', 'E'],
        [0, 'P', 'E'],  # del
        [0, 'X', 'E', 'P'],
        [0, 'X', 'E', 'P', 'A'],
        [0, 'X', 'M', 'P', 'A', 'E'],
        [0, 'P', 'M', 'E', 'A'],  # del
        [0, 'P', 'P', 'E', 'A', 'M'],
        [0, 'P', 'P', 'L', 'A', 'M', 'E'],
        [0, 'P', 'P', 'L', 'A', 'M', 'E', 'E'],
        [0, 'P', 'M', 'L', 'A', 'E', 'E'], # del
    ]
    heap = Heap()
    print(f'Initial: {heap}')
    for item, exp in zip(items, expected):
        if item != 'del':
            heap.insert(item)
            print(f'Inserted "{item}": {heap}')
        else:
            res = heap.del_max()
            print(f'Deleted  "{res}": {heap}')
        if exp == heap._arr:
            print('OK')
        else:
            print('FAIL')
