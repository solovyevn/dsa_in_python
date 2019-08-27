#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Implements Heap."""


class Heap():

    def __init__(self):
        # Not using the first (0 index) element in the array
        self._arr = [0, ]
        self.size = 0

    def _less(self, i, j):
        return self._arr[i] < self._arr[j]

    def _swap(self, i, j):
        self._arr[i], self._arr[j] = self._arr[j], self._arr[i]

    def _swim(self, k):
        while k > 1 and self._less(k//2, k):
            self._swap(k//2, k)
            k = k // 2

    def _sink(self, k):
        while 2*k <= self.size:
            j = 2 * k  # j is left child
            if j < self.size and self._less(j, j+1):
                j += 1  # j is right child
            if not self._less(k, j):
                break
            self._swap(k, j)
            k = j

    def heapify(self, a):
        self._arr = a
        self.size = len(a)
        self._arr.insert(0, 0)
        for i in range(self.size//2, 0, -1):
            self._sink(i)

    def insert(self, item):
        # O(log(n))
        self._arr.append(item)
        self.size += 1
        self._swim(self.size)

    def del_max(self):
        # O(log(n))
        res = self._arr[1]
        self._swap(1, self.size)
        del self._arr[self.size]
        self.size -= 1
        self._sink(1)
        return res

    def _heapify(self, items):
        pass

    def __repr__(self):
        return str(self._arr)


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
    heap = Heap()
    print(f'Initial: {heap}')
    items  = ['A', 'E', 'E', 'M', 'P', 'L']
    exp = [0, 'P', 'M', 'L', 'A', 'E', 'E']
    heap.heapify(items)
    print(f'Heapified: {heap}')
    if exp == heap._arr:
        print('OK')
    else:
        print('FAIL')
