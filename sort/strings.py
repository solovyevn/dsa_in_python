#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""String sorting algorithms."""


def key_indexed_counting(a):
    n = len(a)
    r = 256
    aux = [0, ] * n
    count = [0, ] * (r + 1)
    for c in a:
        count[ord(c)+1] += 1
    i = 0
    while i < r:
        count[i+1] += count[i]
        i += 1
    for c in a:
        aux[count[ord(c)]] = c
        count[ord(c)] += 1
    for i in range(len(aux)):
        a[i] = aux[i]
    return a


def lsd_string_sort(a, w):
    n = len(a)
    r = 256
    aux = [0, ] * n
    d = w - 1
    while d >= 0:
        count = [0, ] * (r + 1)
        for i in range(n):
            count[ord(a[i][d])+1] += 1
        for i in range(r):
            count[i+1] += count[i]
        for i in range(n):
            aux[count[ord(a[i][d])]] = a[i]
            count[ord(a[i][d])] += 1
        for i in range(n):
            a[i] = aux[i]
        d -= 1
    return a


class MSD():

    def __init__(self):
        self._r = 256
        self._m = 5  # Cutoff for small subarrays
        self._aux = []

    def char_at(self, s, d):
        if d < len(s):
            return ord(s[d])
        return -1

    def sort(self, a):
        n = len(a)
        self._aux = [0, ] * n
        a = self._sort(a, 0, n-1, 0)
        return a

    def _sort(self, a, lo, hi, d):
        # if lo >= hi:
            # return a
        if (hi <= lo + self._m):
            return self._insertion_sort(a, lo, hi, d)
        count = [0, ] * (self._r + 2)
        for i in range(lo, hi+1):
            count[self.char_at(a[i], d)+2] += 1
        for r in range(0, self._r+1):
            count[r+1] += count[r]
        for i in range(lo, hi+1):
            self._aux[count[self.char_at(a[i], d)+1]] = a[i]
            count[self.char_at(a[i], d)+1] += 1
        for i in range(lo, hi+1):
            a[i] = self._aux[i-lo]
        for r in range(0, self._r):
            # print(f'_sort({a}, {lo+count[r]}, {lo+count[r+1]-1}, {d+1})')
            self._sort(a, lo+count[r], lo+count[r+1]-1, d+1)
        return a

    def _insertion_sort(self, a, lo, hi, d):
        # print(f'_insertion_sort({a}, {lo}, {hi}, {d})')
        for i in range(lo, hi+1):
            j = i
            while j > lo and self._less(a[j], a[j-1], d):
                a[j], a[j-1] = a[j-1], a[j]
                j -= 1
        return a

    def _less(self, v, w, d):
        return v[d:] < w[d:]


class QuickThreeWayStringSort():

    def __init__(self):
        pass

    def char_at(self, s, d):
        if d < len(s):
            return ord(s[d])
        return -1

    def sort(self, a):
        return self._sort(a, 0, len(a)-1, 0)

    def _sort(self, a, lo, hi, d):
        if hi <= lo:
            return a
        lt = lo
        gt = hi
        v = self.char_at(a[lo], d)
        i = lo + 1
        while i <= gt:
            t = self.char_at(a[i], d)
            if t < v:
                a[lt], a[i] = a[i], a[lt]
                lt += 1
                i += 1
            elif t > v:
                a[gt], a[i] = a[i], a[gt]
                gt -= 1
            else:
                i += 1
        self._sort(a, lo, lt-1, d)
        if v >= 0:
            self._sort(a, lt, gt, d+1)
        self._sort(a, gt+1, hi, d)
        return a


if __name__ == "__main__":
    a = ['w', 'v', 's', 'r', 'a', 'a', 'b', 'e', 'f', 'g', 'h', 'w', 'y', 'n']
    expected_a = ['a', 'a', 'b', 'e', 'f', 'g', 'h', 'n', 'r', 's', 'v', 'w', 'w', 'y']
    a = key_indexed_counting(a)
    print(f'A: {a}')
    if a == expected_a:
        print('OK')
    else:
        print('FAIL')
    l = [
        '4PGC938',
        '2IYE230',
        '3CIO720',
        '1ICK750',
        '1OHV845',
        '4JZY524',
        '1ICK750',
        '3CIO720',
        '1OHV845',
        '1OHV845',
        '2RLA629',
        '2RLA629',
        '3ATW723',
    ]
    expected_l = [
        '1ICK750',
        '1ICK750',
        '1OHV845',
        '1OHV845',
        '1OHV845',
        '2IYE230',
        '2RLA629',
        '2RLA629',
        '3ATW723',
        '3CIO720',
        '3CIO720',
        '4JZY524',
        '4PGC938',
    ]
    l = lsd_string_sort(l, 7)
    print(f'L: {l}')
    print(f'L_: {expected_l}')
    if l == expected_l:
        print('OK')
    else:
        print('FAIL')

    m = [
        'she',
        'sells',
        'seashells',
        'by',
        'the',
        'sea',
        'shore',
        'the',
        'shells',
        'she',
        'sells',
        'are',
        'surely',
        'seashells',
    ]
    expected_m = [
        'are',
        'by',
        'sea',
        'seashells',
        'seashells',
        'sells',
        'sells',
        'she',
        'she',
        'shells',
        'shore',
        'surely',
        'the',
        'the',
    ]
    msd = MSD()
    m_ = msd.sort(m.copy())
    print(f'M: {m_}')
    print(f'M: {expected_m}')
    if m_ == expected_m:
        print('OK')
    else:
        print('FAIL')

    qs = QuickThreeWayStringSort()
    m_ = qs.sort(m.copy())
    print(f'Q: {m_}')
    print(f'Q: {expected_m}')
    if m_ == expected_m:
        print('OK')
    else:
        print('FAIL')
