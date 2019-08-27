#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Substring search algorithms."""


def brute_force_searh(pattern, txt):
    M = len(pattern)
    N = len(txt)
    for i in range(0, N-M+1):
        j = 0
        while j < M:
            if txt[i+j] != pattern[j]:
                break
            j += 1
        if j == M:
            return i
    return N


def brute_force_searh_alt(pattern, txt):
    M = len(pattern)
    N = len(txt)
    i = 0
    j = 0
    while j < M and i < N:
        if txt[i] == pattern[j]:
            j += 1
        else:
            i -= j
            j = 0
        i += 1
    if j == M:
        return i - M
    return N


class KnuthMorrisPratt():

    def __init__(self):
        self._r = 256

    def char_at(self, s, d):
        return ord(s[d])

    def _make_dfa(self, pattern):
        self._dfa[self.char_at(pattern, 0)][0] = 1
        x = 0
        for j in range(1, len(pattern)):
            for c in range(self._r):
                self._dfa[c][j] = self._dfa[c][x]
            self._dfa[self.char_at(pattern, j)][j] = j+1
            x = self._dfa[self.char_at(pattern, j)][x]

    def search(self, pattern, txt):
        M = len(pattern)
        N = len(txt)
        self._dfa = [[0 for _ in range(M)] for _ in range(self._r)]
        self._make_dfa(pattern)
        i = 0
        j = 0
        while j < M and i < N:
            j = self._dfa[self.char_at(txt, i)][j]
            i += 1
        if j == M:
            return i - M
        return N


class BoyerMoore():

    def __init__(self):
        self._r = 256

    def char_at(self, s, d):
        return ord(s[d])

    def _make_table(self, pattern):
        self._right = [-1, ] * self._r
        for j in range(len(pattern)):
            self._right[self.char_at(pattern, j)] = j

    def search(self, pattern, txt):
        self._make_table(pattern)

        M = len(pattern)
        N = len(txt)
        i = 0
        while i < N-M:
            skip = 0
            j = M - 1
            while j >= 0:
                if txt[i+j] != pattern[j]:
                    skip = j - self._right[self.char_at(txt, i+j)]
                    if skip < 1:
                        skip = 1
                j -= 1
            if skip == 0:
                return i
            i += skip
        return N


class RabinKarp():

    def __init__(self):
        self._r = 256

    def get_random_prime(self):
        return 997  # Random enough for a start

    def char_at(self, s, d):
        return ord(s[d])

    def hash(self, key, m):
        h = 0
        for j in range(m):
            h = (self._r * h + self.char_at(key, j)) % self._q
        return h

    def check(self, pattern, txt, i, is_monte_carlo=True):
        if is_monte_carlo:
            return True
        else:
            # Implement "Las Vegas". Check characters match
            for j in range(len(pattern)):
                if txt[i] != pattern[j]:
                    return False
                i += 1
            return True

    def search(self, pattern, txt):
        M = len(pattern)
        N = len(txt)
        self._q = self.get_random_prime()
        pat_hash = self.hash(pattern, M)
        rm = 1
        for i in range(1, M):
            rm = (self._r * rm) % self._q
        txt_hash = self.hash(txt, M)
        if txt_hash == pat_hash:
            return 0
        for i in range(M, N):
            txt_hash = (txt_hash + self._q - rm * self.char_at(txt, i-M) % self._q) % self._q
            txt_hash = (txt_hash * self._r + self.char_at(txt, i)) % self._q
            if pat_hash == txt_hash:
                if self.check(pattern, txt, i-M+1, is_monte_carlo=False):
                    return i - M + 1
        return N


if __name__ == "__main__":
    sample_str = 'ABACADABRAC'
    patterns = ['ABRA', 'SOME', ]
    expected_res = [6, 11, ]
    for s in [brute_force_searh, brute_force_searh_alt,
              KnuthMorrisPratt().search,
              BoyerMoore().search,
              RabinKarp().search]:
        print(f'\n-= {s.__name__ if "__self__" not in dir(s) else s.__self__.__class__.__name__} =-\n')
        for (p, exp_res) in zip(patterns, expected_res):
            print(f'Sample string: {sample_str}')
            print(f'Pattern: {p}')
            res = s(p, sample_str)
            print(f'Match result: {res}')
            if res == exp_res:
                print('OK')
            else:
                print('FAIL')
            print()
