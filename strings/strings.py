#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""String algorithms."""


import math


class Alphabet():

    def __init__(self, s):
        """Creates new alphabet from chars in `s`."""
        self._s = s
        self._a = {}
        for i in range(len(s)):
            self._a[s[i]] = i

    def to_char(self, index):
        """Converts `index` to corresponding alphabet character."""
        return self._s[index]

    def to_index(self, c):
        """Converts `c` to an index between 0 and R-1."""
        return self._a[c]

    def contains(self, c):
        """Checks if `c` is in alphabet."""
        return c in self._a

    def R(self):
        """Returns radix - number of characters in the alphabet."""
        return len(self._s)

    def lg_R(self):
        """Returns number of bits required to represent an index."""
        return int(math.ceil(math.log2(self.R())))

    def to_indices(self, s):
        """Converts characters in `s` to a list of base-R integers."""
        return [self._a[c] for c in s]

    def to_chars(self, indices):
        """Converts a list of base-R integers to characters over this alphabet."""
        return "".join([self._s[i] for i in indices])

    def __str__(self):
        return str(self._a)


if __name__ == "__main__":
    sample_alphabet = 'ABCDR'
    sample_str = 'ABRACADABRA'
    expected_indices = [0, 1, 4, 0, 2, 0, 3, 0, 1, 4, 0]
    for A in [Alphabet, ]:
        print(f'\n-= {A.__name__} =-\n')
        a = A(sample_alphabet)
        print(f'Alphabet:\n{a}\n')
        print(f'Sample string: {sample_str}\n')

        r = a.R()
        print(f'R: {r}')
        if r == 5:
            print('OK')
        else:
            print('FAIL')

        lg_r = a.lg_R()
        print(f'Lg R: {lg_r}')
        if lg_r == 3:
            print('OK')
        else:
            print('FAIL')

        indices = a.to_indices(sample_str)
        print(f'Indices: {indices}')
        if indices == expected_indices:
            print('OK')
        else:
            print('FAIL')

        chars = a.to_chars(indices)
        print(f'Chars: {chars}')
        if chars == sample_str:
            print('OK')
        else:
            print('FAIL')

        c = 'S'
        contains = a.contains(c)
        print(f'Alphabet contains "{c}": {contains}')
        if contains == False:
            print('OK')
        else:
            print('FAIL')
        c = 'R'
        contains = a.contains(c)
        print(f'Alphabet contains "{c}": {contains}')
        if contains == True:
            print('OK')
        else:
            print('FAIL')

        index = a.to_index(c)
        print(f'Index of {c} is {index}.')
        if index == 4:
            print('OK')
        else:
            print('FAIL')

        char = a.to_char(index)
        print(f'Char at {index} is {char}.')
        if char == c:
            print('OK')
        else:
            print('FAIL')

        print('Char count:')
        count = [0, ] * a.R()
        expected_count = [5, 2, 1, 1, 2]
        for c in sample_str:
            if a.contains(c):
                count[a.to_index(c)] += 1
        for i in range(len(count)):
            print(f'{a.to_char(i)}: {count[i]}')
        if count == expected_count:
            print('OK')
        else:
            print('FAIL')
    # ========================================================================
    R = 256
