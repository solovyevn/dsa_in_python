#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Implements Union-Find."""


class UF():

    # Note: For simplicity there're no checks later whether elements provided
    #       as arguments to methods are less than `n`.
    def __init__(self, n):
        """Initializes the data structure with the specified number of sets."""
        self._count = n
        # An array of set identifiers
        self._set = [p for p in range(n)]

    def find(self, p):
        """Returns an integer identifier of a set with p."""
        raise NotImplementedError()

    def union(self, p, q):
        """Merges two sets contatining p and q into a single set."""
        raise NotImplementedError()

    def connected(self, p, q):
        """Checks whether both p and q are in the same set."""
        # Complexity is the same as `find` in specific implementation, see below
        return self.find(p) == self.find(q)

    def count(self):
        """Returns number of sets."""
        # O(1), obviously
        return self._count


class UFQuickFind(UF):
    # Here each index in the `self._set` array is the element and array value
    # with that index is the integer identifier of its set.

    def find(self, p):
        # O(1)
        return self._set[p]

    def union(self, p, q):
        # O(n)
        id_p = self.find(p)
        id_q = self.find(q)
        if id_p == id_q:
            return
        for i in range(0, len(self._set)):
            if self._set[i] == id_q:
                self._set[i] = id_p
        self._count -= 1


class UFQuickUnion(UF):
    # Here each index in the `self._set` array is the element and array value
    # with that index is another element, belonging to the same set, - this
    # forms a `link` between elements of the same set.
    # An element of the array which index and value match is the `root`
    # of the set.

    def find(self, p):
        # O(n)
        # Here `n` is really the height of the tree in the forest (in the worst case it's
        # the size of array), but could be improved by controlling the
        # tree sizes and therefore heights
        while self._set[p] != p:
            p = self._set[p]
        return p

    def union(self, p, q):
        # O(n), because of `find`
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        self._set[root_q] = root_p
        self._count -= 1


class UFWeightedQuickUnion(UF):
    # Same as `UFQuickUnion` but with additional array to hold tree sizes

    def __init__(self, n):
        super().__init__(n)
        # By default each set is represented by a single element,
        # so it's size is 1
        self._sizes = [1, ] * self._count  # Same length and indexing as `_set`

    def find(self, p):
        # O(log(n))
        # Basicly `log(n)` here is the `n` in simple `UFQuickUnion`, but
        # because the heights of the trees is controlled in `union`, it's
        # `log(nodes_count)`, `nodes_count` is the size of the tree in the
        # forest. Note, that it's not the size of the forest, i.e. number of
        # sets, but a tree, i.e. the set size.
        # For a tree with 2^n nodes the height is n.
        while self._set[p] != p:
            p = self._set[p]
        return p

    def union(self, p, q):
        # O(log(n)), again because of `find`
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        # Now the smaller tree is always attached to the larger one, the choice
        # is not arbitrary anymore, the size of the final tree is adjusted
        if self._sizes[root_p] > self._sizes[root_q]:
            self._set[root_q] = root_p
            self._sizes[root_p] += self._sizes[root_q]
        else:
            self._set[root_p] = root_q
            self._sizes[root_q] += self._sizes[root_p]
        self._count -= 1


class UFWeightedQuickUnionWithPathCompression(UF):
    # Enhanced version of `UFWeightedQuickUnion`
    # `find` reduces the height of the trees. This is additional work, but for
    # future `find` calls and `union` calls it decreases the time to find
    # the root.

    def __init__(self, n):
        super().__init__(n)
        # By default each set is represented by a single element,
        # so it's size is 1
        self._sizes = [1, ] * self._count  # Same length and indexing as `_set`

    def find(self, p):
        # O(log(n))
        # It looks like the same as for Weighter Quick-Union, but due to
        # the path compression amortized performance is better.
        while self._set[p] != p:
            self._set[p] = self._set[self._set[p]]
            p = self._set[p]
        return p

    def union(self, p, q):
        # O(log(n)), again because of `find`
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p == root_q:
            return
        # Now the smaller tree is always attached to the larger one, the choice
        # is not arbitrary anymore, the size of the final tree is adjusted
        if self._sizes[root_p] > self._sizes[root_q]:
            self._set[root_q] = root_p
            self._sizes[root_p] += self._sizes[root_q]
        else:
            self._set[root_p] = root_q
            self._sizes[root_q] += self._sizes[root_p]
        self._count -= 1


if __name__ == "__main__":
    all_connections = [
        [
            (4, 3),
            (3, 8),
            (6, 5),
            (9, 4),
            (2, 1),
            (8, 9),
            (5, 0),
            (7, 2),
            (6, 1),
            (1, 0),
            (6, 7)
        ],
        [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (4, 6),
            (0, 4),
            (4, 8),
            (0, 9),
        ]
    ]

    all_expected_connections = [
        [
            (4, 3),
            (3, 8),
            (6, 5),
            (9, 4),
            (2, 1),
            # (8, 9),  # Already connected at this point
            (5, 0),
            (7, 2),
            (6, 1),
            # (1, 0),  # Already connected at this point
            # (6, 7)  # Already connected at this point
        ],
        [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (4, 6),
            (0, 4),
            (4, 8),
            (0, 9),
        ]
    ]

    all_expected_sets_count = [2, 1]
    for expected_connections, expected_sets_count, connections in zip(
            all_expected_connections, all_expected_sets_count, all_connections):
        print('Connections:')
        print('\n'.join([str(c) for c in connections]))
        for uf, name in zip((UFQuickFind, UFQuickUnion, UFWeightedQuickUnion,
                             UFWeightedQuickUnionWithPathCompression),
                            ('Quick-Find', 'Quick-Union', 'Weighted Quick-Union',
                             'Weighted Quick-Union with path compression')):
            print(f'-== Union-Find {name} ==-')
            resulting_connections = []
            uf = uf(10)
            print(f'Initial sets count: {uf.count()}')
            for c in connections:
                if not uf.connected(c[0], c[1]):
                    resulting_connections.append(c)
                    print(f'{c}  # Sets count: {uf.count()}')
                    uf.union(c[0], c[1])
                    print(f'SET: {uf._set}')
                else:
                    print(f'{c} is already connected  # Sets count: {uf.count()}')
            print('Result:')
            print(f'SET: {uf._set}')
            print('\n'.join([ str(rc) for rc in resulting_connections]))
            if expected_sets_count == uf.count():
                print(f'OK: {uf.count()}')
            else:
                print(f'FAIL: {uf.count()}')
            if expected_connections == resulting_connections:
                print('OK')
            else:
                print('FAIL')
