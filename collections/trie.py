#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Implements Trie."""

from collections import deque


class StringST():

    def __init__(self):
        pass

    def put(self, key, value):
        # Deletes key if value is None
        if value is None:
            self.delete(key)
            return

    def get(self, key):
        pass

    def delete(self, key):
        pass

    def contains(self, key):
        return self.get(key) != None

    def is_empty(self):
        return self.size() == 0

    def size(self):
        pass

    def keys(self):
        pass

    def longest_prefix_of(self, s):
        pass

    def keys_with_prefix(self, s):
        pass

    def keys_that_match(self, s):
        pass


class TrieST(StringST):

    class Node():

        def __init__(self, value=None, r=256):
            self.value = value
            self.next = [None, ] * r

    def char_at(self, key, d):
        return ord(key[d])

    def __init__(self):
        self._n = 0
        self._r = 256
        self._root = None

    def put(self, key, value):
        super().put(key, value)
        self._root = self._put(self._root, key, value, 0)

    def _put(self, node, key, value, d):
        # Need to create a new node if node is a leaf or an empty root
        if node is None:
            node = self.Node()
        # We found a node corresponding to a key, set it's value
        if d == len(key):
            if node.value is None:
                self._n += 1
            node.value = value
            return node
        # Following the path to a node corresponding to the key
        c = self.char_at(key, d)
        node.next[c] = self._put(node.next[c], key, value, d+1)
        return node

    def get(self, key):
        node = self._get(self._root, key, 0)
        if node is not None:
            # Value is None if it's a miss and real value if it's a hit
            return node.value
        return None

    def _get(self, node, key, d):
        # Key is longer than its path in a tree, it's a miss
        if node is None:
            return None
        # This is a node, that is a hit, but only if there's a value, else it's a miss
        if d == len(key):
            return node
        c = self.char_at(key, d)
        return self._get(node.next[c], key, d+1)

    def delete(self, key):
        self._root = self._delete(self._root, key, 0)
        self._n -= 1

    def _delete(self, node, key, d):
        if node is None:
            return None
        if d == len(key):
            node.value = None
        else:
            c = self.char_at(key, d)
            node.next[c] = self._delete(node.next[c], key, d+1)
        if node.value is not None:
            return node
        for _node in node.next:
            if _node is not None:
                return node
        return None

    def size(self):
        return self._n

    def size_lazy(self):
        return self._size_lazy(self._root)

    def _size_lazy(self, node):
        if node is None:
            return 0
        count = 0
        if node.value is not None:
            count += 1
        for _node in node.next:
            if _node is not None:
                count += self._size_lazy(_node)
        return count

    def _collect(self, node, prefix, q):
        if node is None:
            return
        if node.value is not None:
            q.append(prefix)
        for c in range(self._r):
            self._collect(node.next[c], f'{prefix}{chr(c)}', q)

    def keys(self):
        return self.keys_with_prefix('')

    def keys_with_prefix(self, prefix):
        q = deque()
        node_with_prefix = self._get(self._root, prefix, 0)
        self._collect(node_with_prefix, prefix, q)
        return list(q)

    def _collect_w_pattern(self, node, prefix, pattern, q):
        if node is None:
            return
        d = len(prefix)
        if d == len(pattern) and node.value is not None:
            q.append(prefix)
        if d == len(pattern):
            return
        next_c = self.char_at(pattern, d)
        for c in range(self._r):
            if c == next_c or ord('.') == next_c:
                self._collect_w_pattern(node.next[c], f'{prefix}{chr(c)}', pattern, q)

    def keys_that_match(self, pattern):
        q = deque()
        self._collect_w_pattern(self._root, '', pattern, q)
        return list(q)

    def longest_prefix_of(self, s):
        max_length = self._search(self._root, s, 0, 0)
        return s[:max_length]

    def _search(self, node, s, d, max_length):
        if node is None:
            return max_length
        if node.value is not None:
            max_length = d
        if d == len(s):
            return max_length
        c = self.char_at(s, d)
        return self._search(node.next[c], s, d+1, max_length)


class TernarySearchTrie(StringST):

    class Node():

        def __init__(self, c, value=None):
            self.c = c
            self.value = value
            self.left = None  # Less than c
            self.mid = None  # Equal to c
            self.right = None  # More than c

    def char_at(self, key, d):
        return ord(key[d])

    def __init__(self):
        self._n = 0
        self._root = None

    def put(self, key, value):
        super().put(key, value)
        self._root = self._put(self._root, key, value, 0)

    def _put(self, node, key, value, d):
        c = self.char_at(key, d)
        # Need to create a new node if node is a leaf or an empty root
        if node is None:
            node = self.Node(c)
        # Following the path to a node corresponding to the key
        if c < node.c:
            node.left = self._put(node.left, key, value, d)
        elif c > node.c:
            node.right = self._put(node.right, key, value, d)
        elif d < len(key) - 1:
            node.mid = self._put(node.mid, key, value, d+1)
        # We found a node corresponding to a key, set it's value
        else:
            if node.value is None:
                self._n += 1
            node.value = value
        return node

    def get(self, key):
        node = self._get(self._root, key, 0)
        if node is not None:
            # Value is None if it's a miss and real value if it's a hit
            return node.value
        return None

    def _get(self, node, key, d):
        # Key is longer than its path in a tree, it's a miss
        if node is None or d >= len(key):
            return None
        c = self.char_at(key, d)
        if c < node.c:
            return self._get(node.left, key, d)
        elif c > node.c:
            return self._get(node.right, key, d)
        elif d < len(key) - 1:
            return self._get(node.mid, key, d+1)
        # This is a node, that is a hit, but only if there's a value, else it's a miss
        return node

    def delete(self, key):
        self._root = self._delete(self._root, key, 0)
        self._n -= 1

    def _delete(self, node, key, d):
        if node is None:
            return None
        if d == len(key):
            node.value = None
        else:
            c = self.char_at(key, d)
            node.next[c] = self._delete(node.next[c], key, d+1)
        if node.value is not None:
            return node
        for _node in node.next:
            if _node is not None:
                return node
        return None

    def size(self):
        return self._n

    def size_lazy(self):
        return self._size_lazy(self._root)

    def _size_lazy(self, node):
        if node is None:
            return 0
        count = 0
        if node.value is not None:
            count += 1
        for _node in [node.left, node.mid, node.right]:
            if _node is not None:
                count += self._size_lazy(_node)
        return count

    def _collect(self, node, prefix, q):
        if node is None:
            return
        if node.value is not None:
            q.append(prefix)
        for _node in [node.left, node.mid, node.right]:
            self._collect(_node, f'{prefix}{chr(node.c)}', q)

    def keys(self):
        return self.keys_with_prefix('')

    def keys_with_prefix(self, prefix):
        q = deque()
        node_with_prefix = self._get(self._root, prefix, 0)
        self._collect(node_with_prefix, prefix, q)
        return list(q)

    def _collect_w_pattern(self, node, prefix, pattern, q):
        if node is None:
            return
        d = len(prefix)
        if d == len(pattern) and node.value is not None:
            q.append(prefix)
        if d == len(pattern):
            return
        next_c = self.char_at(pattern, d)
        for _node in [node.left, node.mid, node.right]:
            if _node.c == next_c or ord('.') == next_c:
                self._collect_w_pattern(_node, f'{prefix}{chr(_node.c)}', pattern, q)

    def keys_that_match(self, pattern):
        q = deque()
        self._collect_w_pattern(self._root, '', pattern, q)
        return list(q)

    def longest_prefix_of(self, s):
        max_length = self._search(self._root, s, 0, 0)
        return s[:max_length]

    def _search(self, node, s, d, max_length):
        if node is None:
            return max_length
        if node.value is not None:
            max_length = d
        if d == len(s):
            return max_length
        c = self.char_at(s, d)
        for _node in [node.left, node.mid, node.right]:
            l = self._search(_node, s, d+1, max_length)
            if l > max_length:
                max_length = l


if __name__ == "__main__":
    keys = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    test_keys = ['shell', 'sea', 'theme', 'she', 's.e']
    longest_prefix = ['she', 'sea', 'the', 'she', '']
    keys_with_prefix = [['shells', ], ['sea', ], [], ['she', 'shells', ], []]
    keys_that_match = [[], ['sea', ], [], ['she', ], ['she', ]]
    vals = [0, 1, 6, 3, 4, 5, 6, 7]
    del_keys = ['she', 'sea', 'shore']
    for S in [TrieST, TernarySearchTrie]:

        print(f'\n-= {S.__name__} =-\n')
        st = S()

        print(f'Is empty: {st.is_empty()}')
        if st.is_empty() == True:
            print('OK')
        else:
            print('FAIL')

        print(f'Contains "she": {st.contains("she")}')
        if st.contains('she') == False:
            print('OK')
        else:
            print('FAIL')

        for value, key in enumerate(keys):
            print(f'Put {key}: {value}')
            st.put(key, value)
        res_vals = []
        for key in keys:
            value = st.get(key)
            print(f'Get {key}: {value}')
            res_vals.append(value)
        if vals == res_vals:
            print('OK')
        else:
            print('FAIL')

        size = len(set(keys))
        print(f'Size: {st.size()}')
        if st.size() == size:
            print('OK')
        else:
            print('FAIL')

        print(f'Size lazy: {st.size_lazy()}')
        if st.size_lazy() == size:
            print('OK')
        else:
            print('FAIL')

        print(f'Contains "she": {st.contains("she")}')
        if st.contains('she') == True:
            print('OK')
        else:
            print('FAIL')

        z = st.get('awesome')
        print(f'Get missing key "awesome": {z}')
        if z == None:
            print('OK')
        else:
            print('FAIL')

        res_keys = st.keys()
        print(f'Keys: {res_keys}')
        if set(res_keys) == set(keys):
            print('OK')
        else:
            print('FAIL')

        l_p = []
        for key in test_keys:
            res = st.longest_prefix_of(key)
            l_p.append(res)
            print(f'Longest prefix of "{key}": {res}')
        if longest_prefix == l_p:
            print('OK')
        else:
            print('FAIL')

        k_w_p = []
        for key in test_keys:
            res = st.keys_with_prefix(key)
            k_w_p.append(res)
            print(f'Keys with prefix "{key}": {res}')
        if keys_with_prefix == k_w_p:
            print('OK')
        else:
            print('FAIL')

        k_t_m = []
        for key in test_keys:
            res = st.keys_that_match(key)
            k_t_m.append(res)
            print(f'Keys that match "{key}": {res}')
        if keys_that_match == k_t_m:
            print('OK')
        else:
            print('FAIL')

        for key in del_keys:
            print(f'Deleting keys: {key}')
            st.delete(key)
        res_keys = st.keys()
        print(f'Keys: {res_keys}')
        if set(res_keys) == set(keys) - set(del_keys):
            print('OK')
        else:
            print('FAIL')

        print(f'Size after removing: {st.size()}')
        if st.size() == size - len(del_keys):
            print('OK')
        else:
            print('FAIL')

        print(f'Size lazy after removing: {st.size_lazy()}')
        if st.size_lazy() == size - len(del_keys):
            print('OK')
        else:
            print('FAIL')

        print(f'Contains "she": {st.contains("she")}')
        if st.contains('she') == False:
            print('OK')
        else:
            print('FAIL')

        print(f'Is empty: {st.is_empty()}')
        if st.is_empty() == False:
            print('OK')
        else:
            print('FAIL')
