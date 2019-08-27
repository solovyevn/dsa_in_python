#!/usr/bin/env python
# _*_ encoding: utf-8 _*_


from collections import deque

"""Implements Symbol Table aka Dictionary."""


class ST():

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


class OrderedST():

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

    def delete_min(self):
        self.delete(self.min())

    def delete_max(self):
        self.delete(self.max())

    def contains(self, key):
        return self.get(key) != None

    def is_empty(self):
        return self.size() == 0

    def size(self, lo=None, hi=None):
        pass

    def keys(self, lo=None, hi=None):
        pass

    def min(self):
        pass

    def max(self):
        pass

    def floor(self, key):
        pass

    def ceiling(self, key):
        pass

    def rank(self, key):
        pass

    def select(self, key):
        pass


class SequentialSearchST(ST):

    class _Node():

        def __init__(self, key, value, next_):
            self.key = key
            self.value = value
            self.next = next_

    def __init__(self):
        self._start = None
        self._size = 0

    def put(self, key, value):
        # O(n)
        # Deletes key if value is None
        if value is None:
            self.delete(key)
            return
        _next = self._start
        while _next:
            if _next.key == key:
                _next.value = value
                return
            _next = _next.next
        self._start = self._Node(key, value, self._start)
        self._size += 1

    def get(self, key):
        # O(n)
        _next = self._start
        while _next:
            if _next.key == key:
                return _next.value
            _next = _next.next
        return None

    def delete(self, key):
        # O(n)
        if self._start is None:
            return
        _next = self._start
        if _next is not None and _next.key == key:
            self._start = _next.next
            self._size -= 1
            return
        while _next:
            if _next.next and _next.next.key == key:
                _next.next = _next.next.next
                self._size -= 1
                return
            _next = _next.next

    def size(self):
        return self._size

    def keys(self):
        def _keys(self):
            _next = self._start
            while _next:
                yield _next.key
                _next = _next.next
        gen = _keys(self)
        return list(gen)


class BinarySearchST(OrderedST):

    def __init__(self):
        self._keys = []
        self._values = []
        self._size = 0

    def put(self, key, value):
        # O(n)
        # Deletes key if value is None
        if value is None:
            self.delete(key)
            return
        rank = self.rank(key)
        # Update
        if rank < self._size and self._keys[rank] == key:
            self._values[rank] = value
        # Insert
        else:
            # Resizing
            self._keys.append(0)
            self._values.append(0)
            # Making place for a new value
            i = self._size - 1
            while i >= rank:
                self._keys[i+1] = self._keys[i]
                self._values[i+1] = self._values[i]
                i -= 1
            # Actual insertion
            self._keys[rank] = key
            self._values[rank] = value
            self._size += 1

    def get(self, key):
        # O(lg(n))
        if self._size == 0:
            return None
        rank = self.rank(key)
        if rank < self._size and self._keys[rank] == key:
            return self._values[rank]
        return None

    def delete(self, key):
        # O(n)
        if self.is_empty():
            return
        rank = self.rank(key)
        if rank < self._size and self._keys[rank] == key:
            i = rank
            while i < self._size - 1:
                self._keys[i] = self._keys[i+1]
                self._values[i] = self._values[i+1]
                i += 1
            self._size -= 1
        return

    def size(self, lo=None, hi=None):
        if lo is None:
            lo = self.min()
        if hi is None:
            hi = self.max()
        if lo >= hi:
            return 0
        if self.contains(hi):
            return self.rank(hi) - self.rank(lo) + 1
        else:
            return self.rank(hi) - self.rank(lo)

    def keys(self, lo=None, hi=None):
        if lo is None:
            lo = 0
        else:
            lo = self.rank(lo)
        if hi is None:
            hi = self._size
        else:
            hi = self.rank(hi) + 1
        return self._keys[lo:hi]

    def min(self):
        return self._keys[0] if self._keys else None

    def max(self):
        return self._keys[self._size-1] if self._keys else None

    def floor(self, key):
        if self.is_empty():
            return None
        rank = self.rank(key)
        if 0 < rank:
            return self._keys[rank-1]
        else:
            return self._keys[rank]

    def ceiling(self, key):
        if self.is_empty():
            return None
        rank = self.rank(key)
        if rank < self._size:
            return self._keys[rank]
        else:
            return self._keys[self._size-1]

    def rank(self, key):
        lo = 0
        hi = self._size - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if key == self._keys[mid]:
                return mid
            if key > self._keys[mid]:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    def select(self, rank):
        if 0 <= rank < self._size:
            return self._keys[rank]
        else:
            return None


class BinarySearchTreeST(OrderedST):

    class _Node():

        def __init__(self, key, value, count=1, left=None, right=None):
            self.key = key
            self.value = value
            self.count = count
            self.left = left
            self.right = right

    def __init__(self):
        self._root = None

    def put(self, key, value):
        # O(n) worst
        # O(lg(n)) average
        # Deletes key if value is None
        if value is None:
            self.delete(key)
            return
        self._root = self._put(key, value, self._root)

    def _put(self, key, value, root):
        if root is None:
            return self._Node(key, value)
        if key < root.key:
            root.left = self._put(key, value, root.left)
        elif key > root.key:
            root.right = self._put(key, value, root.right)
        else:
            root.value = value
        root.count = self._size(root.left) + self._size(root.right) + 1
        return root

    def get(self, key):
        # O(n) worst
        # O(lg(n)) average
        return self._get(key, self._root)

    def _get(self, key, root):
        if root is None:
            return None
        if key < root.key:
            return self._get(key, root.left)
        elif key > root.key:
            return self._get(key, root.right)
        else:
            return root.value

    def delete(self, key):
        # O(n) worst
        # O(lg(n)) average
        self._root = self._delete(key, self._root)

    def _delete(self, key, node):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(key, node.left)
        elif key > node.key:
            node.right = self._delete(key, node.right)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._min(node.right)
                successor.right = self._delete_min(node.right)
                successor.left = node.left
                node = successor
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def delete_min(self):
        self._root = self._delete_min(self._root)

    def _delete_min(self, node):
        if node is None:
            return None
        if node.left is None:
            return node.right
        node.left = self._delete_min(node.left)
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def delete_max(self):
        self._root = self._delete_max(self._root)

    def _delete_max(self, node):
        if node is None:
            return None
        if node.right is None:
            return node.left
        node.right = self._delete_max(node.right)
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def size(self, lo=None, hi=None):
        if lo is None and hi is None:
            return self._size(self._root)
        if lo is None:
            lo = self.min()
        if hi is None:
            hi = self.max()
        return self._count_size(lo, hi, self._root)

    def _count_size(self, lo, hi, node):
        size = 0
        if node is None:
            return size
        if lo <= node.key:
            size += self._count_size(lo, hi, node.left)
        if lo <= node.key <= hi:
            size += 1
        if hi >= node.key:
            size += self._count_size(lo, hi, node.right)
        return size

    def _size(self, root):
        if root is None:
            return 0
        else:
            return root.count

    def keys(self, lo=None, hi=None):
        if lo is None:
            lo = self.min()
        if hi is None:
            hi = self.max()
        q = deque()
        self._keys(lo, hi, q, self._root)
        return list(q)

    def _keys(self, lo, hi, q, node):
        if node is None:
            return
        if lo <= node.key:
            self._keys(lo, hi, q, node.left)
        if lo <= node.key <= hi:
            q.append(node.key)
        if hi >= node.key:
            self._keys(lo, hi, q, node.right)

    def min(self):
        min_node = self._min(self._root)
        if min_node is not None:
            return min_node.key
        return None

    def _min(self, node):
        if node.left is None:
            return node
        else:
            return self._min(node.left)

    def max(self):
        max_node = self._max(self._root)
        if max_node is not None:
            return max_node.key
        return None

    def _max(self, node):
        if node.right is None:
            return node
        else:
            return self._max(node.right)

    def floor(self, key):
        floor_node = self._floor(key, self._root)
        if floor_node is not None:
            return floor_node.key
        return None

    def _floor(self, key, node):
        if node is None:
            return None
        if key == node.key:
            return node
        if key < node.key:
            return self._floor(key, node.left)
        res = self._floor(key, node.right)
        if res is not None:
            return res
        else:
            return node

    def ceiling(self, key):
        ceil_node = self._ceiling(key, self._root)
        if ceil_node is not None:
            return ceil_node.key
        return None

    def _ceiling(self, key, node):
        if node is None:
            return None
        if key == node.key:
            return node
        if key > node.key:
            return self._ceiling(key, node.right)
        res = self._ceiling(key, node.left)
        if res is not None:
            return res
        else:
            return node

    def rank(self, key):
        return self._rank(key, self._root)

    def _rank(self, key, node):
        if node is None:
            return None
        if key < node.key:
            return self._rank(key, node.left)
        elif key > node.key:
            return self._size(node.left) + 1 + self._rank(key, node.right)
        else:
            return self._size(node.left)

    def select(self, rank):
        key_node = self._select(rank, self._root)
        if key_node is not None:
            return key_node.key
        return None

    def _select(self, rank, node):
        if node is None:
            return None
        count = self._size(node.left)
        if rank < count:
            return self._select(rank, node.left)
        elif rank > count:
            return self._select(rank-count-1, node.right)
        else:
            return node


class RedBlackTreeST(BinarySearchST):

    RED = True
    BLACK = False

    class _Node():

        def __init__(self, key, value, count=1, color=RedBlackTreeST.BLACK,
                     left=None, right=None):
            self.key = key
            self.value = value
            self.count = count
            self.color = color
            self.left = left
            self.right = right

    def __init__(self):
        self._root = None

    def _is_red(self, node):
        if node is None:
            return False
        return node.color == self.RED

    def _rotate_left(self, node):
        new_node = node.right
        node.right = new_node.left
        new_node.left = node
        new_node.color = node.color
        node.color = self.RED
        new_node.count = node.count
        node.count = 1 + self._size(node.left) + self._size(node.right)
        return new_node

    def _rotate_right(self, node):
        new_node = node.left
        node.left = new_node.right
        new_node.right = node
        new_node.color = node.color
        node.color = self.RED
        new_node.count = node.count
        node.count = 1 + self._size(node.left) + self._size(node.right)
        return new_node

    def _flip_colors(self, node):
        node.color = self.RED
        node.left.color = self.BLACK
        node.right.color = self.BLACK

    def put(self, key, value):
        # O(n) worst
        # O(lg(n)) average
        # Deletes key if value is None
        if value is None:
            self.delete(key)
            return
        self._root = self._put(key, value, self._root)
        self._root.color = self.BLACK

    def _put(self, key, value, root):
        if root is None:
            return self._Node(key, value, color=self.RED)
        if key < root.key:
            root.left = self._put(key, value, root.left)
        elif key > root.key:
            root.right = self._put(key, value, root.right)
        else:
            root.value = value
        if self._is_red(root.right) and not self._is_red(root.left):
            root = self._rotate_left(root)
        if root.left and self._is_red(root.left) and self._is_red(root.left.left):
            root = self._rotate_right(root)
        if self._is_red(root.right) and self._is_red(root.left):
            self._flip_colors(root)
        root.count = self._size(root.left) + self._size(root.right) + 1
        return root

    def get(self, key):
        # O(n) worst
        # O(lg(n)) average
        return self._get(key, self._root)

    def _get(self, key, root):
        if root is None:
            return None
        if key < root.key:
            return self._get(key, root.left)
        elif key > root.key:
            return self._get(key, root.right)
        else:
            return root.value

    def delete(self, key):
        # O(n) worst
        # O(lg(n)) average
        self._root = self._delete(key, self._root)

    def _delete(self, key, node):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(key, node.left)
        elif key > node.key:
            node.right = self._delete(key, node.right)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._min(node.right)
                successor.right = self._delete_min(node.right)
                successor.left = node.left
                node = successor
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def delete_min(self):
        if self._root is None:
            return
        if not self._is_red(self._root.left):
            if self._root.right:
                self._root.left.color = self.RED
            if self._root.left:
                self._root.right.color = self.RED
        if self._root.left and not self._is_red(self._root.left.left):
            self._root = self._rotate_left(self._root)
            if self._root.left and self._root.left.left:
                self._root.left.left.color = self.RED
        self._root = self._delete_min(self._root)

    def _delete_min(self, node):
        if node is None:
            return None
        if node.left is None:
            return node.right
        if not self._is_red(node.left.left):
            if self._is_red(node.right.left):
                node = self._rotate_left(node)
                if node.left and node.left.left:
                    node.left.left.color = self.RED
        node.left = self._delete_min(node.left)
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def delete_max(self):
        if self._root is None:
            return
        self._root = self._delete_max(self._root)

    def _delete_max(self, node):
        if node is None:
            return None
        if node.right is None:
            return node.left
        node.right = self._delete_max(node.right)
        node.count = self._size(node.left) + self._size(node.right) + 1
        return node

    def size(self, lo=None, hi=None):
        if lo is None and hi is None:
            return self._size(self._root)
        if lo is None:
            lo = self.min()
        if hi is None:
            hi = self.max()
        return self._count_size(lo, hi, self._root)

    def _count_size(self, lo, hi, node):
        size = 0
        if node is None:
            return size
        if lo <= node.key:
            size += self._count_size(lo, hi, node.left)
        if lo <= node.key <= hi:
            size += 1
        if hi >= node.key:
            size += self._count_size(lo, hi, node.right)
        return size

    def _size(self, root):
        if root is None:
            return 0
        else:
            return root.count

    def keys(self, lo=None, hi=None):
        if lo is None:
            lo = self.min()
        if hi is None:
            hi = self.max()
        q = deque()
        self._keys(lo, hi, q, self._root)
        return list(q)

    def _keys(self, lo, hi, q, node):
        if node is None:
            return
        if lo <= node.key:
            self._keys(lo, hi, q, node.left)
        if lo <= node.key <= hi:
            q.append(node.key)
        if hi >= node.key:
            self._keys(lo, hi, q, node.right)

    def min(self):
        min_node = self._min(self._root)
        if min_node is not None:
            return min_node.key
        return None

    def _min(self, node):
        if node.left is None:
            return node
        else:
            return self._min(node.left)

    def max(self):
        max_node = self._max(self._root)
        if max_node is not None:
            return max_node.key
        return None

    def _max(self, node):
        if node.right is None:
            return node
        else:
            return self._max(node.right)

    def floor(self, key):
        floor_node = self._floor(key, self._root)
        if floor_node is not None:
            return floor_node.key
        return None

    def _floor(self, key, node):
        if node is None:
            return None
        if key == node.key:
            return node
        if key < node.key:
            return self._floor(key, node.left)
        res = self._floor(key, node.right)
        if res is not None:
            return res
        else:
            return node

    def ceiling(self, key):
        ceil_node = self._ceiling(key, self._root)
        if ceil_node is not None:
            return ceil_node.key
        return None

    def _ceiling(self, key, node):
        if node is None:
            return None
        if key == node.key:
            return node
        if key > node.key:
            return self._ceiling(key, node.right)
        res = self._ceiling(key, node.left)
        if res is not None:
            return res
        else:
            return node

    def rank(self, key):
        return self._rank(key, self._root)

    def _rank(self, key, node):
        if node is None:
            return None
        if key < node.key:
            return self._rank(key, node.left)
        elif key > node.key:
            return self._size(node.left) + 1 + self._rank(key, node.right)
        else:
            return self._size(node.left)

    def select(self, rank):
        key_node = self._select(rank, self._root)
        if key_node is not None:
            return key_node.key
        return None

    def _select(self, rank, node):
        if node is None:
            return None
        count = self._size(node.left)
        if rank < count:
            return self._select(rank, node.left)
        elif rank > count:
            return self._select(rank-count-1, node.right)
        else:
            return node



if __name__ == "__main__":
    keys = ['S', 'E', 'A', 'R', 'C', 'H', 'E', 'X', 'A', 'M', 'P', 'L', 'E']
    vals = [0, 12, 8, 3, 4, 5, 12, 7, 8, 9, 10, 11, 12]
    ordered_keys = ['A', 'C', 'E', 'H', 'L', 'M', 'P', 'R', 'S', 'X']
    keys_e_p = ['E', 'H', 'L', 'M', 'P']
    m_rank_5 = 5
    select_3_h = 'H'
    ceil_o_p = 'P'
    floor_g_e = 'E'
    del_keys = ['A', 'X', 'S']
    for _ST in [SequentialSearchST, BinarySearchST, BinarySearchTreeST, RedBlackTreeST]:

        print(f'\n-= {_ST.__name__} =-\n')
        st = _ST()
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

        print(f'Size: {st.size()}')
        if st.size() == len(ordered_keys):
            print('OK')
        else:
            print('FAIL')

        z = st.get('Z')
        print(f'Get missing key "Z": {z}')
        if z == None:
            print('OK')
        else:
            print('FAIL')

        res_keys = st.keys()

        if isinstance(st, OrderedST):

            print(f'Keys: {res_keys}')
            if res_keys == ordered_keys:
                print('OK')
            else:
                print('FAIL')

            max_key = st.max()
            print(f'Max: {max_key}')
            if max_key == ordered_keys[-1]:
                print('OK')
            else:
                print('FAIL')

            min_key = st.min()
            print(f'Min: {min_key}')
            if min_key == ordered_keys[0]:
                print('OK')
            else:
                print('FAIL')

            rank_m = st.rank('M')
            print(f'Rank of "M": {rank_m}')
            if rank_m == m_rank_5:
                print('OK')
            else:
                print('FAIL')

            select_3 = st.select(3)
            print(f'Select "3": {select_3}')
            if select_3 == select_3_h:
                print('OK')
            else:
                print('FAIL')

            ceil_o = st.ceiling('O')
            print(f'Ceiling "O": {ceil_o}')
            if ceil_o == ceil_o_p:
                print('OK')
            else:
                print('FAIL')

            floor_g = st.floor('G')
            print(f'Floor "G": {floor_g}')
            if floor_g == floor_g_e:
                print('OK')
            else:
                print('FAIL')

            keys_slice = st.keys('E', 'P')
            print(f'Keys "E" - "P": {keys_slice}')
            if keys_slice == keys_e_p:
                print('OK')
            else:
                print('FAIL')

            size = st.size('E', 'P')
            print(f'Size "E" - "P": {size}')
            if size == len(keys_e_p):
                print('OK')
            else:
                print('FAIL')

            for key in del_keys:
                print(f'Deleting keys: {key}')
                st.delete(key)
            res_keys = st.keys()
            print(f'Keys: {res_keys}')
            if res_keys == ordered_keys[1:8]:
                print('OK')
            else:
                print('FAIL')

        else:

            print(f'Keys: {res_keys}')
            if set(res_keys) == set(ordered_keys):
                print('OK')
            else:
                print('FAIL')

            for key in del_keys:
                print(f'Deleting keys: {key}')
                st.delete(key)
            res_keys = st.keys()
            print(f'Keys: {res_keys}')
            if set(res_keys) == set(ordered_keys[1:8]):
                print('OK')
            else:
                print('FAIL')

        print(f'Size after removing: {st.size()}')
        if st.size() == len(ordered_keys[1:8]):
            print('OK')
        else:
            print('FAIL')
