#!/usr/bin/env python
# _*_ encoding: utf-8 _*_


"""Implements sorting algorithms."""


import random
from functools import wraps


def less(a, b):
    return a < b


def swap(items, i, j):
    # O(1) space for `t`
    # items[i], items[j] = items[j], items[i]
    t = items[i]
    items[i] = items[j]
    items[j] = t


def bubble_sort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps
    Average-case:
        O(n^2) comparisons
        O(n^2) swaps
    Best-case:
        O(n) comparisons
        O(1) swaps, no swaps actually
    O(1) auxilliary space (for swaps)
    Adaptive (less swaps if the array or part of it is already sorted)
    Stable (items with same value are kept in original relative order)
    Non-Online (can't process items added to array as the sort goes on)
    """
    n = len(a)
    has_swapped = True
    while has_swapped:
        has_swapped = False
        for i in range(1, n):
            if less(a[i], a[i-1]):
                swap(a, i, i-1)
                has_swapped = True
    return a


def optimized_bubble_sort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps
    Average-case:
        O(n^2) comparisons
        O(n^2) swaps
    Best-case:
        O(n) comparisons
        O(1) swaps, no swaps actually
    O(1) auxilliary space
    Adaptive
    Stable
    Non-Online
    """
    new_n = len(a)
    while new_n != 0:
        n = new_n
        new_n = 0
        for i in range(1, n):
            if less(a[i], a[i-1]):
                swap(a, i, i-1)
                # If the new_n won't change after this iteration it means, that
                # all items above new_n = i are already sorted, so no need to
                # consider those on next iteration
                new_n = i
    return a


def selection_sort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n) swaps
    Average-case:
        O(n^2) comparisons
        O(n) swaps
    Best-case:
        O(n^2) comparisons
        O(n) swaps
    O(1) auxilliary space
    Non-Adaptive
    Non-Stable, can be made stable with trade-offs
    Non-Online
    """
    n = len(a)
    for i in range(n):
        min_i = i
        for j in range(i+1, n):
            if less(a[j], a[min_i]):
                min_i = j
        swap(a, i, min_i)
    return a


def insertion_sort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps
    Average-case:
        O(n^2) comparisons
        O(n^2) swaps
        Actually O(n^k), where k is length between actual item position
            and its sorted position
    Best-case:
        O(n) comparisons
        O(1) swaps, no swaps actually
    O(1) auxiliary space
    Adaptive
    Stable
    Online
    """
    n = len(a)
    for i in range(1, n):
        j = i
        while j > 0 and less(a[j], a[j-1]):
            swap(a, j-1, j)
            j -= 1
    return a


def optimized_insertion_sort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps, half the swaps of non-optimized version
    Average-case:
        O(n^2) comparisons
        O(n^2) swaps, half the swaps of non-optimized version
        Actually O(n^k), where k is length between actual item position
            and its sorted position
    Best-case:
        O(n) comparisons
        O(1) swaps, no swaps actually
    O(1) auxiliary space
    Adaptive
    Stable
    Online
    """
    n = len(a)
    for i in range(1, n):
        # Here `swap` is put inline spread over the outer loop
        a_i = a[i]
        j = i
        while j > 0 and less(a_i, a[j-1]):
            a[j] = a[j-1]
            j -= 1
        a[j] = a_i
    return a


def shell_sort(a):
    """
    Worst-case:
        O(n^2) comparisons, worst known sequence
        O(n^2) swaps, worst known sequence
        O(n^(3/2)) comparisons and swaps for sequence used here
    Average-case:
        Depends on the gap sequence (`h`)
    Best-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    O(1) auxiliary space
    Adaptive, not as good as insertion sort
    Non-Stable
    Online
    """
    n = len(a)
    # Using gap sequence ((3^k)-1)/2, not greater than n/3
    # 1, 4, 13, 40, 121, 364, 1093
    h = 1
    while h < n//3:
        h = 3*h + 1
    while h >= 1:
        for i in range(h, n):
            j = i
            while j >= h and less(a[j], a[j-h]):
                swap(a, j, j-h)
                j -= h
            i += i
        h //= 3
    return a


def optimized_shell_sort(a):
    """
    Worst-case:
        O(n^2) comparisons, worst known sequence
        O(n^2) swaps, worst known sequence, half the swaps of non-optimized version
        O(n^(3/2)) comparisons and swaps for sequence used here
    Average-case:
        Depends on the gap sequence (`h`), half the swaps of non-optimized version
    Best-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    O(1) auxiliary space
    Adaptive, not as good as insertion sort
    Non-Stable
    Online
    """
    n = len(a)
    # Using gap sequence ((3^k)-1)/2, not greater than n/3
    # 1, 4, 13, 40, 121, 364, 1093
    h = 1
    while h < n//3:
        h = 3*h + 1
    while h >= 1:
        for i in range(h, n):
            # Here `swap` is put inline spread over the outer loop
            a_i = a[i]
            j = i
            while j >= h and less(a_i, a[j-h]):
                a[j] = a[j-h]
                j -= h
            a[j] = a_i
            i += i
        h //= 3
    return a


def _merge(a, lo, mid, hi, aux):
    i = lo
    j = mid+1
    # aux = a.copy()
    for k in range(lo, hi+1):
        aux[k] = a[k]
    for k in range(lo, hi+1):
        if i > mid:
            a[k] = aux[j]
            j += 1
        elif j > hi:
            a[k] = aux[i]
            i += 1
        # Note here we compare j to i, not i to j, this leads to stable sorting
        elif less(aux[j], aux[i]):
            a[k] = aux[j]
            j += 1
        else:
            a[k] = aux[i]
            i += 1
    return a


def top_down_merge_sort(a, lo=None, hi=None, aux=None):
    """
    Worst-case:
        O(nlogn) comparisons
    Average-case:
        O(nlogn) comparisons
    Best-case:
        O(nlogn) comparisons
    O(n) auxiliary space
    Non-Adaptive
    Stable
    Non-Online
    """
    # Will pass the auxilliary array over to prevent multiple allocations
    if aux is None:
        # Creating an array of zeros to use indexing instead of append
        aux = [0, ] * len(a)
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a) - 1

    if hi <= lo:
        return a
    mid = lo + (hi - lo) // 2
    top_down_merge_sort(a, lo, mid, aux)
    top_down_merge_sort(a, mid+1, hi, aux)
    _merge(a, lo, mid, hi, aux)
    return a


def _insertion_sort(a, lo=None, hi=None):
    """Same as optimized insertion sort, but adding indexes to use other algos."""
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a) - 1
    for i in range(lo+1, hi+1):
        # Here `swap` is put inline spread over the outer loop
        a_i = a[i]
        j = i
        while j > lo and less(a_i, a[j-1]):
            a[j] = a[j-1]
            j -= 1
        a[j] = a_i
    return a


def optimized_top_down_merge_sort(a, lo=None, hi=None, aux=None):
    """
    Worst-case:
        O(nlogn) comparisons
    Average-case:
        O(nlogn) comparisons
    Best-case:
        O(n) comparisons
    O(n) auxiliary space
    Non-Adaptive
    Stable
    Non-Online
    """
    # Will pass the auxilliary array over to prevent multiple allocations
    if aux is None:
        # Creating an array of zeros to use indexing instead of append
        aux = [0, ] * len(a)
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a) - 1

    if hi <= lo:
        return a

    # Can use simple sort for small arrays to speed things up
    if hi-lo < 5:  # Will consider less than 5 as small for this example
        _insertion_sort(a, lo, hi)
        return a

    mid = lo + (hi - lo) // 2
    optimized_top_down_merge_sort(a, lo, mid, aux)
    optimized_top_down_merge_sort(a, mid+1, hi, aux)

    # No need to merge if the subarrays are already in order
    if a[mid] <= a[mid+1]:
        return a

    _merge(a, lo, mid, hi, aux)
    return a


def bottom_up_merge_sort(a, lo=None, hi=None, aux=None):
    """
    Worst-case:
        O(nlogn) comparisons
    Average-case:
        O(nlogn) comparisons
    Best-case:
        O(nlogn) comparisons
    O(n) auxiliary space
    Non-Adaptive
    Stable
    Non-Online
    """
    # Will pass the auxilliary array over to prevent multiple allocations
    if aux is None:
        # Creating an array of zeros to use indexing instead of append
        aux = [0, ] * len(a)
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a) - 1

    n = len(a)
    sz = 1
    while sz < n:
        lo = 0
        while lo < n-sz:
            _merge(a, lo, lo+sz-1, min(lo+sz+sz-1, n-1), aux)
            lo = lo + sz + sz
        sz = sz + sz
    return a


def two_way_quicksort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps
    Average-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    Best-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    O(logn) auxiliary space
    Non-Adaptive
    Non-Stable
    Non-Online
    """
    # Prevent possible bad inputs to allow for average case performance.
    # Other method is to choose random element in `partition` instead of `lo`.
    random.shuffle(a)
    return _two_way_quicksort(a, 0, len(a)-1)


def _two_way_partition(a, lo, hi):
    p = a[lo]
    i = lo + 1
    j = hi
    while True:
        while less(a[i], p):
            i += 1
            if i >= hi:
                break
        while less(p, a[j]):
            j -= 1
            if j <= lo:
                break
        if i >= j:
            break
        swap(a, i, j)
        i += 1
        j -= 1
    swap(a, lo, j)
    return j


def _two_way_quicksort(a, lo, hi):
    # Possible imporvements:
    # - Use insertion sort for small subarrays, see optimized merge sort.
    # - Use median-of-three partitioning.
    # - Do not partition subarrays of duplicate values, see three way quicksort.
    if lo >= hi:
        return a
    j = _two_way_partition(a, lo, hi)
    _two_way_quicksort(a, lo, j-1)
    _two_way_quicksort(a, j+1, hi)
    return a


def three_way_quicksort(a):
    """
    Worst-case:
        O(n^2) comparisons
        O(n^2) swaps
    Average-case:
        O(nlogn) comparisons
        O(nlogn) swaps
        Actually O(nH), where H is the Shannon entropy from
            frequences of key values, for distinct keys H = logn
    Best-case:
        O(nlogn) comparisons
        O(nlogn) swaps
        Best case is closer to O(n) for arrays
            with large number of duplicate keys
    O(logn) auxiliary space
    Adaptive
    Non-Stable
    Non-Online
    """
    # Prevent possible bad inputs to allow for average case performance.
    # Other method is to choose random element in `partition` instead of `lo`.
    random.shuffle(a)
    return _three_way_quicksort(a, 0, len(a)-1)


def _three_way_partition(a, lo, hi):
    p = a[lo]
    i = lo + 1
    lt = lo
    gt = hi
    while i <= gt:
        if a[i] == p:
            i += 1
            continue
        if less(a[i], p):
            swap(a, i, lt)
            lt += 1
            i += 1
        else:
            swap(a, i, gt)
            gt -= 1
    return (lt, gt)



def _three_way_quicksort(a, lo, hi):
    if lo >= hi:
        return a
    lt, gt = _three_way_partition(a, lo, hi)
    _three_way_quicksort(a, lo, lt-1)
    _three_way_quicksort(a, gt+1, hi)
    return a


def _sink(a, k, n):
    while 2*k + 1 < n:
        j = 2*k + 1 # j is left child
        if j+1 < n and less(a[j], a[j+1]):
            j += 1  # j is right child
        if not less(a[k], a[j]):
            break
        swap(a, k, j)
        k = j


def heap_sort(a):
    """
    Worst-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    Average-case:
        O(nlogn) comparisons
        O(nlogn) swaps
    Best-case:
        O(nlogn) comparisons
        O(nlogn) swaps
        O(n) for duplicate keys
    O(1) auxiliary space
    Non-Adaptive
    Non-Stable
    Non-Online
    """
    n = len(a)
    # O(n)
    for i in range((n-1)//2, -1, -1):
        _sink(a, i, n)
    # O(nlogn)
    for i in range(n-1, -1, -1):
        swap(a, 0, i)  # i is 'inclusive' in this action
        _sink(a, 0, i)  # i is 'exclusive' in this action, i.e. already final
    return a


def counting_sort(a):
    """
    Worst-case:
        O(n+k), where k is the greatest value in initial array
    Average-case:
        O(n+k)
    Best-case:
        O(n+k)
    O(n+k) auxiliary space
    Non-Adaptive
    Stable
    Non-Online
    """
    if not a:
        return a
    n = len(a)
    k = max(a) + 1  # `+ 1` for 0
    res = [0, ] * n
    # Count occurences of each key
    counts = [0, ] * k
    for i in range(n):
        counts[a[i]] += 1
    # Calculate position for each key in result
    i = 0
    for j in range(k):
        count = counts[j]
        counts[j] = i
        i += count
    # Place each key from original array in its final position
    for i in range(n):
        res[counts[a[i]]] = a[i]
        counts[a[i]] += 1
    return res


def bucket_sort(a):
    """
    Worst-case:
        O(n^2)
    Average-case:
        O(n+k), where k is the number of buckets
    Best-case:
        O(n+k)
    O(n+k) auxiliary space
    Non-Adaptive
    Stable, depends on the choice of inner sort algorithm for values in buckets
    Non-Online
    """
    n = len(a)
    k = 10  # Number of buckets
    # k = n
    buckets = [[] for _ in range(k)]
    # Scatter values over buckets
    for i in range(n):
        index = int(a[i] * k / (max(a) + 1))
        buckets[index].append(a[i])
    # Sort values in each bucket
    for i in range(k):
        buckets[i] = insertion_sort(buckets[i])
    # Combine values into resulting array
    h = 0
    for i in range(k):
        b = buckets[i]
        for j in range(len(b)):
            a[h] = b[j]
            h += 1
    return a


def _counting_sort(a, exp, radix):
    """
    Same as `counting_sort`, but with parameters to use as radix sort subroutine.
    """
    if not a:
        return a
    n = len(a)
    k = radix
    res = [0, ] * n
    # Count occurences of each key
    counts = [0, ] * k
    for i in range(n):
        key = int((a[i] // radix**exp) % radix)
        counts[key] += 1
    # Calculate position for each key in result
    i = 0
    for j in range(k):
        count = counts[j]
        counts[j] = i
        i += count
    # Place each key from original array in its final position
    for i in range(n):
        key = int((a[i] // radix**exp) % radix)
        res[counts[key]] = a[i]
        counts[key] += 1
    return res


def radix_sort(a, radix=10):
    """
    Worst-case:
        O(w*n), where w is the number of digits in the greatest value
        Actually w*(n+r), but usually r is a small constant, e.g. 2, 10, 16
    Average-case:
        O(w*n)
    Best-case:
        O(n), in case w is 1
    O(n+r) auxiliary space, where r is the radix
    Non-Adaptive
    Stable
    Non-Online
    """
    if not a:
        return a
    max_a = max(a)
    exp = 0
    while radix**exp <= max_a:
        a = _counting_sort(a, exp, radix)

        # buckets = [[] for _ in range(r)]
        # for j in range(n):
        #     index = int((a[j] // (r**i)) % r)
        #     print(a[j], ' ', r**i, ' ', i, ' ', index)
        #     buckets[index].append(a[j])

        # print(buckets)
        # h = 0
        # for k in range(r):
        #     b = buckets[k]
        #     for j in range(len(b)):
        #         a[h] = b[j]
        #         h += 1

        exp += 1
    return a


def _int_dec(func):
    """Wrapper for local testing of char sequence with int sorts."""
    @wraps(func)
    def wrapper(a):
        if a and isinstance(a[0], str):
            return [chr(c) for c in func([ord(c) for c in a])]
        else:
            return func(a)
    return wrapper


if __name__ == "__main__":

    sequences = [
        [],
        [1, ],
        [0, ],
        [0, 0],
        [0, 0, 0],
        [100, 10],
        [3, 1, 2],
        ['S', 'O', 'R', 'T', 'E', 'X', 'A', 'M', 'P', 'L', 'E'],
        [111, 54, 102, 98, 87, 75, 87, 63, 54,
         42, 33, 27, 15, 16, 28, 98, 5, 7, 2, 1]
    ]

    expected_sequences = [
        [],
        [1, ],
        [0, ],
        [0, 0],
        [0, 0, 0],
        [10, 100],
        [1, 2, 3],
        ['A', 'E', 'E', 'L', 'M', 'O', 'P', 'R', 'S', 'T', 'X'],
        [1, 2, 5, 7, 15, 16, 27, 28, 33, 42, 54,
         54, 63, 75, 87, 87, 98, 98, 102, 111]
    ]

    sorts = [bubble_sort, optimized_bubble_sort,
             selection_sort,
             insertion_sort, optimized_insertion_sort,
             shell_sort, optimized_shell_sort,
             top_down_merge_sort, optimized_top_down_merge_sort,
             bottom_up_merge_sort,
             two_way_quicksort, three_way_quicksort,
             heap_sort,
             _int_dec(counting_sort),
             _int_dec(bucket_sort),
             _int_dec(radix_sort)]

    for sort in sorts:
        print(f'-== {sort.__name__} ==-')
        for sequence, expected_sequence in zip(sequences, expected_sequences):
            sequence = sequence.copy()
            print('Initial:', ' '.join([str(c) for c in sequence]))
            result = sort(sequence)
            print(' Result:', ' '.join([str(c) for c in result]))
            if result == expected_sequence:
                print('OK\n')
            else:
                print('FAIL\n')
