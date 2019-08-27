#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Implements binary search."""


def bin_search(needle, haystack):
    lo = 0
    hi = len(haystack) - 1
    while hi >= lo:
        mid = (hi+lo)//2
        print(mid)
        if needle == haystack[mid]:
            return mid
        elif needle < haystack[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


def bin_search_first(needle, haystack):
    lo = 0
    hi = len(haystack) - 1
    while lo <= hi:
        mid = (hi+lo)//2
        print(mid)
        if (haystack[mid] == needle and (mid == 0 or haystack[mid-1] < needle)):
            return mid
        elif needle > haystack[mid]:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def bin_search_first_alt(needle, haystack):
    lo = 0
    hi = len(haystack) - 1
    res = -1
    while lo <= hi:
        mid = (hi+lo)//2
        print(mid)
        if (haystack[mid] == needle):
            res = mid
            hi = mid - 1
        elif needle > haystack[mid]:
            lo = mid + 1
        else:
            hi = mid - 1
    return res


def bin_search_last(needle, haystack):
    lo = 0
    hi = len(haystack) - 1
    while lo <= hi:
        mid = (hi+lo)//2
        print(mid)
        if (haystack[mid] == needle and (mid == len(haystack) - 1 or haystack[mid+1] > needle)):
            return mid
        elif needle < haystack[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


def bin_search_last_alt(needle, haystack):
    lo = 0
    hi = len(haystack) - 1
    res = -1
    while lo <= hi:
        mid = (hi+lo)//2
        print(mid)
        if (haystack[mid] == needle):
            res = mid
            lo = mid + 1
        elif needle < haystack[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return res


if __name__ == "__main__":
    cases = []
    not_ok = 0
    for i in range(0, 6):
        arr = list(range(0, i))
        for j in range(0, i):
            cases.append((arr, j, j))
    cases.append((list(range(0, 10)), 123, -1))
    cases.append((list(range(0, 10)), -10, -1))
    cases.append(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], -5, 0))
    cases.append(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 0, 5))
    cases.append(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 5, 10))
    cases.append(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], -2, 3))
    cases.append(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 2, 7))
    first_cases = []
    first_cases.append(([1, 2, 3, 3, 3, 4, 5], 3, 2))
    first_cases.append(([1, 1, 1, 3, 3, 4, 5], 1, 0))
    first_cases.append(([1, 1, 1, 1, 1, ], 1, 0))
    first_cases.append(([1, 1, 1, ], 1, 0))
    first_cases.append(([1, 1, ], 1, 0))
    first_cases.append(([1, ], 1, 0))
    first_cases.append(([1, 1, 3, 3, 4, 5, 5], 5, 5))
    first_cases.append(([1, 1, 3, 3, 4, 5, 5], 1, 0))
    first_cases.append(([1, 1, 3, 3, 4, 5, 5], 3, 2))
    first_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 5, 12))
    first_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 3, 4))
    first_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 4, 8))
    first_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 1, 0))
    last_cases = []
    last_cases.append(([1, 2, 3, 3, 3, 4, 5], 3, 4))
    last_cases.append(([1, 1, 1, 3, 3, 4, 5], 1, 2))
    last_cases.append(([1, 1, 1, 1, 1, ], 1, 4))
    last_cases.append(([1, 1, 1, ], 1, 2))
    last_cases.append(([1, 1, ], 1, 1))
    last_cases.append(([1, ], 1, 0))
    last_cases.append(([1, 1, 3, 3, 4, 5, 5], 5, 6))
    last_cases.append(([1, 1, 3, 3, 4, 5, 5], 1, 1))
    last_cases.append(([1, 1, 3, 3, 4, 5, 5], 3, 3))
    last_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 5, 15))
    last_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 3, 7))
    last_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 4, 11))
    last_cases.append(([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], 1, 3))
    print('-== BinSearch =-')
    for case in cases:
        print(case)
        res = bin_search(case[1], case[0])
        if res == case[2]:
            print(f'OK: {res}')
        else:
            not_ok += 1
            print(f'FAIL: {res}')
    print('-== BinSearch First =-')
    for case in cases+first_cases:
        print(case)
        res = bin_search_first(case[1], case[0])
        if res == case[2]:
            print(f'OK: {res}')
        else:
            not_ok += 1
            print(f'FAIL: {res}')
    print('-== BinSearch Last =-')
    for case in cases+last_cases:
        print(case)
        res = bin_search_last(case[1], case[0])
        if res == case[2]:
            print(f'OK: {res}')
        else:
            not_ok += 1
            print(f'FAIL: {res}')
    print('-== BinSearch First Alt =-')
    for case in cases+first_cases:
        print(case)
        res = bin_search_first_alt(case[1], case[0])
        if res == case[2]:
            print(f'OK: {res}')
        else:
            not_ok += 1
            print(f'FAIL: {res}')
    print('-== BinSearch Last Alt =-')
    for case in cases+last_cases:
        print(case)
        res = bin_search_last_alt(case[1], case[0])
        if res == case[2]:
            print(f'OK: {res}')
        else:
            not_ok += 1
            print(f'FAIL: {res}')
    print(f'NOT OK: {not_ok}')

