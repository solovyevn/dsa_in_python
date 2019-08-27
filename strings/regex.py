#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Regular expression matching algorithms."""


class Graph():

    def __init__(self, v, conns=None):
        if conns is None:
            conns = []
        self._v = v
        self._e = 0
        self._adj_list = [[] for _ in range(v)]
        for conn in conns:
            self.add_edge(*conn)

    def V(self):
        return self._v

    def E(self):
        return self._e

    def add_edge(self, v, w):
        self._adj_list[v].append(w)
        self._adj_list[w].append(v)
        self._e += 1

    def adj(self, v):
        return self._adj_list[v]

    def degree(self, v):
        deg = 0
        for w in self.adj(v):
            deg += 1
        return deg

    def max_degree(self):
        deg = 0
        for v in range(0, self.V()):
            d = self.degree(v)
            if d > deg:
                deg = d
        return deg

    def avg_degree(self):
        return 2 * self.E() // self.V()

    def self_loops_count(self):
        count = 0
        for v in range(0, self.V()):
            for w in self.adj(v):
                if v == w:
                    count += 1
        # Each edge counted twice, because w-v and v-w each grant +1 in count
        return count // 2;

    def __str__(self):
        s = f'{self.V()} vertices, {self.E()} edges\n'
        for v in range(0, self.V()):
            s = f'{s}{v}: '
            for w in self.adj(v):
                s = f'{s}{w} '
            s = f'{s}\n'
        return s


class Digraph(Graph):

    def add_edge(self, v, w):
        self._adj_list[v].append(w)
        self._e += 1

    def reverse(self):
        r = Digraph(self._v)
        for v in range(self._v):
            for w in self._adj_list[v]:
                r.add_edge(w, v)
        return r

    def out_degree(self, v):
        deg = 0
        for w in self.adj(v):
            deg += 1
        return deg

    def max_out_degree(self):
        deg = 0
        for v in range(0, self.V()):
            d = self.degree(v)
            if d > deg:
                deg = d
        return deg

    def avg_out_degree(self):
        return 2 * self.E() // self.V()

    def self_loops_count(self):
        count = 0
        for v in range(0, self.V()):
            for w in self.adj(v):
                if v == w:
                    count += 1
        # No doubling here unlike in undirected graph
        # Each edge counted only once, because adjacency list contains v->w only
        return count;


class Search():

    def __init__(self, g, s):
        self._g = g
        self._s = s
        self._marked = [False, ] * g.V()
        self._count = 0

    def _mark(self, v):
        self._marked[v] = True

    def marked(self, v):
        return self._marked[v]

    def count(self):
        return self._count

    def __str__(self):
        return str([f'{i}: {m}' for i, m in enumerate(self._marked)])


class DFS(Search):

    def __init__(self, g, s):
        super().__init__(g, s)
        self.dfs(g, s)

    def dfs(self, g, v):
        self._mark(v)
        self._count += 1
        for w in g.adj(v):
            if not self.marked(w):
                self.dfs(g, w)


class DirectedDFS(DFS):

    def __init__(self, g, s):
        # Note here DFS in super, not DirectedDFS,
        # so it's calling Search's initializer
        super(DFS, self).__init__(g, s)
        if not isinstance(s, list):
            self.dfs(g, s)
        else:
            for _s in s:
                if not self.marked(_s):
                    self.dfs(g, _s)


class Regex():
    # NFA

    def __init__(self, pattern):
        self._create_nfa(pattern)

    def _create_nfa(self, pattern):
        self._m = len(pattern)
        self._re = pattern  # Match transitions
        self._g = Digraph(self._m+1)  # Epsilon transitions
        ops = []
        for i in range(self._m):
            lp = i
            if self._re[i] == '(' or self._re[i] == '|':
                ops.append(i)
            elif self._re[i] == ')':
                or_i = ops.pop()
                if self._re[or_i] == '|':
                    lp = ops.pop()
                    self._g.add_edge(lp, or_i+1)
                    self._g.add_edge(or_i, i)
                else:
                    lp = or_i
            if i < self._m - 1 and self._re[i+1] == '*':
                self._g.add_edge(lp, i+1)
                self._g.add_edge(i+1, lp)
            if self._re[i] == '(' or self._re[i] == '*' or self._re[i] == ')':
                self._g.add_edge(i, i+1)

    def recognizes(self, txt):
        N = len(txt)
        pc = set()
        dfs = DirectedDFS(self._g, 0)
        for v in range(self._g.V()):
            if dfs.marked(v):
                pc.add(v)
        print(f'DFS 0: {dfs}')

        for i in range(N):
            match = set()
            for v in pc:
                if v < self._m:
                    if self._re[v] == txt[i] or self._re[v] == '.':
                        match.add(v+1)
            pc = set()
            dfs = DirectedDFS(self._g, list(match))
            for v in range(self._g.V()):
                if dfs.marked(v):
                    pc.add(v)
            print(f'DFS {i}: {dfs}')

        for v in pc:
            if v == self._m:
                return True
        return False


if __name__ == "__main__":
    sample_str = 'ABCCBD'
    patterns = ['(A*B|AC)D', 'SOME', ]
    expected_res = [True, False, ]
    for R in [Regex, ]:
        print(f'\n-= {R.__name__} =-\n')
        for (p, exp_res) in zip(patterns, expected_res):
            r = R(p)
            print(f'Sample string: {sample_str}')
            print(f'Pattern: {p}')
            res = r.recognizes(sample_str)
            print(f'Match result: {res}')
            if res == exp_res:
                print('OK')
            else:
                print('FAIL')
            print()
