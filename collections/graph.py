#!/usr/bin/env python
# _*_ encoding: utf-8 _*_


"""Implements Graphs."""


from collections import deque
import heapq
import math


class UF():
    # Enhanced version of `UFWeightedQuickUnion`
    # `find` reduces the height of the trees. This is additional work, but for
    # future `find` calls and `union` calls it decreases the time to find
    # the root.

    def __init__(self, n):
        self._count = n
        # An array of set identifiers
        self._set = [p for p in range(n)]
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

    def connected(self, p, q):
        """Checks whether both p and q are in the same set."""
        # Complexity is the same as `find` in specific implementation, see below
        return self.find(p) == self.find(q)

    def count(self):
        """Returns number of sets."""
        # O(1), obviously
        return self._count


class SympolGraph():

    def __init__(self, conns):
        self._st = {}
        self._keys = []
        new_conns = []
        for v, w in conns:
            if v not in self._st:
                self._st[v] = len(self._st)
                self._keys.append(v)
            if w not in self._st:
                self._st[w] = len(self._st)
                self._keys.append(w)
            new_conns.append((self._st[v], self._st[w]))
        self._g = Graph(len(self._st), new_conns)

    def contains(self, key):
        return key in self._st

    def index(self, key):
        return self._st[key]

    def name(self, v):
        return self._keys[v]

    def G(self):
        return self._g

    def __str__(self):
        g = self.G()
        s = f'{g.V()} vertices, {g.E()} edges\n'
        for v in range(0, g.V()):
            s = f'{s}{self.name(v)}: '
            for w in g.adj(v):
                s = f'{s}{self.name(w)} '
            s = f'{s}\n'
        return s


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


class Edge():

    def __init__(self, v, w, weight):
        self._v = v
        self._w = w
        self._weight = weight

    def weight(self):
        return self._weight

    def either(self):
        return self._v

    def other(self, v):
        if v == self._v:
            return self._w
        elif v == self._w:
            return self._v
        else:
            raise RuntimeError('Edge %s doesn\'t connect to vertex %s',
                               self, v)

    def __lt__(self, other):
        return self.weight() < other.weight()

    def __gt__(self, other):
        return self.weight() > other.weight()

    def __eq__(self, other):
        return self.weight() == other.weight()

    def __str__(self):
        return f'{self._v} - {self._w} ({self._weight:.2f})'

    def __repr__(self):
        return str(self)


class EdgeWeightedGraph():

    def __init__(self, v, conns=None):
        if conns is None:
            conns = []
        self._v = v
        self._e = 0
        self._adj_list = [[] for _ in range(v)]
        for conn in conns:
            self.add_edge(Edge(*conn))

    def V(self):
        return self._v

    def E(self):
        return self._e

    def add_edge(self, e):
        v = e.either()
        w = e.other(v)
        self._adj_list[v].append(e)
        self._adj_list[w].append(e)
        self._e += 1

    def adj(self, v):
        return self._adj_list[v]

    def edges(self):
        edges = []
        for v in range(self.V()):
            for e in self.adj(v):
                if e.other(v) > v:
                    edges.append(e)
        return edges

    def __str__(self):
        s = f'{self.V()} vertices, {self.E()} edges\n'
        for v in range(0, self.V()):
            s = f'{s}{v}: '
            for w in self.adj(v):
                s = f'{s}{w} '
            s = f'{s}\n'
        return s


class DirectedEdge(Edge):

    def out(self):
        return self._v

    def to(self):
        return self._w

    def __str__(self):
        return f'{self._v} -> {self._w} ({self._weight:.2f})'


class EdgeWeightedDigraph(EdgeWeightedGraph):

    def __init__(self, v, conns=None):
        if conns is None:
            conns = []
        self._v = v
        self._e = 0
        self._adj_list = [[] for _ in range(v)]
        for conn in conns:
            self.add_edge(DirectedEdge(*conn))

    def add_edge(self, e):
        self._adj_list[e.out()].append(e)
        self._e += 1

    def edges(self):
        edges = []
        for v in range(self.V()):
            for e in self.adj(v):
                edges.append(e)
        return edges


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


class BFS(Search):

    def __init__(self, g, s):
        super().__init__(g, s)
        self.bfs(g, s)

    def bfs(self, g, v):
        q = deque()
        q.append(v)
        self._mark(v)
        self._count += 1
        while q:
            v = q.popleft()
            for w in g.adj(v):
                if not self.marked(w):
                    self._mark(w)
                    self._count += 1
                    q.append(w)


class Paths(Search):

    def __init__(self, g, s):
        super().__init__(g, s)
        self._paths = [None, ] * g.V()

    def has_path_to(self, v):
        return

    def path_to(self, v):
        return


class DFSPaths(Paths):

    def __init__(self, g, s):
        super().__init__(g, s)
        self.paths(g, s)

    def paths(self, g, v):
        self._mark(v)
        self._count += 1
        for w in g.adj(v):
            if not self.marked(w):
                self._paths[w] = v
                self.paths(g, w)

    def has_path_to(self, v):
        return self.marked(v)

    def path_to(self, v):
        if not self.has_path_to(v):
            return []
        path_stack = []
        w = v
        while w is not None and w != self._s:
            path_stack.append(w)
            w = self._paths[w]
        path_stack.append(self._s)
        return [path_stack.pop() for _ in range(len(path_stack))]


class BFSPaths(Paths):

    def __init__(self, g, s):
        super().__init__(g, s)
        self.paths(g, s)

    def paths(self, g, v):
        q = deque()
        q.append(v)
        self._mark(v)
        self._count += 1
        while q:
            v = q.popleft()
            for w in g.adj(v):
                if not self.marked(w):
                    self._mark(w)
                    self._count += 1
                    self._paths[w] = v
                    q.append(w)

    def has_path_to(self, v):
        return self.marked(v)

    def path_to(self, v):
        if not self.has_path_to(v):
            return []
        path_stack = []
        w = v
        while w is not None and w != self._s:
            path_stack.append(w)
            w = self._paths[w]
        path_stack.append(self._s)
        return [path_stack.pop() for _ in range(len(path_stack))]


class CC(Search):

    def __init__(self, g):
        super().__init__(g, None)
        self._ids = [0, ] * g.V()

    def is_connected(self, v, w):
        return self._ids[v] == self._ids[w]

    def id(self, v):
        return self._ids[v]


class DFSCC(CC):

    def __init__(self, g):
        super().__init__(g)
        self.calc(g)

    def calc(self, g):
        for v in range(g.V()):
            if not self.marked(v):
                self._dfs_w_id(g, v, self._count)
                self._count += 1

    def _dfs_w_id(self, g, v, i):
        self._mark(v)
        self._ids[v] = i
        for w in g.adj(v):
            if not self.marked(w):
                self._dfs_w_id(g, w, i)


class Cycle(Search):

    def __init__(self, g, s):
        super().__init__(g, s)
        self._has_cycle = False
        for v in range(g.V()):
            if not self.marked(v):
                self.dfs(g, v, v)

    def dfs(self, g, v, u):
        self._mark(v)
        for w in g.adj(v):
            if not self.marked(w):
                self.dfs(g, w, v)
            elif w != u:
                # True if w is already visited, but we didn't come to this
                # iteration from it (u), that means we have visited it at some
                # previous step, but not the immediate previous step, therefore
                # it is part of some other "path" in graph, but connected to
                # this verticle, so it's a cycle.
                self._has_cycle = True

    def has_cycle(self):
        return self._has_cycle


class TwoColor(Search):

    def __init__(self, g, s):
        super().__init__(g, s)
        self._is_two_color = True
        self._colors = [False, ] * g.V()
        for v in range(g.V()):
            if not self.marked(v):
                self.dfs(g, v)

    def dfs(self, g, v):
        self._mark(v)
        for w in g.adj(v):
            if not self.marked(w):
                self._colors[w] = not self._colors[v]
                self.dfs(g, w)
            elif self._colors[w] == self._colors[v]:
                self._is_two_color = False

    def is_two_color(self):
        return self._is_two_color


class DirectedCycle(Paths):

    def __init__(self, g):
        super().__init__(g, None)
        self._cycle_stack = [];
        self._on_call_stack = [False, ] * g.V()
        self._g = g
        for v in range(g.V()):
            if not self.marked(v):
                self.dfs(g, v)

    def dfs(self, g, v):
        self._on_call_stack[v] = True
        self._mark(v)
        for w in g.adj(v):
            if self.has_cycle():
                return
            if not self.marked(w):
                self._paths[w] = v
                self.dfs(g, w)
            elif self._on_call_stack[w]:
                # Got a cycle
                self._cycle_stack = []
                x = v
                while x is not None and x != w:
                    self._cycle_stack.append(x)
                    x = self._paths[x]
                self._cycle_stack.append(w)
                self._cycle_stack.append(v)
        self._on_call_stack[v] = False

    def has_cycle(self):
        return bool(self._cycle_stack)

    def cycle(self):
        return [self._cycle_stack.pop() for _ in range(len(self._cycle_stack))]


class Topological():

    def __init__(self, g):
        self._order = []
        self._d_cycle = DirectedCycle(g)
        if not self._d_cycle.has_cycle():
            self._dfs = DepthFirstOrder(g)
            self._order = self._dfs.reverse_post()

    def is_dag(self):
        return bool(self._order)

    def order(self):
        return self._order


class DepthFirstOrder(Search):

    def __init__(self, g):
        super().__init__(g, None)
        self._pre = deque()
        self._post = deque()
        self._rev_post = []  # stack
        for v in range(g.V()):
            if not self.marked(v):
                self.dfs(g, v)

    def dfs(self, g, v):
        self._pre.append(v)
        self._mark(v)
        self._count += 1
        for w in g.adj(v):
            if not self.marked(w):
                self.dfs(g, w)
        self._post.append(v)
        self._rev_post.append(v)

    def pre(self):
        return self._pre

    def post(self):
        return self._post

    def reverse_post(self):
        return [self._rev_post.pop() for _ in range(len(self._rev_post))]


class KosarajuDFSStrongCC(CC):

    def __init__(self, g):
        super().__init__(g)
        self.calc(g)

    def calc(self, g):
        rev_g = g.reverse()
        order = DepthFirstOrder(rev_g)
        rev_post_order = order.reverse_post()
        for v in rev_post_order:
            if not self.marked(v):
                self._dfs_w_id(g, v, self._count)
                self._count += 1

    def _dfs_w_id(self, g, v, i):
        self._mark(v)
        self._ids[v] = i
        for w in g.adj(v):
            if not self.marked(w):
                self._dfs_w_id(g, w, i)

    def is_strongly_connected(self, v, w):
        return super().is_connected(v, w)


class TransitiveClosure():

    def __init__(self, g):
        self._conn_matrix = []
        for v in range(g.V()):
            self._conn_matrix.append(DirectedDFS(g, v)._marked)

    def reachable(self, v, w):
        return self._conn_matrix[v][w]


class MST():

    def __init__(self, g):
        self._g = g

    def edges(self):
        return

    def weight(self):
        return


class LazyPrimMST(MST):

    def __init__(self, g):
        super().__init__(g)
        self._marked = [False, ] * g.V()
        self._mst_e = deque()
        self._x_edges = []  # Crossing edges, Min Heap
        self._find_MST(g, 0)

    def _find_MST(self, g, v):
        self._visit(g, v)
        while self._x_edges:
            min_e = heapq.heappop(self._x_edges)
            v = min_e.either()
            w = min_e.other(v)
            if self._marked[v] and self._marked[w]:
                continue
            self._mst_e.append(min_e)
            if not self._marked[v]:
                self._visit(g, v)
            if not self._marked[w]:
                self._visit(g, w)

    def _visit(self, g, v):
        self._marked[v] = True
        for e in g.adj(v):
            if not self._marked[e.other(v)]:
                heapq.heappush(self._x_edges, e)

    def edges(self):
        return list(self._mst_e)

    def weight(self):
        weight = 0
        for e in self.edges():
            weight += e.weight()
        return weight


class KruskalMST(MST):

    def __init__(self, g):
        super().__init__(g)
        self._mst_e = deque()
        self._x_edges = g.edges()  # Min Heap
        heapq.heapify(self._x_edges)
        self._ms_forest = UF(g.V())
        self._find_MST(g, 0)

    def _find_MST(self, g, v):
        while self._x_edges and len(self._mst_e) < g.V() - 1:
            min_e = heapq.heappop(self._x_edges)
            v = min_e.either()
            w = min_e.other(v)
            if self._ms_forest.connected(v, w):
                continue
            self._mst_e.append(min_e)
            self._ms_forest.union(v, w)

    def edges(self):
        return list(self._mst_e)

    def weight(self):
        weight = 0
        for e in self.edges():
            weight += e.weight()
        return weight


class ShortestPath():

    def __init__(self, g, s):
        self._g = g
        self._s = s
        self._edge_to = [None, ] * g.V()
        self._dist_to = [float('inf'), ] * g.V()
        self._dist_to[s] = 0

    def dist_to(self, v):
        return self._dist_to[v]

    def has_path_to(self, v):
        return self._dist_to[v] < float('inf')

    def path_to(self, v):
        if not self.has_path_to(v):
            return None
        path_stack = []
        e = self._edge_to[v]
        while e is not None:
            path_stack.append(e)
            e = self._edge_to[e.out()]
        # Actual stack print would be already in the right order
        return reversed(path_stack) if path_stack else []

    def relax_edge(self, e):
        v = e.out()
        w = e.to()
        if self._dist_to[w] > self._dist_to[v] + e.weight():
            self._dist_to[w] = self._dist_to[v] + e.weight()
            self._edge_to[w] = e

    def relax(self, g, v):
        for e in g.adj(v):
            self.relax_edge(e)
            # Original book implementation
            # w = e.to()
            # if self._dist_to[w] > self._dist_to[v] + e.weight():
            #     self._dist_to[w] = self._dist_to[v] + e.weight()
            #     self._edge_to[w] = e


class DijkstraSP(ShortestPath):

    class PQItem():

        def __init__(self, v, dist):
            self.v = v
            self.dist = dist

        def __lt__(self, other):
            return self.dist < other.dist

        def __eq__(self, other):
            return abs(self.dist - other.dist) < 0.0001

    def __init__(self, g, s):
        super().__init__(g, s)
        self._items_in_pq = {}  # Imitating IndexPQ
        self._x_edges = []  # Min priority queue
        self.find_path(g, s)

    def _pq_update(self, v):
        if v in self._items_in_pq:
            self._items_in_pq[v].dist = self._dist_to[v]
            heapq.heapify(self._x_edges)
        else:
            pq_item = self.PQItem(v, self._dist_to[v])
            self._items_in_pq[v] = pq_item
            heapq.heappush(self._x_edges, pq_item)

    def _pq_get_min(self):
        pq_item = heapq.heappop(self._x_edges)
        self._items_in_pq.pop(pq_item.v)
        return pq_item.v

    def find_path(self, g, s):
        self._pq_update(s)
        while self._x_edges:
            self.relax(g, self._pq_get_min())

    def relax(self, g, v):
        for e in g.adj(v):
            w = e.to()
            if self._dist_to[w] > self._dist_to[v] + e.weight():
                self._dist_to[w] = self._dist_to[v] + e.weight()
                self._edge_to[w] = e
                self._pq_update(w)


class DAGSP(ShortestPath):

    def __init__(self, g, s):
        super().__init__(g, s)
        self._rev_post = []  # stack
        self.marked = [False, ] * g.V()
        self.find_path(g, s)

    def find_path(self, g, s):
        self.order_topologically(g)
        topological_order = self.reverse_post()
        print(f'Topological order: {topological_order}')
        for v in topological_order:
            self.relax(g, v)

    def relax(self, g, v):
        for e in g.adj(v):
            w = e.to()
            if self._dist_to[w] > self._dist_to[v] + e.weight():
                self._dist_to[w] = self._dist_to[v] + e.weight()
                self._edge_to[w] = e

    def order_topologically(self, g):
        for v in range(g.V()):
            if not self.marked[v]:
                self.dfs(g, v)

    def dfs(self, g, v):
        self.marked[v] = True
        for e in g.adj(v):
            w = e.to()
            if not self.marked[w]:
                self.dfs(g, w)
        self._rev_post.append(v)

    def reverse_post(self):
        return [self._rev_post.pop() for _ in range(len(self._rev_post))]


class DAGLP(ShortestPath):

    def __init__(self, g, s):
        # Create a DAG copy and inverse edge weights
        import copy
        g = copy.deepcopy(g)
        for e in g.edges():
            e._weight = -e._weight
        super().__init__(g, s)
        self._rev_post = []  # stack
        self.marked = [False, ] * g.V()
        self.find_path(g, s)

    def find_path(self, g, s):
        self.order_topologically(g)
        topological_order = self.reverse_post()
        print(f'Topological order: {topological_order}')
        for v in topological_order:
            self.relax(g, v)
        # Inverse weights back
        for i in range(len(self._dist_to)):
            self._dist_to[i] = -self._dist_to[i]
        for e in self._edge_to:
            if e is None:
                continue
            e._weight = -e._weight

    def relax(self, g, v):
        for e in g.adj(v):
            w = e.to()
            if self._dist_to[w] > self._dist_to[v] + e.weight():
                self._dist_to[w] = self._dist_to[v] + e.weight()
                self._edge_to[w] = e

    def order_topologically(self, g):
        for v in range(g.V()):
            if not self.marked[v]:
                self.dfs(g, v)

    def dfs(self, g, v):
        self.marked[v] = True
        for e in g.adj(v):
            w = e.to()
            if not self.marked[w]:
                self.dfs(g, w)
        self._rev_post.append(v)

    def reverse_post(self):
        return [self._rev_post.pop() for _ in range(len(self._rev_post))]


class DAGLPAlt(ShortestPath):
    # Longest Path

    def __init__(self, g, s):
        super().__init__(g, s)
        self._rev_post = []  # stack
        self.marked = [False, ] * g.V()
        # Inverse initialization
        self._dist_to = [float('-inf'), ] * g.V()
        self._dist_to[s] = 0
        self.find_path(g, s)

    def has_path_to(self, v):
        # Inverse comparison
        return self._dist_to[v] > float('-inf')

    def find_path(self, g, s):
        self.order_topologically(g)
        topological_order = self.reverse_post()
        print(f'Topological order: {topological_order}')
        for v in topological_order:
            self.relax(g, v)

    def relax(self, g, v):
        for e in g.adj(v):
            w = e.to()
            # Inverse comparison
            if self._dist_to[w] < self._dist_to[v] + e.weight():
                self._dist_to[w] = self._dist_to[v] + e.weight()
                self._edge_to[w] = e

    def order_topologically(self, g):
        for v in range(g.V()):
            if not self.marked[v]:
                self.dfs(g, v)

    def dfs(self, g, v):
        self.marked[v] = True
        for e in g.adj(v):
            w = e.to()
            if not self.marked[w]:
                self.dfs(g, w)
        self._rev_post.append(v)

    def reverse_post(self):
        return [self._rev_post.pop() for _ in range(len(self._rev_post))]


class CPMScheduling():
    """Critical path method for parallel precedence-constrained job scheduling."""

    def __init__(self, jobs):
        # * 2 for job start and end vertexes
        # + 2 for schedule start and end vertexes
        n = len(jobs)
        v_count = n * 2 + 2
        start = 2 * n
        end = 2 * n + 1
        g = EdgeWeightedDigraph(v_count)
        for v in range(n):
            duration = jobs[v][0]
            # Edge between job start and job end
            g.add_edge(DirectedEdge(v, v+n, duration))
            # Edge between schedule start and job start
            g.add_edge(DirectedEdge(start, v, 0))
            # Edge between job end and schedule end
            g.add_edge(DirectedEdge(v+n, end, 0))
            for w in jobs[v][1:]:
                # Edge between this job end and its successor
                g.add_edge(DirectedEdge(v+n, w, 0))
        self._g = g
        self._n = n
        self._start = start
        self._end = end
        self._lp = DAGLP(g, start)

    def print_schedule(self):
        print('Start times:')
        for v in range(self._n):
            print(f'{v:4d}: {self._lp._dist_to[v]:5.1f}')

    def print_total_time(self):
        print(f'Total time: {self._lp._dist_to[self._end]:5.1f}')


class GeneralSP(ShortestPath):

    def __init__(self, g, s):
        super().__init__(g, s)
        self.find_path(g, s)

    def find_path(self, g, s):
        for _ in range(g.V()):
            for v in range(g.V()):
                self.relax(g, v)


class BellmanFordSP(ShortestPath):

    def __init__(self, g, s):
        super().__init__(g, s)
        self._v_queue = deque()
        self._on_queue = [False, ] * g.V()
        self._cost = 0
        self._cycle = None
        self.find_path(g, s)

    def find_path(self, g, s):
        self._v_queue.append(s)
        self._on_queue[s] = True
        while (self._v_queue and not self.has_negative_cycle()):
            v = self._v_queue.popleft()
            self._on_queue[v] = False
            self.relax(g, v)

    def relax(self, g, v):
        for e in g.adj(v):
            w = e.to()
            if self._dist_to[w] > self._dist_to[v] + e.weight():
                self._dist_to[w] = self._dist_to[v] + e.weight()
                self._edge_to[w] = e
                if not self._on_queue[w]:
                    self._v_queue.append(w)
                    self._on_queue[w] = True

            self._cost += 1
            if self._cost % g.V() == 0:
                self._find_negative_cycle()

    def _find_negative_cycle(self):
        V = len(self._edge_to)
        g = EdgeWeightedDigraph(V)
        for e in self._edge_to:
            if e is None:
                continue
            g.add_edge(e)
        cycle_finder = EdgeWeightedDirectedCycle(g)
        self._cycle = cycle_finder.cycle()

    def has_path_to(self, v):
        if self.has_negative_cycle():
            return False
        return super().has_path_to(v)

    def has_negative_cycle(self):
        return bool(self._cycle)

    def negative_cycle(self):
        return self._cycle or None


class EdgeWeightedDirectedCycle(Paths):

    def __init__(self, g):
        super().__init__(g, None)
        self._cycle_stack = [];
        self._on_call_stack = [False, ] * g.V()
        self._g = g
        for v in range(g.V()):
            if not self.marked(v):
                self.dfs(g, v)

    def dfs(self, g, v):
        self._on_call_stack[v] = True
        self._mark(v)
        for e in g.adj(v):
            w = e.to()
            if self.has_cycle():
                return
            if not self.marked(w):
                self._paths[w] = e
                self.dfs(g, w)
            elif self._on_call_stack[w]:
                # Got a cycle
                self._cycle_stack = []
                x = e
                while x is not None and x.out() != w:
                    self._cycle_stack.append(x)
                    x = self._paths[x.out()]
                self._cycle_stack.append(x)
        self._on_call_stack[v] = False

    def has_cycle(self):
        return bool(self._cycle_stack)

    def cycle(self):
        return [self._cycle_stack.pop() for _ in range(len(self._cycle_stack))]


class Arbitrage():

    def __init__(self, convs):
        V = len(convs)
        self._names = []
        g = EdgeWeightedDigraph(V)
        for v in range(V):
            conv = convs[v]
            self._names.append(conv[0])
            conv = conv[1:]
            for w in range(V):
                g.add_edge(DirectedEdge(v, w, -math.log(conv[w])))
        print(f'Currency graph:\n{g}')
        self._bf_sp = BellmanFordSP(g, v)

    def arbitrage(self):
        if self._bf_sp.has_negative_cycle():
            stake = 1000
            for e in self._bf_sp.negative_cycle():
                rate = math.exp(-e.weight())
                old_stake = stake
                stake *= rate
                print(f'{old_stake:10.5f} {self._names[e.out()]} = '
                      f'{stake:10.5f} {self._names[e.to()]} @ {rate:10.5f}')
        else:
            print('No cycle found.')



if __name__ == "__main__":
    undi_V = 13
    undi_E = 13
    undi_conns = [
        (0, 5),
        (4, 3),
        (0, 1),
        (9, 12),
        (6, 4),
        (5, 4),
        (0, 2),
        (11, 12),
        (9, 10),
        (0, 6),
        (7, 8),
        (9, 11),
        (5, 3),
    ]
    di_V = 13
    di_E = 22
    di_conns = [
        (4, 2),
        (2, 3),
        (3, 2),
        (6, 0),
        (0, 1),
        (2, 0),
        (11, 12),
        (12, 9),
        (9, 10),
        (9, 11),
        (8, 9),
        (10, 12),
        (11, 4),
        (4, 3),
        (3, 5),
        (7, 8),
        (8, 7),
        (5, 4),
        (0, 5),
        (6, 4),
        (6, 9),
        (7, 6),
    ]
    dag_V = 13
    dag_E = 22
    dag_conns = [
        (0, 1),
        (0, 5),
        (0, 6),
        (2, 0),
        (2, 3),
        (3, 5),
        (5, 4),
        (6, 4),
        (6, 9),
        (7, 6),
        (8, 7),
        (9, 10),
        (9, 11),
        (9, 12),
        (11, 12),
    ]
    g = Graph(undi_V, undi_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'G: {g}')

    d = g.degree(12)
    print(f'Degree 12: {d}')
    if d == 2:
        print('OK')
    else:
        print('FAIL')

    d = g.max_degree()
    print(f'Max Degree: {d}')
    if d == 4:
        print('OK')
    else:
        print('FAIL')

    d = g.avg_degree()
    print(f'Avg Degree: {d}')
    if d == 2:
        print('OK')
    else:
        print('FAIL')

    sl = g.self_loops_count()
    print(f'Self loops count: {sl}')
    if sl == 0:
        print('OK')
    else:
        print('FAIL')

    print()
    di_g = Digraph(di_V, di_conns)
    print(f'Di G: {di_g}')

    print()
    di_g_r = di_g.reverse()
    print(f'Reversed Di G: {di_g_r}')

    sl = di_g.self_loops_count()
    print(f'Self loops count: {sl}')
    if sl == 0:
        print('OK')
    else:
        print('FAIL')

    for S in [Search, DFS, BFS]:
        print(f'\n-= {S.__name__} =-\n')
        for source in [0, 9]:
            s = S(g, source)
            connected = []
            for v in range(g.V()):
                if s.marked(v):
                    connected.append(v)
            print(f'Connected to {source}: {connected}.')
            print(f'Count connected to {source}: {s.count()}.')
            if s.count() == g.V():
                print('Connected graph!')
            else:
                print('Not connected graph.')

    for P in [Paths, DFSPaths, BFSPaths]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [0, 9]:
            p = P(g, source)
            for v in range(g.V()):
                if p.has_path_to(v):
                    print('-'.join([str(pth) for pth in p.path_to(v)]))

    for C in [CC, DFSCC]:
        print(f'\n-= {C.__name__} =-\n')
        c = C(g)
        m = c.count() or 1
        print(f'Components count: {m}')
        components = [[] for _ in range(m)]
        for v in range(g.V()):
            components[c.id(v)].append(v)
        for comp in components:
            print(f'{c.id(comp[0])}: {comp}')

    for C in [Cycle, ]:
        print(f'\n-= {C.__name__} =-\n')
        for source in [0, 9]:
            c = C(g, source)
            print(f'Source {source} has cycle: {c.has_cycle()}.')

    for C in [TwoColor, ]:
        print(f'\n-= {C.__name__} =-\n')
        for source in [0, 9]:
            c = C(g, source)
            print(f'Source {source} is two-colorable: {c.is_two_color()}.')

    conns = [
        ('JFK', 'MCO'),
        ('ORD', 'DEN'),
        ('ORD', 'HOU'),
        ('DFW', 'PHX'),
        ('JFK', 'ATL'),
        ('ORD', 'DFW'),
        ('ORD', 'PHX'),
        ('ATL', 'HOU'),
        ('DEN', 'PHX'),
        ('PHX', 'LAX'),
        ('JFK', 'ORD'),
        ('DEN', 'LAS'),
        ('DFW', 'HOU'),
        ('ORD', 'ATL'),
        ('LAS', 'LAX'),
        ('ATL', 'MCO'),
        ('HOU', 'MCO'),
        ('LAS', 'PHX'),
    ]
    g = SympolGraph(conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'G: {g}')

    d = g.G().degree(g.index('ORD'))
    print(f'Degree "ORD": {d}')
    if d == 6:
        print('OK')
    else:
        print('FAIL')

    d = g.G().max_degree()
    print(f'Max Degree: {d}')
    if d == 6:
        print('OK')
    else:
        print('FAIL')

    d = g.G().avg_degree()
    print(f'Avg Degree: {d}')
    if d == 3:
        print('OK')
    else:
        print('FAIL')

    sl = g.G().self_loops_count()
    print(f'Self loops count: {sl}')
    if sl == 0:
        print('OK')
    else:
        print('FAIL')

    for P in [BFSPaths, ]:
        print(f'\n-= {P.__name__} =-\n')
        for source in ['LAS', 'DFW']:
            p = P(g.G(), g.index(source))
            v = g.index('JFK')
            if p.has_path_to(v):
                print('-'.join([g.name(pth) for pth in p.path_to(v)]))

    g = di_g
    print(f'\n-= {g.__class__.__name__} =-\n')
    for S in [Search, DirectedDFS]:
        print(f'\n-= {S.__name__} =-\n')
        for source in [1, 2, [1, 2, 6]]:
            s = S(g, source)
            connected = []
            for v in range(g.V()):
                if s.marked(v):
                    connected.append(v)
            print(f'Connected to {source}: {connected}.')
            print(f'Count connected to {source}: {s.count()}.')
            if s.count() == g.V():
                print('Connected graph!')
            else:
                print('Not connected graph.')

    for S in [BFS, ]:
        print(f'\n-= {S.__name__} =-\n')
        for source in [1, 2, 6]:
            s = S(g, source)
            connected = []
            for v in range(g.V()):
                if s.marked(v):
                    connected.append(v)
            print(f'Connected to {source}: {connected}.')
            print(f'Count connected to {source}: {s.count()}.')
            if s.count() == g.V():
                print('Connected graph!')
            else:
                print('Not connected graph.')

    for P in [Paths, DFSPaths, BFSPaths]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [1, 2, 6]:
            p = P(g, source)
            for v in range(g.V()):
                if p.has_path_to(v):
                    print('-'.join([str(pth) for pth in p.path_to(v)]))

    for C in [DirectedCycle, ]:
        print(f'\n-= {C.__name__} =-\n')
        c = C(g)
        print(f'Has cycle: {c.has_cycle()}.')
        print(f'Cycle: {c.cycle()}.')

    print()
    dag_g = Digraph(dag_V, dag_conns)
    print(f'DAG G: {dag_g}')

    g = dag_g
    print(f'\n-= {g.__class__.__name__} =-\n')

    for O in [DepthFirstOrder, ]:
        print(f'\n-= {O.__name__} =-\n')
        o = O(g)
        print(f'Pre: {o.pre()}.')
        print(f'Post: {o.post()}.')
        print(f'Reverse post: {o.reverse_post()}.')

    for T in [Topological, ]:
        print(f'\n-= {T.__name__} =-\n')
        t = T(g)
        print(f'Topological order: {t.order()}.')


    for g in [di_g, dag_g]:
        for C in [KosarajuDFSStrongCC, ]:
            print(f'\n-= {C.__name__} =-\n')
            c = C(g)
            m = c.count() or 1
            print(f'Components count: {m}')
            components = [[] for _ in range(m)]
            for v in range(g.V()):
                components[c.id(v)].append(v)
            for comp in components:
                print(f'{c.id(comp[0])}: {comp}')


        for T in [TransitiveClosure, ]:
            print(f'\n-= {T.__name__} =-\n')
            for v in [1, 2, 6, 12]:
                for w in [1, 2, 6, 12]:
                    t = T(g)
                    print(f'{w} is reachable from {v}: {t.reachable(v, w)}')
            print(f'   {" ".join([str(i) for i in range(g.V())])}')
            for n, row in enumerate(t._conn_matrix):
                print(f'{n if n > 9 else " %s" % n} {" ".join([str(int(i)) for i in row])}')

    w_V = 8
    w_E = 16
    w_conns = [
        (4, 5, 0.35),
        (4, 7, 0.37),
        (5, 7, 0.28),
        (0, 7, 0.16),
        (1, 5, 0.32),
        (0, 4, 0.38),
        (2, 3, 0.17),
        (1, 7, 0.19),
        (0, 2, 0.26),
        (1, 2, 0.36),
        (1, 3, 0.29),
        (2, 7, 0.34),
        (6, 2, 0.40),
        (3, 6, 0.52),
        (6, 0, 0.58),
        (6, 4, 0.93),
    ]
    g = EdgeWeightedGraph(w_V, w_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'Edge Weighted Graph G: {g}')

    for M in [MST, LazyPrimMST, KruskalMST, ]:
        print(f'\n-= {M.__name__} =-\n')
        m = M(g)
        print(f'MST edges: {m.edges()}')
        print(f'MST weight: {m.weight()}')

    di_w_V = 8
    di_w_E = 15
    di_w_conns = [
        (4, 5, 0.35),
        (5, 4, 0.35),
        (4, 7, 0.37),
        (5, 7, 0.28),
        (7, 5, 0.28),
        (5, 1, 0.32),
        (0, 4, 0.38),
        (0, 2, 0.26),
        (7, 3, 0.39),
        (1, 3, 0.29),
        (2, 7, 0.34),
        (6, 2, 0.40),
        (3, 6, 0.52),
        (6, 0, 0.58),
        (6, 4, 0.93),
    ]
    g = EdgeWeightedDigraph(di_w_V, di_w_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'Edge Weighted Digraph G: {g}')

    for P in [ShortestPath, DijkstraSP, ]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [0, 2, 6]:
            p = P(g, source)
            for v in range(g.V()):
                print(f'Distance {source} - {v}: {p.dist_to(v):.2f}')
                if p.has_path_to(v):
                    print(' >> '.join([str(pth) for pth in p.path_to(v)]))

    dag_w_V = 8
    dag_w_E = 13
    dag_w_conns = [
        (5, 4, 0.35),
        (4, 7, 0.37),
        (5, 7, 0.28),
        (5, 1, 0.32),
        (4, 0, 0.38),
        (0, 2, 0.26),
        (3, 7, 0.39),
        (1, 3, 0.29),
        (7, 2, 0.34),
        (6, 2, 0.40),
        (3, 6, 0.52),
        (6, 0, 0.58),
        (6, 4, 0.93),
    ]
    g = EdgeWeightedDigraph(dag_w_V, dag_w_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'Edge Weighted DAG G: {g}')

    for P in [DAGSP, DAGLP, DAGLPAlt, ]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [5, ]:
            p = P(g, source)
            for v in range(g.V()):
                print(f'Distance {source} - {v}: {p.dist_to(v):.2f}')
                if p.has_path_to(v):
                    print(' >> '.join([str(pth) for pth in p.path_to(v)]))

    jobs = [
        (41.0, 1, 7, 9),
        (51.0, 2),
        (50.0, ),
        (36.0, ),
        (38.0, ),
        (45.0, ),
        (21.0, 3, 8),
        (32.0, 3, 8),
        (32.0, 2),
        (29.0, 4, 6),
    ]
    print(f'Jobs:')
    for j in jobs:
        print(j)
    cpm = CPMScheduling(jobs)
    cpm.print_total_time()
    cpm.print_schedule()

    di_nw_V = 8
    di_nw_E = 15
    di_nw_conns = [
        (4, 5, 0.35),
        (5, 4, 0.35),
        (4, 7, 0.37),
        (5, 7, 0.28),
        (7, 5, 0.28),
        (5, 1, 0.32),
        (0, 4, 0.38),
        (0, 2, 0.26),
        (7, 3, 0.39),
        (1, 3, 0.29),
        (2, 7, 0.34),
        (6, 2, -1.20),
        (3, 6, 0.52),
        (6, 0, -1.40),
        (6, 4, -1.25),
    ]
    g = EdgeWeightedDigraph(di_nw_V, di_nw_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'Edge Weighted Directed Graph (with negative weights) G: {g}')

    for P in [GeneralSP, BellmanFordSP, ]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [0, 2, 6]:
            p = P(g, source)
            for v in range(g.V()):
                print(f'Distance {source} - {v}: {p.dist_to(v):.2f}')
                if p.has_path_to(v):
                    print(' >> '.join([str(pth) for pth in p.path_to(v)]))
            if isinstance(p, BellmanFordSP):
                print(f'Negative Cycle: {p.negative_cycle()}')

    di_nw_nc_V = 8
    di_nw_nc_E = 15
    di_nw_nc_conns = [
        (4, 5, 0.35),
        (5, 4, -0.66),
        (4, 7, 0.37),
        (5, 7, 0.28),
        (7, 5, 0.28),
        (5, 1, 0.32),
        (0, 4, 0.38),
        (0, 2, 0.26),
        (7, 3, 0.39),
        (1, 3, 0.29),
        (2, 7, 0.34),
        (6, 2, -1.20),
        (3, 6, 0.52),
        (6, 0, -1.40),
        (6, 4, -1.25),
    ]
    g = EdgeWeightedDigraph(di_nw_nc_V, di_nw_nc_conns)
    print(f'\n-= {g.__class__.__name__} =-\n')
    print(f'Edge Weighted Directed Graph (with negative weights) G: {g}')

    for P in [BellmanFordSP, ]:
        print(f'\n-= {P.__name__} =-\n')
        for source in [0, 2, 6]:
            p = P(g, source)
            for v in range(g.V()):
                print(f'Distance {source} - {v}: {p.dist_to(v):.2f}')
                if p.has_path_to(v):
                    print(' >> '.join([str(pth) for pth in p.path_to(v)]))
            if isinstance(p, BellmanFordSP):
                print(f'Negative Cycle: {p.negative_cycle()}')
    print()

    convs = [
        ('USD', 1, 0.741, 0.657, 1.061, 1.005),
        ('EUR', 1.349, 1, 0.888, 1.433, 1.366),
        ('GBP', 1.521, 1.126, 1, 1.614, 1.538),
        ('CHF', 0.942, 0.698, 0.619, 1, 0.953),
        ('CAD', 0.995, 0.732, 0.650, 1.049, 1),
    ]
    a = Arbitrage(convs)
    print(f'\n-= {a.__class__.__name__} =-\n')
    a.arbitrage()
