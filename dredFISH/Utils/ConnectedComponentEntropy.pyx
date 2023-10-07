# cython: language_level=3
cimport cython

from libc.stdio cimport printf
from libc.math cimport log2

import numpy as np
cimport numpy as np

cdef class ConnectedComponentEntropy:
    cdef int n
    cdef np.ndarray parent, rank
    cdef np.ndarray size
    cdef double entropy

    def __init__(self, int n_section, int n_total = -1):
        if n_total == -1: 
            n_total = n_section
        self.n = n_total
        self.parent = np.arange(n_section, dtype=np.int32)
        self.rank = np.zeros(n_section, dtype=np.int32)
        self.size = np.ones(n_section, dtype=np.float64)
        self.entropy = n_section / n_total * log2(n_total)

    cpdef int find(self, int u):
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    cpdef merge(self, int u, int v):
        cdef int u_root, v_root
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
        if self.rank[u_root] > self.rank[v_root]:
            u_root, v_root = v_root, u_root
        self.parent[u_root] = v_root
        if self.rank[u_root] == self.rank[v_root]:
            self.rank[v_root] += 1
        v_prev_size = self.size[v_root]
        u_prev_size = self.size[u_root]
        self.size[v_root] += self.size[u_root]
        self.size[u_root] = 0
        self.entropy += v_prev_size/self.n * log2(v_prev_size/self.n)
        self.entropy += u_prev_size/self.n * log2(u_prev_size/self.n)
        self.entropy -=  self.size[v_root]/self.n * log2(self.size[v_root]/self.n)

    cpdef merge_all(self, np.ndarray[np.int_t, ndim=2] pairs, float entropy_low_bound = 0.0):
        cdef int i, num_pairs = pairs.shape[0]
        cdef double run_fraction
        cdef np.ndarray entropy_log = np.ones(num_pairs, dtype=np.float64) * entropy_low_bound
        for i in range(num_pairs):
            self.merge(pairs[i, 0], pairs[i, 1])
            entropy_log[i] = self.entropy
            if self.entropy <= entropy_low_bound:
                run_fraction = 100*i/num_pairs
                printf("reached lower bound at iter %d - %.0f%% of edges\n", i, run_fraction)
                break
        return np.asarray(entropy_log)