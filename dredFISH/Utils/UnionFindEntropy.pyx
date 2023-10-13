# cython: language_level=3
cimport cython

from libc.math cimport log2
from array import array

cdef class UnionFindEntropy:
    cdef int n
    cdef int[:] parent, rank
    cdef double[:] size
    cdef double entropy

    def __init__(self, int n):
        self.n = n
        self.parent = array('i', range(n))
        self.rank = array('i', [0]*n)
        self.size = array('d', [1.0]*n)
        self.entropy = log2(n)

    cpdef int find(self, int u):
        if self.parent[u] == u:
            return u
        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    cpdef merge(self, int u, int v):
        cdef int u_root, v_root
        # find which cc the u,v nodes belong to using root of each cc
        u_root = self.find(u)
        v_root = self.find(v)
        # is in the same cc, move on
        if u_root == v_root:
            return

        # actual merge
        if self.rank[u_root] > self.rank[v_root]:
            u_root, v_root = v_root, u_root
        self.parent[u_root] = v_root
        if self.rank[u_root] == self.rank[v_root]:
            self.rank[v_root] += 1

        # update size
        v_prev_size = self.size[v_root]
        u_prev_size = self.size[u_root]
        self.size[v_root] += self.size[u_root]
        self.size[u_root] = 0

        # update entropy (- of - is + so add prev entropy and then subtract current)
        self.entropy += v_prev_size/self.n * log2(v_prev_size/self.n)
        self.entropy += u_prev_size/self.n * log2(u_prev_size/self.n)
        self.entropy -=  self.size[v_root]/self.n * log2(self.size[v_root]/self.n)

    cpdef merge_all(self,list pairs, float entropy_low_bound = 0.0):
        cdef int u,v
        cdef double run_fraction
        cdef double[:] entropy_log = array('d', [entropy_low_bound]*len(pairs))
        for i,pair in enumerate(pairs):
            u, v = pair
            self.merge(u,v)
            entropy_log[i] = self.entropy
            if self.entropy <= entropy_low_bound:
                run_fraction = 100*i/len(pairs)
                print(f"reached lower bound at iter {i} - {run_fraction:.0f}% of edges") 
                break
        return list(entropy_log)