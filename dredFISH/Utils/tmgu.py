import numpy as np
from collections import Counter
import igraph
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

def count_values(V,refvals,sz = None,norm_to_one = True):
    """
    count_values - simple tabulation with default values (refvals)
    """
    Cnt = Counter(V)
    if sz is not None: 
        for i in range(len(V)): 
            Cnt.update({V[i] : sz[i]-1}) # V[i] is already represented, so we need to subtract 1 from sz[i]  
            
    cntdict = dict(Cnt)
    missing = list(set(refvals) - set(V))
    cntdict.update(zip(missing, np.zeros(len(missing))))
    Pv = np.array([cntdict.get(k) for k in sorted(cntdict.keys())])
    if norm_to_one:
        Pv=Pv/np.sum(Pv)
    return(Pv)

def adjacency_to_igraph(adj_mtx, weighted=False, directed=True, simplify=True):
    """
    Converts an adjacency matrix to an igraph object
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        directed (bool): If graph should be directed
    
    Returns:
        G (igraph object): igraph object of adjacency matrix
    
    Uses code from:
        https://github.com/igraph/python-igraph/issues/168
        https://stackoverflow.com/questions/29655111

    Author:
        Wayne Doyle 
        (Fangming Xie modified) 
    """
    nrow, ncol = adj_mtx.shape
    if nrow != ncol:
        raise ValueError('Adjacency matrix should be a square matrix')
    vcount = nrow
    sources, targets = adj_mtx.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = igraph.Graph(n=vcount, edges=edgelist, directed=directed)
    if weighted:
        G.es['weight'] = adj_mtx.data
    if simplify:
        G.simplify() # simplify inplace; remove duplicated and self connection (Important to avoid double counting from adj_mtx)
    return G

def split_spatial_graph_to_sections(graph_csr,section_ids):
    """
    splits a spatial graph into components based on section information
    """
    
    # if input is iGraph, convert to sparse csr: 
    if isinstance(graph_csr,igraph.Graph):
        graph_csr = graph_csr.get_adjacency_sparse()
    
    unqS,countS = np.unique(section_ids,return_counts = True)
    SG = [None] * len(unqS)
                
    # break this sparse matrix into components, one per section:
    strt=0 
    for i in range(len(unqS)):
        sg_section = graph_csr[strt:strt+countS[i],strt:strt+countS[i]]
        strt=strt+countS[i]
        SG = adjacency_to_igraph(sg_section, directed=False)

    return SG

def get_local_type_abundance(
    types, 
    edgelist=None, 
    SG=None, 
    XY=None, 
    k_spatial=10,
    ):
    """
    types - type labels on the nodes
    
    edgelist - a list of edges (assume duplicated if undirected) 
    SG - spatial neighborhood graph (undirected); Use this to generate edgelist
    XY - spatial coordinates; Use this to first generate kNN graph; then to generate edgelist
    
    return - relative abundace of types for each node
    """
    N = len(types)
    ctg, ctg_idx = np.unique(types, return_inverse=True) 
    if edgelist is not None:
        i, j = edgelist

    elif SG is not None and isinstance(SG, igraph.Graph):
        # assume undirected; edges need to be counted twice
        edges = np.asarray(SG.get_edgelist()) 
        # once
        i = edges[:,0] # cells
        j = ctg_idx[edges[:,1]] # types it connects
        # twice
        i2 = edges[:,1] # cells
        j2 = ctg_idx[edges[:,0]] # types it connects
        # merge
        i = np.hstack([i,i2])
        j = np.hstack([j,j2])

    elif XY is not None and isinstance(XY, np.ndarray):
        NN = NearestNeighbors(n_neighbors=k_spatial)
        NN.fit(XY)
        knn = NN.kneighbors(XY, return_distance=False)

        i = np.repeat(knn[:,0], k_spatial-1) # cells
        j = ctg_idx[knn[:,1:]].reshape(-1,) # types it connects

    dat = np.repeat(1, len(i))

    # count
    env_mat = sparse.coo_matrix((dat, (i,j)), shape=(N, len(ctg))).toarray() # dense
    env_mat = env_mat/env_mat.sum(axis=1).reshape(-1,1)
    env_mat = np.nan_to_num(env_mat, 0)
    
    return env_mat

