import numpy as np
from collections import Counter
import igraph
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


import sys
sys.path.append('/home/rwollman/MyProjects/AH/Repos/dredFISH')
from dredFISH.Utils.ConnectedComponentEntropy import ConnectedComponentEntropy

def score_percolation_entropy(type_vec,ELD,total_cells = None, fudge_factor = 1.05, distvec = None,return_entropies = True): 
    # type vec - the integer vector of types we want to test
    # ELD - a sorted (by distances or K) list of edges in the spatial graph with the distance between them. 
    # total_cells - in case we are doing a multi-section analysiis, specificy the number of section, decault is len(type_vec)

    if total_cells is None: 
        total_cells = len(type_vec)

    # calcualte the entropy of type_vec as convergance criteria 
    _,cnt = np.unique(type_vec, return_counts=True)
    freq = cnt/len(type_vec)
    entropy_low_bound = -np.sum(freq * np.log2(freq))

    # update the entropy in case we are dealing with multi-section data
    fudge_factor = fudge_factor * len(type_vec) / total_cells

    # permute types and find edges that connect the same type
    type_vec_perm = np.random.permutation(type_vec)
    ELD_real = ELD[np.equal(type_vec[ELD[:,0].astype(int)],type_vec[ELD[:,1].astype(int)]),:]
    ELD_perm = ELD[np.equal(type_vec_perm[ELD[:,0].astype(int)],type_vec_perm[ELD[:,1].astype(int)]),:]

    # calculate entropies as edges are edded
    uf_real = ConnectedComponentEntropy(len(type_vec),total_cells)
    entropy_real = uf_real.merge_all(ELD_real[:,:2].astype(int),entropy_low_bound * fudge_factor)

    uf_perm = ConnectedComponentEntropy(len(type_vec),total_cells)
    entropy_perm = uf_perm.merge_all(ELD_perm[:,:2].astype(int),entropy_low_bound * fudge_factor)

    # interpolate entropies to the same grid
    if distvec is None: 
        distvec = np.linspace(ELD[:,2].min(),ELD[:,2].max(),1000)
    
    entropy_vecs = np.zeros((len(distvec),2))
    entropy_vecs[:,0] = np.interp(distvec, ELD_real[:,2], entropy_real)
    entropy_vecs[:,1] = np.interp(distvec, ELD_perm[:,2], entropy_perm)

    scr = np.diff(entropy_vecs, axis=1).max()

    if return_entropies: 
        return (scr,distvec,entropy_vecs)
    else:
        return scr


def edge_list_from_XY_with_max_dist(XY,max_dist):
    nbrs = NearestNeighbors(radius = max_dist, algorithm = 'ball_tree').fit(XY)
    distances, indices = nbrs.radius_neighbors(XY)
    nn =[len(d) for d in distances]
    ix_rows = np.repeat(np.arange(len(nn)),nn)
    ix_cols = np.hstack(indices)
    ix_dist = np.hstack(distances)
    ELD = np.hstack((ix_rows[:,np.newaxis],ix_cols[:,np.newaxis],ix_dist[:,np.newaxis]))
    ELD = ELD[ELD[:,2]>0,:]
    ELD = ELD[ELD[:,2].argsort(),:]

    return ELD

def edge_list_from_XY_with_k(XY,k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(XY)
    distances, indices = nbrs.kneighbors(XY)
    distances = distances[:,1:]
    indices = indices[:,1:]

    ix_ks,ix_rows = np.meshgrid(np.arange(1,k+1),np.arange(XY.shape[0]))
    ix_rows = ix_rows.T.flatten()
    ix_ks = ix_ks.T.flatten()
    ix_cols = indices.T.flatten()
    ELK = np.hstack((ix_rows[:,np.newaxis],ix_cols[:,np.newaxis],ix_ks[:,np.newaxis]))

    return ELK

def entropy_from_ELO_and_type(ELO,type_vec,epsilon,return_all = False):
    # first, remove edges that are of order > K to convert ELO to EL
    EL = ELO[ELO[:,2]<=epsilon,:2]

    # calculate zones for real and perm cases
    type_perm = np.random.permutation(type_vec)
    EL_real = EL[np.take(type_vec,EL[:,0].astype(int)) == np.take(type_vec,EL[:,1].astype(int)),:]
    EL_perm = EL[np.take(type_perm,EL[:,0].astype(int)) == np.take(type_perm,EL[:,1].astype(int)),:]

    N = len(type_vec)
    ZG_real = igraph.Graph(N, edges=EL_real, directed = False)
    ZG_perm = igraph.Graph(N, edges=EL_perm, directed = False)

    Pzone_real = np.array(ZG_real.components().sizes())/N
    Pzone_perm = np.array(ZG_perm.components().sizes())/N

    Entropy_real = -np.sum(Pzone_real * np.log2(Pzone_real))
    Entropy_perm = -np.sum(Pzone_perm * np.log2(Pzone_perm))

    Scr = Entropy_perm - Entropy_real

    if return_all: 
        return (Scr,Entropy_perm,Entropy_real)
    else: 
        return Scr

def cond_entropy_from_EL_and_type(EL,TypeVec,return_all = False):
    EL = EL[np.take(TypeVec,EL[:,0]) == np.take(TypeVec,EL[:,1]),:]
    N = len(TypeVec)
    IsoZonesGraph = igraph.Graph(N, edges=EL, directed = False)
    cmp = IsoZonesGraph.components()
    IxMapping = np.array(cmp.membership)
    Pzones = np.bincount(IxMapping)/N
    Ptypes = np.bincount(TypeVec)/N
    Entropy_Zone = -np.sum(Pzones * np.log2(Pzones))

    Ptypes = Ptypes[Ptypes>0]
    Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
    
    cond_entropy = Entropy_Zone-Entropy_Types
    if return_all: 
        return (Entropy_Zone,Entropy_Types,cond_entropy)
    else: 
        return(cond_entropy)



def count_values(V,refvals,max_val = None, sz = None,norm_to_one = True):
    """
    count_values - simple tabulation with default values (refvals)
    """
    raise NotImplementedError('moved functionality into TissueGraph.type_freq')
    # below are performance optimization attempts that I ended up moving to TG.type_freq
    # before AI: 
    # Cnt = Counter(V)
    # if sz is not None: 
    #     for i in range(len(V)): 
    #         Cnt.update({V[i] : sz[i]-1}) # V[i] is already represented, so we need to subtract 1 from sz[i]  
            
    # cntdict = dict(Cnt)

    # suggestion 1 from AI: 
    # unique, counts = np.unique(V, return_counts=True)
    # cntdict = dict(zip(unique, counts))
    # if sz is not None:
    #     cntdict.update(dict(zip(V, sz)))
    # missing = list(set(refvals) - set(V))
    # cntdict.update(zip(missing, np.zeros(len(missing))))

    # suggestion 2 from AI (to avoid the np.unique in the outside call of TissueGraph.type_freq)
    # if max_val is None:
    #     max_val = np.max(refvals)
    # cnts = np.bincount(V, minlength=max_val+1)
    # if sz is not None:
    #     cnts += np.bincount(V, weights=sz, minlength=max_val+1) - 1

    # Pv = cnts / np.sum(cnts) if norm_to_one else cnts

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
    edgelist = list(zip(sources, targets))
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

