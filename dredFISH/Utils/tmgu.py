import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import seaborn as sns
import os

import igraph
import pynndescent


from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from multiprocessing import Pool, cpu_count


import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 512

from IPython.display import HTML

import sys
import time
sys.path.append('/home/rwollman/MyProjects/AH/Repos/dredFISH')
# from dredFISH.Utils.ConnectedComponentEntropy import ConnectedComponentEntropy

def list_entropy(X):
        _, cnt = np.unique(X, return_counts=True)
        freq = cnt / len(X)
        return -(np.sum(freq * np.log2(freq)))

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

def edge_list_from_XY_with_k(XY,k,include_dist = False):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(XY)
    distances, indices = nbrs.kneighbors(XY)
    distances = distances[:,1:]
    indices = indices[:,1:]

    ix_ks,ix_rows = np.meshgrid(np.arange(1,k+1),np.arange(XY.shape[0]))
    ix_rows = ix_rows.T.flatten()
    ix_ks = ix_ks.T.flatten()
    ix_cols = indices.T.flatten()
    dists = distances.T.flatten()
    ELK = np.hstack((ix_rows[:,np.newaxis],ix_cols[:,np.newaxis],ix_ks[:,np.newaxis]))
    if include_dist: 
        ELK = np.hstack((ELK,dists[:,np.newaxis]))
    return ELK


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

def build_knn_graph(X, metric, n_neighbors=15, accuracy={'prob':1, 'extras':1.5}, metric_kwds={}, allow_KDtree = True): 
    """
    Buils the knn graph. If X is small uses KDTree, if large pynndescent
    """

    # checks if we have enough rows 
    n_neighbors = min(X.shape[0]-1,n_neighbors)

    if X.shape[0] < 200000 and allow_KDtree:
        knn = cKDTree(X)
        distances, indices = knn.query(X, k=n_neighbors+1)
    else:
        knn = pynndescent.NNDescent(X, n_neighbors=n_neighbors+1,
                                    metric=metric,
                                    diversify_prob=accuracy['prob'],
                                    pruning_degree_multiplier=accuracy['extras'],
                                    metric_kwds=metric_kwds)
        indices, distances = knn.neighbor_graph
    
    indices = indices[:, 1:]  # remove self indices
    distances = distances[:, 1:]  # remove self distances

    id_from = np.tile(np.arange(indices.shape[0]),indices.shape[1])
    id_to = indices.flatten(order='F')

    # build graph
    edgeList = np.vstack((id_from,id_to)).T
    G = igraph.Graph(n=X.shape[0], edges=edgeList, edge_attrs={'weight': distances.flatten(order='F')})
    G.simplify()

    return (G,knn)

def compute_neighborhood(args):
    subgraph, offset, order = args
    
    # Compute neighborhood for each node in the subgraph
    neighborhoods = subgraph.neighborhood(order = order)
    adjusted_neighborhoods = np.full((neighborhoods.shape[0], order), -1, dtype=int)
    for i,nbr in enumerate(neighborhoods): 
        adjusted_neighborhoods[i, :len(nbr)] = nbr
    adjusted_neighborhoods += offset

    return adjusted_neighborhoods

def find_graph_neighborhoold_parallel_by_components(G,order):   
    components = G.clusters()
    subgraphs = [G.subgraph(component) for component in components]
    subgraph_sizes = np.array([subgraph.vcount() for subgraph in subgraphs])

    offsets = np.zeros(len(subgraphs))
    offsets[1:] = np.cumsum(subgraph_sizes[:-1])

    args = [(subgraphs[i], offsets[i], order) for i in range(len(subgraphs))]

    num_cores = cpu_count() //2

    with Pool(processes=num_cores) as pool:
        results = pool.map(compute_neighborhood, args)

    neighborhood_matrix = np.vstack(results)

    return neighborhood_matrix



class GraphPercolation:
    def __init__(self, XY, type_vec, maxK = None):
        self.N = XY.shape[0]
        self.XY = XY
        self.type_vec = type_vec
        self.type_vec_perm = np.random.permutation(self.type_vec)
        self.maxK = maxK
        self.unq_types = np.unique(type_vec)
        self.n_types = len(np.unique(type_vec))

        if maxK is None: 
            self.maxK = self.N-1

    def save(self,filename): 
        np.savez(filename,XY = self.XY,
                         type_vec = self.type_vec,
                         type_vec_perm = self.type_vec_perm,
                         Zones = self.Zones,
                         Zones_perm = self.Zones_perm,
                         ent_real = self.ent_real,
                         ent_perm = self.ent_perm,
                         pbond_vec = self.pbond_vec,
                         ent_type_real = self.ent_type_real,
                         ent_type_perm = self.ent_type_perm,
                         maxK = self.maxK
                         )
        
    def load(self,filename): 
        dump = np.load(filename)
        self.XY = dump['XY']
        self.type_vec = dump['type_vec']
        self.type_vec_perm = dump['type_vec_perm']
        self.Zones = dump['Zones']
        self.Zones_perm = dump['Zones_perm']
        self.ent_real = dump['ent_real']
        self.ent_perm = dump['ent_perm']
        self.pbond_vec = dump['pbond_vec']
        self.ent_type_real = dump['ent_type_real']
        self.ent_type_perm = dump['ent_type_perm']
        self.maxK = dump['maxK']
        self.unq_types = np.unique(self.type_vec)
        self.n_types = len(np.unique(self.type_vec))
        self.N = len(self.type_vec)

    def calc_ELKexp(self, permute=False): 
        ELK = self.ELKfull
        if permute: 
            type_vec = self.type_vec_perm
        else: 
            type_vec = self.type_vec

        type_left = type_vec[ELK[:,0].astype(int)]
        type_right = type_vec[ELK[:,1].astype(int)]

        _,ix_inv,type_frac = np.unique(type_vec,return_inverse = True,return_counts=True)
        type_frac = type_frac[ix_inv]/len(type_vec)
        type_frac  = type_frac[ELK[:,0].astype(int)][type_left==type_right]
        ELKexp = np.hstack((ELK[type_left==type_right,:],type_frac[:,np.newaxis]))

        blocks = np.split(ELKexp, np.where(np.diff(ELK[:,2]))[0]+1)
        blocks_mod = list()
        for block in blocks:
            if  block.shape[0] == 0: 
                continue
            np.random.shuffle(block)
            k_cont = block[:,2] + np.arange(block.shape[0])/block.shape[0]-1
            pbond = 1-np.exp(-k_cont*block[:,3])
            ordr = np.argsort(pbond)
            blocks_mod.append(np.hstack((block[ordr,:2],pbond[ordr,np.newaxis])))

        ELKexp = np.concatenate(blocks_mod)
        return ELKexp

    def bond_percolation(self,permute = False, pbond_vec = np.linspace(0,1,101), return_zones = False, verbose = False): 
        # get the edges of same type with weights
        ELP = self.calc_ELKexp(permute = permute)

        # union find object: 
        uf = ConnectedComponentEntropy(self.N)

        # zone entropy at each pbond_vec value
        ent = np.full(len(pbond_vec)-1, np.nan)
        if return_zones: 
            Zones = np.full((self.N, len(pbond_vec)-1),np.nan)

        start = time.time()
        for i in range(len(pbond_vec)-1):
            if verbose: 
                print(f"iter: {i}/{len(pbond_vec)} time: {time.time()-start:.0f}")

            ix_to_merge = np.logical_and(ELP[:,2] > pbond_vec[i],ELP[:,2] <= pbond_vec[i+1])
            if not ix_to_merge.any():
                if i>0:
                    ent[i] = ent[i-1]
                else: 
                    ent[i]  = np.log2(self.N) 
                if return_zones:
                    if i>0: 
                        Zones[:,i] = Zones[:,i-1]
                    else: 
                        Zones[:,i] = np.arange(self.N)
            else: 
                ent[i] = uf.merge_all(ELP[ix_to_merge,:2].astype(int))[-1]
                if return_zones:
                    Zones[:,i] = [uf.find(c) for c in range(self.N)] 
        
        # update self attribytes
        self.pbond_vec = pbond_vec[1:]
        if permute: 
            self.ent_perm = ent
        else: 
            self.ent_real = ent
        if return_zones: 
            if not permute:
                self.Zones = Zones.astype(int) 
            else: 
                self.Zones_perm = Zones.astype(int)
            return ent, Zones

        return ent

    def percolation(self, return_zones = True, verbose = False):
        self.with_zones = return_zones
        start=time.time()
        if verbose: 
            print(f"Calc edges from XY")
        self.ELKfull = edge_list_from_XY_with_k(self.XY, self.maxK)
        if verbose: 
            print(f"Done: Calc edges from XY - time: {time.time()-start:.0f}")
            print(f"Starting real percolation return_zones = {return_zones}")
  
        self.bond_percolation(permute = False, return_zones = return_zones, verbose = verbose)
        if verbose:
            print(f"Done with real percolation - time: {time.time()-start:.0f}")
            print(f"Starting perm percolation - return_zones = {return_zones}")  

        self.bond_percolation(permute = True, return_zones = return_zones, verbose = verbose)
        if verbose:
            print(f"Done with perm percolation - time: {time.time()-start:.0f}")

        if return_zones:
            if verbose:
                print(f"Starting to calc type entropy") 
            self.ent_type_real = np.zeros((self.Zones.shape[1],self.n_types))
            self.ent_type_perm = np.zeros((self.Zones_perm.shape[1],self.n_types))

            for t,typ in enumerate(self.unq_types): 
                for i in range(self.Zones.shape[1]): 
                    self.ent_type_real[i,t] = list_entropy(self.Zones[self.type_vec==typ,i])
                    self.ent_type_perm[i,t] = list_entropy(self.Zones_perm[self.type_vec_perm==typ,i])
            if verbose:
                print(f"Done with calc type entropy - time: {time.time()-start:.0f}")

    def score(self):
        dent = self.ent_perm-self.ent_real
        scr = np.trapz(np.abs(dent), x=self.pbond_vec, axis=0)
        return scr
    
    def plot_entropy_by_type(self, ax = None):
        if not self.with_zones: 
            raise ValueError(f"Zones were not calcualted, so cant show entropy_by_type")
        if ax is None: 
            plt.figure()
            ax = plt.gca()

        ax.plot(self.pbond_vec, self.ent_type_real, '.-')
        ax.axhline(y=np.log2(self.N), color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Pbond')
        ax.set_ylabel('Entropy')


    def plot_entropy_vs_perm(self,ax = None): 
        if ax is None: 
            plt.figure()
            ax = plt.gca()
        
        ax.plot(self.pbond_vec, self.ent_real, '-')
        ax.plot(self.pbond_vec, self.ent_perm, '--')
        ax.axhline(y=np.log2(self.N), color='black', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Entropy')
        ax.set_xlabel('Pbond')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1.1*np.log2(self.N)])
        ax.set_title(f"Score = {self.score():.2f}")


    def show_all(self):
        fig,axs = plt.subplots(1,2,figsize=(10,5))
        axs = axs.T
        self.plot_entropy_by_type(ax=axs.flatten()[0])
        self.plot_entropy_vs_perm(ax=axs.flatten()[1])
        plt.show()

    def create_animation_vs_perm(self,ani_pth,filename, save_gif = True):
        # create colormap with a color per cells, random, and fixed
        cmap = plt.get_cmap('nipy_spectral')
        rgb_cells = cmap(np.linspace(0, 1, len(self.type_vec)))
        rgb_cells = rgb_cells[np.random.permutation(len(self.type_vec)),:]
        cmap = mcolors.ListedColormap(rgb_cells)

        ani_pth = 'Animations'
        os.makedirs(ani_pth, exist_ok=True)

        fig = plt.figure(figsize=(30, 8))
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[: , 0])
        ax2 = fig.add_subplot(gs[: , 1])
        ax3 = fig.add_subplot(gs[0 , 2])
        ax4 = fig.add_subplot(gs[1 , 2])

        def animate(i):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.scatter(self.XY[:,0],-self.XY[:,1],s=1,c=self.Zones[:,i],alpha=0.5,cmap=cmap)
            ax2.scatter(self.XY[:,0],-self.XY[:,1],s=1,c=self.Zones_perm[:,i],alpha=0.5,cmap=cmap)
            ax1.set_title('Percolation - real', color = 'blue')
            ax2.set_title('Percolation - perm', color = 'orange')
            
            self.plot_entropy_by_type(ax = ax3)
            ax3.axvline(x=self.pbond_vec[i], color='r', linestyle=':')
            
            self.plot_entropy_vs_perm(ax = ax4)
            ax4.axvline(x=self.pbond_vec[i], color='r', linestyle=':')

        ani = FuncAnimation(fig, animate, frames=self.Zones.shape[1], repeat=False)

        html = HTML(ani.to_jshtml())
        filename = f"{ani_pth}/division_isozones"
        if filename is not None: 
            with open(filename + '.html', 'w') as f:
                f.write("<html>\n")
                f.write("<head>\n")
                f.write("<title>Animation</title>\n")
                f.write("</head>\n")
                f.write("<body>\n")
                f.write(html.data)
                f.write("</body>\n")
                f.write("</html>")
            if save_gif: 
                ani.save(filename + '.gif', writer='imagemagick')


def merge_nested_clusters(GPmat, top_down = True): 

    Ncells = sum(GPmat[i,0].N for i in range(GPmat.shape[0]))

    # create the type matrix from GP objects: 
    type_vecs=list()
    for row in GPmat: 
        type_vecs.append(np.hstack([gp.type_vec[:,np.newaxis] for gp in row]).astype(int))
    type_int_mat = np.vstack(type_vecs)
    Ntypes_per_lvl = (type_int_mat.max(axis=0)+1).astype(int)

    # now extract the type entropy from each GO object to create the overall type x entropy across all possible types (mutliple levels)
    
    # Init all empty matrices inside the list
    delta_ent_per_pbond_type_lvl = [None] * GPmat.shape[1]
    type_freq_lvl = [None] * GPmat.shape[1]

    for lvl in range(len(delta_ent_per_pbond_type_lvl)):
        delta_ent_per_pbond_type_lvl[lvl] = np.zeros((Ntypes_per_lvl[lvl],GPmat.shape[0],len(GPmat[0,0].pbond_vec)))
        type_freq_lvl[lvl] = np.zeros((Ntypes_per_lvl[lvl],GPmat.shape[0]))

    # extract the actual values
    for lvl in range(GPmat.shape[1]):
        for sec in range(GPmat.shape[0]): 
            types_in_section,type_freq = np.unique(GPmat[sec,lvl].type_vec,return_counts=True)
            type_freq_lvl[lvl][types_in_section.astype(int),sec] = type_freq/Ncells
            dent = np.abs(GPmat[sec,lvl].ent_type_perm.T-GPmat[sec,lvl].ent_type_real.T) 
            delta_ent_per_pbond_type_lvl[lvl][types_in_section.astype(int),sec,:] = dent


    # update the type numbers (so they are continous and not restaring each lebvel)
    # then concatenate the entropys x pbond and frequencies
    # now each row of delta_ent_per_pbond correspond to a type from the continous numbering

    cmsm_type_mat = type_int_mat.copy()
    cmsm_type_mat[:,1:] = cmsm_type_mat[:,1:]+np.cumsum(Ntypes_per_lvl[np.newaxis,:-1])
    cmsm_type_mat = cmsm_type_mat.astype(int)
    delta_ent_per_pbond = np.concatenate(delta_ent_per_pbond_type_lvl,axis=0)
    cmsm_type_freq = np.concatenate(type_freq_lvl,axis=0)

    def score(curr_ix):
        dent_by_type_by_section_for_curr_types = delta_ent_per_pbond[curr_ix,:,:]
        freq_by_type_by_section_for_curr_types = cmsm_type_freq[curr_ix,:,np.newaxis]
        dent_by_section_summed_over_types = (dent_by_type_by_section_for_curr_types * freq_by_type_by_section_for_curr_types).sum(axis=0)
        dent_by_pbond = dent_by_section_summed_over_types.sum(axis=0)
        # scr = np.trapz(dent_by_pbond,GPmat[0,0].pbond_vec)
        scr = dent_by_pbond.mean()

        return scr

    # now create the mapping dict so that each type (key) has the list of subtype under it (values)
    # Initialize an empty dictionary
    adj_dict = {}

    # Iterate over the rows of the matrix
    for row in cmsm_type_mat:
        # Iterate over each element in the row
        for i in range(len(row) - 1):
            # If the element is not already a key in the dictionary, add it with an empty list as its value
            if row[i] not in adj_dict:
                adj_dict[row[i]] = []
            # If the next element is not already in the list of adjacent values for the current element, add it
            if row[i+1] not in adj_dict[row[i]]:
                adj_dict[row[i]].append(row[i+1])

    # ok, now we have everything we need, we can loop over all types to see if it's worth expanding any of them: 
    if top_down: 
        curr_ix = list(np.unique(cmsm_type_mat[:,0]))
    else: #i.e. bottom up
        curr_ix = list(np.unique(cmsm_type_mat[:,-1]))

    best_score = score(curr_ix)

    improved = True
    if top_down: 
        while improved:
            improved = False
            for i in range(len(curr_ix)):
                if curr_ix[i] in adj_dict:
                    new_ix = curr_ix[:i] + curr_ix[i+1:] + adj_dict[curr_ix[i]]
                    # new_score = np.sum(delta_ent_per_pbond[new_ix, :] * cmsm_type_freq[new_ix, :],axis=0).mean()
                    new_score = score(new_ix)
                    if new_score > best_score:
                        curr_ix = new_ix
                        best_score = new_score
                        improved = True
        type_vec = cmsm_type_mat[np.isin(cmsm_type_mat,curr_ix)]
    else: #i.e. bottom up
        keys_per_level = [np.unique(cmsm_type_mat[:, lvl]) for lvl in range(5)]

        for lvl in range(3, -1, -1):
            print(f"starting level: {lvl}")
            improved = True
            for key in keys_per_level[lvl]:  # Loop over keys at the current level
                subtypes = adj_dict[key]
                if all(subtype in curr_ix for subtype in subtypes): 
                    new_ix = [ix for ix in curr_ix if ix not in subtypes] + [key]
                    new_score = score(new_ix)
                    if new_score > best_score:
                        curr_ix = new_ix
                        best_score = new_score
                        improved = True
        mask = np.isin(cmsm_type_mat,curr_ix)
        new_mask = np.full(mask.shape, False)
        last_true_indices = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1) - 1
        new_mask[np.arange(mask.shape[0]), last_true_indices] = True                        
        type_vec = cmsm_type_mat[new_mask]

    return curr_ix,best_score,type_vec

    


class ToyGraph:
    def __init__(self, Nside, pattern = 'random', n_types = None, perm_xy = True, **kwargs,):
        self.N = Nside**2
        self.X, self.Y = np.meshgrid(np.arange(Nside), np.arange(Nside))
        self.XY = np.hstack((self.X.flatten()[:,np.newaxis], self.Y.flatten()[:,np.newaxis])).astype(float)


        if pattern == 'random': 
            self.frac = kwargs.get('frac', 0.5)
            self.type_vec = np.zeros(self.N)
            indices = np.random.choice(np.arange(self.N), int(self.frac * self.N), replace=False)
            self.type_vec[indices] = 1
            self.G = np.zeros((Nside,Nside))
            rows, cols = np.unravel_index(indices, (Nside, Nside))
            self.G[rows,cols] = 1
        elif pattern == 'squares': 
            self.G = np.zeros((Nside,Nside))
            self.border = kwargs.get('border', 0)
            self.square_side = kwargs.get('size', int(Nside/4))
            ix1=np.arange(self.border,self.border + self.square_side)
            ix2=np.arange(Nside - self.border - self.square_side,Nside - self.border)
            self.G[ix1[:, None],ix1] = 1
            self.G[ix1[:, None],ix2] = 1
            self.G[ix2[:, None],ix1] = 1
            self.G[ix2[:, None],ix2] = 1
            self.type_vec = self.G.ravel()
            self.frac = self.G.mean()
        elif pattern == 'grid': 
            self.G = np.zeros((Nside,Nside))
            self.num_squares = kwargs.get('num_squares', 4)
            square_size = Nside // self.num_squares
            cnt=-1
            for i in range(self.num_squares):
                for j in range(self.num_squares):
                    cnt += 1
                    self.G[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = cnt
            self.type_vec = self.G.ravel()
            self.frac = self.G.mean()

        if perm_xy:
            prm = np.random.permutation(np.arange(self.N))
        else: 
            prm = np.arange(self.N)

        self.ordr = np.argsort(prm)
        self.XY=self.XY[prm,:]
        self.type_vec=self.type_vec[prm]

        self.n_types = len(np.unique(self.type_vec))

    def update_type(self,type_vec): 
        self.type_vec = type_vec
        self.G = type_vec.reshape(self.G.shape)
        self.n_types = len(np.unique(self.type_vec))

    def show_graph(self,ax = None):
        ax = sns.heatmap(self.G,cbar=False,ax=ax)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def show_zones(self, ZoneVec, ax = None, clrs = None):
        ZoneVec = ZoneVec[self.ordr]
        zones = np.reshape(ZoneVec,(np.sqrt(self.N).astype(int),np.sqrt(self.N).astype(int)))
        cmap = plt.cm.get_cmap('nipy_spectral', len(np.unique(ZoneVec)))
        if clrs is None: 
            clrs = cmap(np.linspace(0, 1, len(np.unique(ZoneVec))))
            np.random.shuffle(clrs)
        cmap = mcolors.ListedColormap(clrs)
        ax = sns.heatmap(zones, cmap=cmap,cbar=False,ax = ax,vmin=0,vmax=len(clrs))
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def show_all(self, ZoneVec):
        fig,axs = plt.subplots(1,2,figsize=(10,5))
        axs = axs.T
        self.show_graph(ax=axs.flatten()[0])
        self.show_zones(ZoneVec,ax=axs.flatten()[1])
        plt.show()

    def create_animation(self, Zones, filename = None):
        fig, ax = plt.subplots()

        all_rgbs = list()
        cmap = plt.cm.get_cmap('nipy_spectral', self.N)

        rgb_real = cmap(np.linspace(0, 1, self.N))
        np.random.shuffle(rgb_real)

        def animate(i):
            ax.clear()    
            self.show_zones(Zones[:,i], ax=ax, clrs = rgb_real)

        ani = animation.FuncAnimation(fig, animate, frames=Zones.shape[1], interval=200)
        if filename is not None:
            ani.save(filename + '.gif', writer='imagemagick')

        html = HTML(ani.to_jshtml())
        if filename is not None: 
            with open(filename + '.html', 'w') as f:
                f.write("<html>\n")
                f.write("<head>\n")
                f.write("<title>Animation</title>\n")
                f.write("</head>\n")
                f.write("<body>\n")
                f.write(html.data)
                f.write("</body>\n")
                f.write("</html>")
                
        return html