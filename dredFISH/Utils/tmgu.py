import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import igraph
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import matplotlib.animation as animation
from IPython.display import HTML

import sys
import time
sys.path.append('/home/rwollman/MyProjects/AH/Repos/dredFISH')
from dredFISH.Utils.ConnectedComponentEntropy import ConnectedComponentEntropy

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

class GraphPercolation:
    def __init__(self, XY, type_vec, maxK = None):
        self.N = XY.shape[0]
        self.XY = XY
        self.type_vec = type_vec
        self.maxK = maxK
        self.n_types = len(np.unique(type_vec))

        if maxK is None: 
            self.maxK = self.N-1

    def save(self,filename): 
        np.savez(filename,XY = self.XY,
                         type_vec = self.type_vec,
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
        self.Zones = dump['Zones']
        self.Zones_perm = dump['Zones_perm']
        self.ent_real = dump['ent_real']
        self.ent_perm = dump['ent_perm']
        self.pbond_vec = dump['pbond_vec']
        self.ent_type_real = dump['ent_type_real']
        self.ent_type_perm = dump['ent_type_perm']
        self.maxK = dump['maxK']
        self.n_types = len(np.unique(self.type_vec))
        self.N = len(self.type_vec)

    def calc_ELKexp(self, permute=False): 
        ELK = self.ELKfull
        if permute: 
            type_vec = np.random.permutation(self.type_vec)
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
            print(f"Starting real percolation - return_zones = {return_zones}") 

        self.bond_percolation(permute = False, return_zones= return_zones, verbose = verbose)
        if verbose:
            print(f"Done with real percolation - time: {time.time()-start:.0f}")
            print(f"Starting perm percolation - return_zones = {return_zones}")  

        self.bond_percolation(permute = True, return_zones= return_zones, verbose = verbose)
        if verbose:
            print(f"Done with perm percolation - time: {time.time()-start:.0f}")

        if return_zones:
            if verbose:
                print(f"Starting to calc type entropy") 
            self.ent_type_real = np.zeros((self.Zones.shape[1],self.n_types))
            self.ent_type_perm = np.zeros((self.Zones_perm.shape[1],self.n_types))

            for t in range(self.n_types): 
                for i in range(self.Zones.shape[1]): 
                    self.ent_type_real[i,t] = list_entropy(self.Zones[self.type_vec==t,i])
                    self.ent_type_perm[i,t] = list_entropy(self.Zones_perm[self.type_vec==t,i])
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