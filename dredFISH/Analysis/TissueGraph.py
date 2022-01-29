"""TissueGraph module. 

The module contains main classes required for maximally infomrative biocartography (MIB)
MIB includes the following key classes: 

TissueMultiGraph: the key organizing class used to create and manage graphs across layers (hence multi)

TissueGraph: Each layers (cells, zones, microenv, regions) is defined using a graph. 

"""

# dependeices
from igraph import *

import pynndescent 
from scipy.spatial import Delaunay,Voronoi
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, fcluster, cut_tree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import entropy

from sklearn.neighbors import NearestNeighbors

from numpy.random import default_rng
rng = default_rng()

from collections import Counter

import numpy as np
import pandas as pd
import torch

import itertools
import dill as pickle

from scipy.spatial.distance import jensenshannon, cdist, pdist, squareform

from pyemd import emd

# for debuding mostly
import warnings
import time
from IPython import embed

# to create geomtries
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString

from dredFISH.Visualization.cell_colors import *
from dredFISH.Visualization.vor import voronoi_polygons, bounding_box

##### Some simple accessory funcitons
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

def treenomialJSD(Q1,Q2 = None):
    """
    calcualtes mean JSD of bernulli represenation of taxonomy tree. 
    if Q2 is None, return pairwise distances fo all Q1 pairs
    """
    # in cases where Q2 is none, return pairwise distance of all Q1 elements.  
    if Q2 is None:
        print("calculating pairwise distances")
        cmb = np.array(list(itertools.combinations(np.arange(Q1.shape[0]), r=2)))
        D = treenomialJSD(Q1[cmb[:,0],:],Q1[cmb[:,1],:])
        return D
    
    Q1 = torch.from_numpy(Q1)
    Q2 = torch.from_numpy(Q2)

    # calculate treenomialJSD between two matrices
    M = (Q1+Q2)/2

    # Q includes all the p and 1-p for each tree branch, so already doing "double" 
    KL1 = Q1*(torch.log2(Q1/M))
    KL1[Q1==0]=0
    KL2 = Q2*(torch.log2(Q2/M))
    KL2[Q2==0]=0
    JSD = (KL1+KL2)
    
    JSDavg = torch.mean(JSD,axis=1)
    # we get some numerical issues with values being -eps, so zero them out
    JSDavg[JSDavg<0] = 0    
    return(JSDavg.cpu().detach().numpy())


def dist_emd(E1,E2,Dsim):
    
    if E2 is None: 
        cmb = np.array(list(itertools.combinations(np.arange(E1.shape[0]), r=2)))
        D = dist_emd(E1[cmb[:,0],:],E1[cmb[:,1],:],Dsim)
    else:
        sum_of_rows = E1.sum(axis=1)
        E1 = E1 / sum_of_rows[:, None]
        E1 = E1.astype('float64')
        sum_of_rows = E2.sum(axis=1)
        E2 = E2 / sum_of_rows[:, None]
        E2 = E2.astype('float64')
        D = np.zeros(E1.shape[0])
        Dsim = Dsim.astype('float64')
        for i in range(E1.shape[0]):
            e1=E1[i,:]
            e2=E2[i,:]
            D[i] = emd(e1,e2,Dsim)
    
    return D

# def dist_nn(IX1,IX2 = None):
#     # perform nn search (using accuracy x number of neighbors to improve accuracy)
#     knn = pynndescent.NNDescent(X,n_neighbors = n_neighbors_with_extras ,metric=metric,diversify_prob=0.5,metric_kwds = metric_kwds)

#     # get indices and remove self. 
#     (indices,distances) = knn.neighbor_graph
    

def dist_jsd(M1,M2 = None):
    if M2 is None: 
        D = pdist(M1,metric = 'jensenshannon')
    else: 
        D = jensenshannon(M1,M2,base=2,axis=1)
        
    return D

def buildgraph(X,n_neighbors = 15,metric = None,accuracy = 4,metric_kwds = {}):
    # different methods used to build nearest neighbor type graph. 
    # the algo changes with the metric. 
    #     For JSD uses KD-tree
    #     for  precomputed, assumes X is output of squareform(pdist(X)) in format and finds first n_neighbors directly
    #     all other metrics, uses PyNNdescent for fast knn computation of large range of metrics. 
    # 
    # accuracy params is only used in cases where pynndescent is used and tries to increase accuracy by calculating 
    # more neighbors than needed. 
    
    
    if metric is None:
        raise ValueError('metric was not specified')
    
    # checks if we have enough rows 
    n_neighbors = min(X.shape[0]-1,n_neighbors)
    
    if metric == 'jsd':
        knn = NearestNeighbors(n_neighbors = n_neighbors,metric = jensenshannon)
        knn.fit(X)
        (distances,indices) = knn.kneighbors(X, n_neighbors = n_neighbors)
        
    elif metric == 'precomputed':
        indices = np.zeros((X.shape[0],n_neighbors))
        for i in range(X.shape[0]):
            ordr = np.argsort(X[i,:])
            if np.any(ordr[:n_neighbors]==i):
                indices[i,:]=np.setdiff1d(ordr[:n_neighbors+1],i)
            else:
                indices[i,:]=ordr[:n_neighbors]
    else:
        # update number of neighbors (x accuracy) to increase accuracy
        n_neighbors_with_extras = min(X.shape[0]-1,n_neighbors * accuracy)

        # perform nn search (using accuracy x number of neighbors to improve accuracy)
        knn = pynndescent.NNDescent(X,n_neighbors = n_neighbors_with_extras ,metric=metric,diversify_prob=0.5,metric_kwds = metric_kwds)

        # get indices and remove self. 
        (indices,distances) = knn.neighbor_graph
       
    indices=indices[:,1:n_neighbors+1]
    
    id_from = np.tile(np.arange(indices.shape[0]),indices.shape[1])
    id_to = indices.flatten(order='F')
    
    # build graph
    edgeList = np.vstack((id_from,id_to)).T
    G = Graph(n=X.shape[0], edges=edgeList)
    G.simplify()
    return G

###### Main TissueGraph classes

class TissueMultiGraph: 
    """
    TissueMultiGraph - Main class that manages the creation of multi-layer graph representation of tissues. 
    
    main attributes: 
        Layers = where we store specific spatial representation of cells, zones, microenvironments, and regions as graphs
        Geoms = geometrical aspects of the TMG required for plotting. 
        Views = data and methods for plotting maps. Views are defined with a detailed OOP class structure. 

    main methods:

        Misc: 
            * save (and load during __init__ if fullfilename is provided)

        Create: 
            * create_cell_and_zone_layers - creates the first pair of Cell and Zone layers of the TMG
            * create_communities_and_region_layers - create the second pair of layers (3 and 4) of the TMG
            * add_geoms
            * add_view

        Query: 
            * map_to_cell_level -
            * find_max_edge_level - 
            * N
            * Ntypes
    """
    def __init__(self,fullfilename = None):
        # init to empty
        self.Layers = list()
        self.Geoms = {}
        self.Views = {}
        self.cell_attributes = pd.DataFrame()
        
        if fullfilename is not None:
            
            # load from file
            TMGload = pickle.load(open(fullfilename,'rb'))
            
            # key data is stored in the Layers list
            # Layers should always be saved, 
            self.Layers = TMGload.Layers
            
            # Geoms / Views 
            if hasattr(TMGload,'Geoms'): 
                self.Geoms = TMGload.Geoms
                
            if hasattr(TMGload,'Views'): 
                self.Views = TMGload.Views
                
            if hasattr(TMGload,'cell_attributes'):
                self.cell_attributes = TMGload.cell_attributes
            

        return None
    
    def save(self,fullfilename):
        """ pickle and save
        """
        pickle.dump(self,open(fullfilename,'wb'),recurse=True)
        
    def create_cell_and_zone_layers(self,XY,PNMF,celltypes = None): 
        """
        Creating cell and zone layers. 
        Cell layer is unique as it's the only one where spatial information is directly used with Voronoi
        Zone layer is calculated based on optimization of cond entropy betweens zones and types. 
        """
        # creating first layer - cell tissue graph
        TG = TissueGraph()
        TG.layer_type = "cells"
        
        # build two key graphs
        TG.build_spatial_graph(XY)
        TG.FG = buildgraph(PNMF,metric = 'cosine')

        if celltypes is None:
            # cluster cell types optimally
            celltypes = TG.multilayer_Leiden_with_cond_entropy()
        
        # add types and key data
        TG.Type = celltypes
        TG.feature_mat = PNMF
        
        # add layer
        self.Layers.append(TG)
        
        # contract and create the Zone graph
        NewLayer = self.Layers[0].contract_graph(celltypes)
        self.Layers.append(NewLayer)
        
        # add feature graph - for zones, this graph will have one node per type. 
        self.Layers[1].FG = self.Layers[-2].FG.copy()
        self.Layers[1].FG.contract_vertices(celltypes)
        self.Layers[1].FG.simplify()
        self.Layers[1].layer_type = 'isozones'
        
        df = pd.DataFrame(data = PNMF)
        df['type']=self.Layers[1].UpstreamMap
        self.Layers[1].feature_mat = np.array(df.groupby(['type']).mean())
        df = pd.DataFrame(data = PNMF)
        df['type']=celltypes
        self.Layers[1].feature_type_mat = np.array(df.groupby(['type']).mean())
        
        self.Layers[1].Dtype = squareform(pdist(self.Layers[1].feature_type_mat,metric = 'cosine'))
        self.Layers[1].calc_type2()
        
        return None
    
    def find_heterozones_based_on_coherence(self, ordr = 3): 
        """
        Creating heterozones and neighborhood layers based on local coherence. 
        Initial guess of heterozone locaiton
        """
        # find existing environment 
        Env = self.Layers[1].extract_environments(ordr=ordr)
        
        # perform watershed with seeds as "peaks" in NodeWeight graph
        
        # first create the landscape - calc coherence and do local averaging (and nan removal)
        EdgeWeight,NodeWeight = self.Layers[1].calc_graph_env_coherence(Env,dist_jsd)
        smooth_ordr = 3
        smooth_kernel=np.array([0.4,0.3,0.2,0.1])
        NodeWeight = self.Layers[1].graph_local_avg(NodeWeight,ordr = smooth_ordr,kernel = smooth_kernel)
        
        self.cell_attributes['coherence'] = self.map_to_cell_level(1,NodeWeight)
        
        HoodId,DistToPeak = self.Layers[1].watershed(NodeWeight)
        self.cell_attributes['watersheds'] = self.map_to_cell_level(1,DistToPeak)
        return HoodId
    
    def create_heterozone_and_neighborood_layers(self,hz_id,min_neigh_size = 10,ordr = 3):
        """
        Function gets vector of heterozones id (size of self.N[1]) and uses it 
        to add heterozones and neighborhoods to the TMG
        """
    
        # add heterozone layer: 
        HZlayer = self.Layers[1].contract_graph(hz_id)
        if len(self.Layers)==2:
            self.Layers.append(HZlayer)
        else: 
            self.Layers[2] = HZlayer
        self.Layers[2].layer_type = 'heterozones'
        
        # calculate true environments vector for all communities: 
        hz_env = self.Layers[1].extract_environments(typevec = hz_id)
        
        # save hz enz as feature mat data
        self.Layers[2].feature_mat = hz_env
                     
        # calcualte pairwise distances between environments using treenomial_dist and build feature graph
        D = dist_jsd(hz_env)
        Dsqr = squareform(D)
        self.Layers[2].FG = buildgraph(Dsqr,metric = 'precomputed')

        # cluster and give each heterozone it's type
        self.Layers[2].Type = self.Layers[2].multilayer_Leiden_with_cond_entropy()
        
        # create neighborhood layers through graph contraction
        NeihLayers = self.Layers[2].contract_graph()
        if len(self.Layers)==3:
            self.Layers.append(NeihLayers)
        else: 
            self.Layers[3] = NeihLayers
        
        # replace type of heterozones that are very small with local neighborhood type (labelprop) 
        self.fill_holes(3,min_neigh_size)

        # fix heterozone type vector as types might have changed after removing small gaps
        self.Layers[2].Type = self.Layers[3].Type[self.Layers[3].UpstreamMap]
        
        # create the feature_mat and feature_type_mat 
        regions_types_mapped_to_cells = self.map_to_cell_level(3)
        feature_type_mat = self.Layers[0].extract_environments(typevec = regions_types_mapped_to_cells)
        sum_of_rows = feature_type_mat.sum(axis=1)
        feature_type_mat = feature_type_mat / sum_of_rows[:, None]
        self.Layers[3].feature_type_mat = feature_type_mat
        
    def refine_heterozone_and_neighborood_layers(self):

        # only keep edges that are of across two neighborhood types
        Env_hz = self.Layers[2].feature_mat.copy()

        # test for each one of them if flipping it increases the JSD distance 
        # between the heterozone it used to belond to and the neighborhood type. 

        flipped = list();
        n_flipped = np.zeros(15);
        start=time.time()
        
        for j in range(len(n_flipped)):
            ix = self.Layers[3].UpstreamMap
            ix = ix[self.Layers[2].UpstreamMap]
            isozone_neigh_types = self.Layers[3].Type[ix]
            # find iso-zones that are on the border between neighborhoods 
            EL = self.Layers[1].SG.get_edgelist()
            EL = np.array(EL).astype("int")
            EL = EL[isozone_neigh_types[EL[:,0]] != isozone_neigh_types[EL[:,1]],:]
            
             # remove from EL isozones that only have 1 neighbor across types
            iz_ids,cnt = np.unique(np.hstack((EL[:,0],EL[:,1])),return_counts=True)
            iz_ids = iz_ids[cnt>2]
            # remove iz that were already flipped
            iz_ids = np.setdiff1d(iz_ids,flipped)
            to_keep = np.logical_or(np.isin(EL[:,0],iz_ids),np.isin(EL[:,1],iz_ids))
            EL = EL[to_keep,:]

            Env_neigh = self.Layers[3].feature_type_mat[TMG.Layers[2].Type,:]
            Dhz_ne = dist_jsd(Env_hz,Env_neigh)

            for i in range(EL.shape[0]): 
                # check which os the two isozones could be flipped, if both, choose randomly
                if np.isin(EL[i,0],iz_ids) and not np.isin(EL[i,1],iz_ids): 
                    lr = np.array([0,1])
                elif np.isin(EL[i,1],iz_ids) and not np.isin(EL[i,0],iz_ids):
                    lr = np.array([1,0])
                else:        
                    # permute edge order (it's undirected and who is 0 and who is 1 is arbitrary)
                    lr = np.random.permutation(2)

                # keep all indexes for the pair in lr
                iz = EL[i,lr]
                iz_types = self.Layers[1].Type[EL[i,lr]]

                # get the two env for lr heterozones
                hz = self.Layers[2].UpstreamMap[iz]
                hz_types = self.Layers[2].Type[hz]
                dz_env = Env_hz[hz,:]

                # flip (move from a type from 0 to 1 (of lr))
                dz_env[0,iz_types[0]] -= 1
                dz_env[1,iz_types[0]] += 1

                # calculate JSD for both environments
                newJSD = cdist(dz_env,self.Layers[3].feature_type_mat[hz_types,:],metric= 'jensenshannon')
                JSDsum = newJSD.flatten()[[0,3]].sum()

                # update neighborhoods
                if JSDsum < Dhz_ne[hz].sum():
                    # update heterozone composition by changing mapping between isozones and heterozones
                    self.Layers[2].UpstreamMap[iz[0]] = self.Layers[2].UpstreamMap[iz[1]]
                    Env_hz[hz,:]=dz_env
                    flipped.append(iz[0])

            print(f"{j}: {time.time()-start:.2f} flipped: {len(flipped)}")
            n_flipped[j] = len(flipped)
            if j>0 and n_flipped[j]- n_flipped[j-1] < 10: 
                break
            
        # bookeeping on hz ids: 
        CG = self.Layers[1].contract_graph(self.Layers[2].UpstreamMap)
        Id = np.arange(CG.N)
        hz_id = Id[CG.UpstreamMap]

        return(hz_id)

        
    def finalize_heterozone_and_neighborhood_layers(self):
        """
        Adds a few more attributes to neighnborhood layer: feature_mat and type2
        """
        regions_mapped_to_cells = self.map_to_cell_level(3,VecToMap = np.arange(self.Layers[3].N))
        feature_mat = self.Layers[0].extract_environments(typevec = regions_mapped_to_cells)
        sum_of_rows = feature_mat.sum(axis=1)
        feature_mat = feature_mat / sum_of_rows[:, None]
        self.Layers[3].feature_mat = feature_mat
        
        # add feature graph - for regions this graph will have one node per type. 
        D = dist_emd(self.Layers[3].feature_type_mat,None,self.Layers[1].Dtype)
        self.Layers[3].Dtype = squareform(D)
        self.Layers[3].FG = buildgraph(self.Layers[3].Dtype,metric = 'precomputed',n_neighbors = 5)
        
        self.Layers[3].calc_type2()
        self.Layers[3].layer_type = 'neighborhood'
    
    def create_communities_and_region_layers(self,min_neigh_size = 10,ordr = 3,metric_kwds = {}):
        """
        Creating heterozones and neighborhood layers. 
        heterozone layer is created based on watershed using local env coherence. 
        Region layer is calculated based on optimization of cond entropy betweens community and regions. 
        """
        # find existing environment 
        Env = self.Layers[1].extract_environments(ordr=ordr)
        
        # add entropy attribute to all cells
        self.cell_attributes['local_entropy'] = self.map_to_cell_level(1,entropy(Env,axis=1))
        
        # perform watershed with seeds as "peaks" in NodeWeight graph
        
        # first create the landscape - calc coherence and do local averaging (and nan removal)
        EdgeWeight,NodeWeight = self.Layers[1].calc_graph_env_coherence(Env,dist_jsd)
        smooth_ordr = 3
        smooth_kernel=np.array([0.4,0.3,0.2,0.1])
        NodeWeight = self.Layers[1].graph_local_avg(NodeWeight,ordr = smooth_ordr,kernel = smooth_kernel)
        
        self.cell_attributes['coherence'] = self.map_to_cell_level(1,NodeWeight)
        
        HoodId,DistToPeak = self.Layers[1].watershed(NodeWeight)
        self.cell_attributes['watersheds'] = self.map_to_cell_level(1,DistToPeak)
                
        # add heterozone layer: 
        NewLayer = self.Layers[1].contract_graph(HoodId)
        self.Layers.append(NewLayer)
        self.Layers[2].layer_type = 'heterozones'
        
        # calculate true environments vector for all communities: 
        WatershedEnvs = self.Layers[1].extract_environments(typevec = HoodId)
        
        # save in taxonomy data
        self.Layers[-1].feature_mat = WatershedEnvs
                     
        # calcualte pairwise distances between environments using treenomial_dist and build feature graph
        D = dist_jsd(WatershedEnvs)
        Dsqr = squareform(D)
        self.Layers[-1].FG = buildgraph(Dsqr,metric = 'precomputed')

        # TODO - check alternatives
        # D = self.Layers[-2].TX.treenomial_dist(WatershedEnvs)
        # OR 
        # self.Layers[-1].FG = buildgraph(Dsqr,metric = 'kantorovich',metric_kwds = metric_kwds)
        
        # cluster and create regions graph
        commtypes = self.Layers[-1].multilayer_Leiden_with_cond_entropy()
        
        # add types to comm/microenv layer and than contract to get regions
        self.Layers[-1].Type = commtypes
        
        NewLayer = self.Layers[-1].contract_graph(commtypes)
        self.Layers.append(NewLayer)
        
        # replace type of heterozones that are very small with local type (labelprop) 
        self.fill_holes(3,min_neigh_size)

        # fix heterozone type vector as types might have changed after removing small gaps
        self.Layers[-2].Type = self.Layers[-1].Type[self.Layers[-1].UpstreamMap]
        
        # create the feature_mat and feature_type_mat 
        regions_types_mapped_to_cells = self.map_to_cell_level(3)
        feature_type_mat = self.Layers[0].extract_environments(typevec = regions_types_mapped_to_cells)
        sum_of_rows = feature_type_mat.sum(axis=1)
        feature_type_mat = feature_type_mat / sum_of_rows[:, None]
        self.Layers[-1].feature_type_mat = feature_type_mat
        
        regions_mapped_to_cells = self.map_to_cell_level(3,VecToMap = np.arange(self.Layers[3].N))
        feature_mat = self.Layers[0].extract_environments(typevec = regions_mapped_to_cells)
        sum_of_rows = feature_mat.sum(axis=1)
        feature_mat = feature_mat / sum_of_rows[:, None]
        self.Layers[-1].feature_mat = feature_mat
        
        # add feature graph - for regions this graph will have one node per type. 
        D = dist_emd(self.Layers[3].feature_type_mat,None,self.Layers[1].Dtype)
        self.Layers[3].Dtype = squareform(D)
        self.Layers[3].FG = buildgraph(self.Layers[3].Dtype,metric = 'precomputed',n_neighbors = 5)
        
        # PyNNDescent implementation of EMD (kantorovich) is slower than pyemd 
        # might be worth going back to if for the approximate neighbor aspect, but for ~100 types it's x3 slower and doesn't give distance
        # matrix. 
        # self.Layers[-1].FG = buildgraph(self.Layers[3].feature_type_mat,
        #                                 n_neighbors = 5, 
        #                                 metric = 'kantorovich',
        #                                 metric_kwds={'cost' : self.Layers[1].Dtype})

        self.Layers[-1].calc_type2()
        self.Layers[-1].layer_type = 'neighborhood'

        
    def fill_holes(self,lvl_to_fill,min_node_size):

        # get types (with holes)
        region_types = self.Layers[lvl_to_fill].Type.copy()

        # mark all verticies with small node size
        region_types[self.Layers[lvl_to_fill].node_size < min_node_size]=-1
        fixed = region_types>-1

        # renumber region types in case removing some made numbering discontinuous
        unq,ix = np.unique(region_types[fixed],return_inverse=True)
        cont_region_types = region_types.copy()
        cont_region_types[fixed] = ix

        # map to level-0 to do the label propagation at the cell level: 
        fixed_celllvl = self.map_to_cell_level(lvl_to_fill,fixed)
        contregion_celllvl = self.map_to_cell_level(lvl_to_fill,cont_region_types)

        # fill holes through label propagation
        lblprp = self.Layers[0].SG.community_label_propagation(weights=None, initial=contregion_celllvl, fixed=fixed_celllvl)
        region_type_filled = np.array(lblprp.membership)

        # shrink indexes from level-0 to requested level-1 
        _,ix = self.map_to_cell_level(lvl_to_fill-1,return_ix=True)
        _, ix_zero_to_two = np.unique(ix, return_index=True)
        new_type = region_type_filled[ix_zero_to_two]

        # recreate the layer
        NewLayer = self.Layers[lvl_to_fill-1].contract_graph(new_type)
        self.Layers[lvl_to_fill] = NewLayer

                    
    def map_to_cell_level(self,lvl,VecToMap = None,return_ix = False):
        """
        Maps values to first layer of the graph, mostly used for plotting. 
        lvl is the level we want to map all the way to 
        """
        # if VecToMap is not supplied, will use the layer Type as default thing to map to lower levels
        if VecToMap is None: 
            VecToMap = self.Layers[lvl].Type.astype(np.int64)
        elif len(VecToMap) != self.Layers[lvl].N:
            raise ValueError("Number of elements in VecToMap doesn't match requested Layer size")
            
        # if needed (i.e. not already at cell level) expand indexing backwards in layers    
        if lvl>0:
            ix=self.Layers[lvl].UpstreamMap
            for i in np.arange(lvl-1,0,-1):
                ix=ix[self.Layers[i].UpstreamMap]
            VecToMap = VecToMap[ix]
        
        if return_ix: 
            return (VecToMap,ix)
        else: 
            return VecToMap
    
    def find_max_edge_level(self):
        """
        determine the maximal level that an edges between two cell still exists in, i.e. it was not contracted
        returns a dict with sorted edge tuples as keys and max level is values.   
        """
        
        # create edge list with sorted tuples (the geom convention)
        edge_list = list()
        for e in self.Layers[0].SG.es:
            t=e.tuple
            if t[0]>t[1]:
                t=(t[1],t[0])
            edge_list.append(t)
        
        np_edge_list = np.asarray(edge_list)
        max_edge_levels = np.zeros(len(edge_list))
        for lvl in range(1,len(self.Layers)):
            Vs = self.map_to_cell_level(lvl,np.arange(self.N[lvl]))
            max_edge_levels += Vs[np_edge_list[:,0]] == Vs[np_edge_list[:,1]]
            
        return dict(zip(edge_list,max_edge_levels))
    
    @property
    def N(self):
        return([L.N for L in self.Layers])
    
    @property
    def Ntypes(self):
        return([L.Ntypes for L in self.Layers])
    
    @property
    def cond_entropy(self):
        return([L.cond_entropy() for L in self.Layers])
    
    def add_geoms(self):
        """ 
        Creates the geometies needed (boundingbox, lines, points, and polygons) to be used in views to create maps. 
        """
        
        # Bounding box geometry 
        XY = self.Layers[0].XY
        diameter, bb = bounding_box(XY, fill_holes=False)
        self.Geoms['BoundingBox'] = bb

        # Polygon geometry 
        # this geom is just the voronoi polygons
        # after intersection with the bounding box
        # if the intersection splits a polygon into two, take the one with largest area
        vor = Voronoi(XY)
        vp = list(voronoi_polygons(vor, diameter))
        vp = [p.intersection(self.Geoms['BoundingBox']) for p in vp]
        
        verts = list()
        for i in range(len(vp)):
            if isinstance(vp[i],MultiPolygon):
                allparts = [p.buffer(0) for p in vp[i].geoms]
                areas = np.array([p.area for p in vp[i].geoms])
                vp[i] = allparts[np.argmax(areas)]
        
            xy = vp[i].exterior.xy
            verts.append(np.array(xy).T)
        
        self.Geoms['poly'] = verts
        
        # Line Geometry
        # Next section deals with edge line between any two voroni polygons. 
        # Overall it relies on vor.ridge_* attributes, but need to deal 
        # with bounding box and with lines that goes to infinity
        # 
        # This geom maps to edges so things are stored using dict with (v1,v2) tuples as keys
        # the tuples are always sorted from low o high as a convension. 
        mx = np.max(np.array(self.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        mn = np.min(np.array(self.Geoms['BoundingBox'].exterior.xy).T,axis=0)

        mins = np.tile(mn, (vor.vertices.shape[0], 1))
        bounded_vertices = np.max((vor.vertices, mins), axis=0)
        maxs = np.tile(mx, (vor.vertices.shape[0], 1))
        bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

        segs = list()
        center = XY.mean(axis=0)
        for i in range(len(vor.ridge_vertices)):
            pointidx = vor.ridge_points[i,:]
            simplex = np.asarray(vor.ridge_vertices[i])
            if np.all(simplex >= 0):
                line=[(bounded_vertices[simplex[0], 0], bounded_vertices[simplex[0], 1]),
                      (bounded_vertices[simplex[1], 0], bounded_vertices[simplex[1], 1])]
            else:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = XY[pointidx[1]] - XY[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = XY[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                line=[(vor.vertices[i,:]),(far_point)]
    
            LS = LineString(line)
            LS = LS.intersection(self.Geoms['BoundingBox'])
            if isinstance(LS,MultiLineString):
                allparts = list(LS.intersection(self.Geoms['BoundingBox']).geoms)
                lengths = [l.length for l in allparts]
                LS = allparts[np.argmax(lengths)]
                
            xy = np.asarray(LS.xy)
            line=xy.T
            segs.append(line)
        
        # make sure the keys for edges are always sorted, a convenstion that will make life easier. 
        ridge_points_sorted = np.sort(vor.ridge_points,axis=1)
        keys=list()
        for i in range(ridge_points_sorted.shape[0]):
            keys.append(tuple(ridge_points_sorted[i,:]))

        self.Geoms['line'] = dict(zip(keys, segs))
        
        # Geom Points are easy, just use XY, the order is correct :)
        self.Geoms['point'] = XY
            
    def add_view(self,view):
        # add the view to the view dict
        self.Views[view.name]=view
        
    def plot_cond_entropy_at_different_resolutions(self,lvl,pth = None):
        
        EntropyCalcsL1 = self.Layers[lvl].calc_entropy_at_different_Leiden_resolutions()
        fig= plt.figure(figsize = (8,8))
        ax1 = plt.gca()
        yopt = self.cond_entropy[1]
        xopt = self.Ntypes[1]
        ax1.plot(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
        ylm = ax1.get_ylim()
        ax1.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
        ax1.set_xlabel('# of types',fontsize=18)
        ax1.set_ylabel('H (Map | Type)',fontsize=18)
        fig = plt.gcf()
        left, bottom, width, height = [0.6, 0.55, 0.25, 0.25]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.semilogx(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
        ylm = ax2.get_ylim()
        ax2.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)

        if pth is not None: 
            lvl=lvl+1
            fig.savefig(pth + "CondEntL" + lvl +".png")
        
        return(fig)
        

class TissueGraph:
    """
    TissueGraph: the core class used to analyze tissues using Graph representation is TissueGraph 

    Tissue graphs can be created in three different ways; 
        1) using XY possition of centroids (cell layer)
        2) using local spatial coherence and watershed (microenv)
        3) contracting existing graph based on taxonomy and maximial conditional entropy (zones and regions)
        
    key attributes: 
        * feature_mat - feature matrix for each vertex (shape[0] == N, i.e. size of SG)
        * feature_type_mat - feature matrix for each type (for PNMF, simple average, for neighberhoods - counts per type ignoring)
        * SG - iGraph representation of the spatial graph (i.e. cells, zones, microenv, regions)
        * FG - iGraph representation of feature graph. 
               For primary layers (cells, microenv) the vertices are the same as the spatial graph. 
               for secondary layers (zones, regions) the vertices are pure "types" and the relationship between them, 
               FG in secondary layers is just contraction of the type layer done by Leiden 
        
    main methods:
        * build_spatial_graph - construct graph based on Delauny neighbors
        * contract_graph - find zones/region, i.e. spatially continous areas in the graph the same (cell/microenvironment) type
        * cond_entopy - calculates the conditional entropy of a graph given types (or uses defaults existing types)
        * watershed - devide into regions based on watershed
        * calc_graph_env_coherence - estimate spatial local coherence (distance metric defined by Taxonomy object) 

    Many other accessory functions that are used to set/get TissueGraph information.

    """
    def __init__(self):
        
        self.layer_type = None # label layers by their type
        self.Type = None # Type vector
        
        self.feature_mat = None # feature - one per vertex in SG
        
        # only avaliable for even layers (i.e. iso-zones & regions)
        self.feature_type_mat = None # features - one per type (i.e. vertex in FG)
        self.Dtype = None # pairwise distace vector storing type similarities (used for Viz)
        self.Type2 = None # classification of the unique types into coarse level types (used for Viz)
            
        self.SG = None # spatial graph
        self.FG = None # Feature graph (that was used to create types)

        
        self.UpstreamMap = None
        # self.TX = Taxonomy()

        return None
        
    @property
    def N(self):
        """Size of the tissue graph
            internally stored as igraph size
        """
        if self.SG is not None:
            return len(self.SG.vs)
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with build_spatial_graph')
    
    @property
    def node_size(self):
        if self.SG is not None:
            return np.asarray(self.SG.vs['Size'])
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with build_spatial_graph or contract_graph methods')
    
    @node_size.setter
    def node_size(self,Nsz):
        self.SG.vs["Size"]=Nsz
    
    @property
    def XY(self):
        """
            XY : dependent property - will query info from Graph and return
        """
        XY=np.zeros((self.N,2))
        XY[:,0]=self.SG.vs["X"]
        XY[:,1]=self.SG.vs["Y"]
        return(XY)
        
    @property    
    def X(self):
        """
            X : dependent property - will query info from Graph and return
        """
        return(self.SG.vs["X"])
        
    @property
    def Y(self):
        """Y : dependent property - will query info from Graph and return
        """
        return(self.SG.vs["Y"])
    
    @property    
    def Ntypes(self): 
        """ 
            Ntypes: returns number of unique types in the graph
        """ 
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count how many")
        return(len(np.unique(self.Type)))
    
    def build_spatial_graph(self,XY):
        """
        build_spatial_graph will create an igrah using Delaunay triangulation

        Params: 
            XY - centroid regions to build a graph around

        Out: 
            G - an igraph 

        """
        # validate input
        if not isinstance(XY, np.ndarray):
            raise ValueError('XY must be a numpy array')
        
        if not XY.shape[1]==2:
            raise ValueError('XY must have only two columns')
            
        # start with triangulation
        dd=Delaunay(XY)

        # create Graph from edge list
        EL = np.zeros((dd.simplices.shape[0]*3,2))
        for i in range(dd.simplices.shape[0]): 
            EL[i*3,:]=[dd.simplices[i,0],dd.simplices[i,1]]
            EL[i*3+1,:]=[dd.simplices[i,0],dd.simplices[i,2]]
            EL[i*3+2,:]=[dd.simplices[i,1],dd.simplices[i,2]]

        self.SG = Graph(n=XY.shape[0],edges=EL,directed=False).simplify()
        self.SG.vs["X"]=XY[:,0]
        self.SG.vs["Y"]=XY[:,1]
        self.SG.vs["Size"]=np.ones(len(XY[:,1]))
        
        # set up names
        self.SG.vs["name"]=list(range(self.N))
        return(self)

    def calc_type2(self,n_cls = 5):
        it_cnt = 0
        nt2=0
        res = 0.01
        while nt2<n_cls and it_cnt < 1000: 
            res = res+0.1
            T2 = np.array(self.FG.community_leiden(objective_function='modularity',resolution_parameter = res).membership).astype(np.int64)
            nt2 = len(np.unique(T2))
            it_cnt+=1
                
        if it_cnt>=1000:
            raise RuntimeError('Infinite loop adjusting Leiden resolution')
        
        self.Type2 = T2
        
    
    def contract_graph(self,TypeVec = None):
        """contract_graph : reduce graph size by merging neighbors of same type. 
            Given a vector of types, will contract the graph to merge vertices that are 
            both next to each other and of the same type. 
        
        Input: TypeVec - a vector of Types for each node. 
               If TypeVec is not provided will attempty to use the Type property of the graph itself. 
        
        Output: a new TissueGraph after vertices merging. 
        """

        # Figure out which type to use
        if TypeVec is None: 
            TypeVec = self.Type
        
        # get edge list - work with names and not indexes in case things shift around (they shouldn't),     
        EL = np.asarray(self.SG.get_edgelist()).astype("int")
        nm=self.SG.vs["name"]
        EL[:,0] = np.take(nm,EL[:,0])
        EL[:,1] = np.take(nm,EL[:,1])
        
        # only keep edges where neighbors are of same types
        EL = EL[np.take(TypeVec,EL[:,0]) == np.take(TypeVec,EL[:,1]),:]
        
        # remake a graph with potentially many components
        IsoZonesGraph = Graph(n=self.N, edges=EL)
        IsoZonesGraph = IsoZonesGraph.as_undirected().simplify()
        IsoZonesGraph.vs["name"] = IsoZonesGraph.vs.indices

        # because we used both type and proximity, the original graph (based only on proximity)
        # that was a single component graph will be broken down to multiple components 
        # finding clusters for each component. 
        cmp = IsoZonesGraph.components()
        
        IxMapping = np.asarray(cmp.membership)
        
        ZoneName, ZoneSingleIx = np.unique(IxMapping, return_index=True)
        
        # zone size sums the current graph zone size per each aggregate (i.e. zone or microenv)
        df = pd.DataFrame(data = self.node_size)
        df['type'] = IxMapping
        ZoneSize = df.groupby(['type']).sum()
        ZoneSize = np.array(ZoneSize).flatten()
         
        # create a new Tissue graph by copying existing one, contracting, and updating XY
        ZoneGraph = TissueGraph()
        ZoneGraph.SG = self.SG.copy()
        
        comb = {"X" : "mean",
               "Y" : "mean",
               "Type" : "ignore",
               "name" : "ignore"}
        
        ZoneGraph.SG.contract_vertices(IxMapping,combine_attrs=comb)
        ZoneGraph.SG.simplify()
        ZoneGraph.SG.vs["Size"] = ZoneSize
        ZoneGraph.SG.vs["name"] = ZoneName
        ZoneGraph.Type = TypeVec[ZoneSingleIx]
        ZoneGraph.UpstreamMap = IxMapping
        
        return(ZoneGraph)
                             

                             
    def type_freq(self): 
        """
            type_freq: return the catogorical probability for each type
        """
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count frequencies")
        unqTypes = np.unique(self.Type)
        Ptypes = count_values(self.Type,unqTypes,self.node_size)
        
        return Ptypes,unqTypes
    
    
    def cond_entropy(self):
        """
        cond_entropy: calculate conditional entropy of the tissue graph
                     cond entropy is the difference between graph entropy based on pagerank and type entropy
        """
        Pzones = self.node_size
        Pzones = Pzones/np.sum(Pzones)
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        # validate that type exists
        if self.Type is None: 
            raise ValueError("Can't calculate cond-entropy without Types, please check")
            
        Ptypes = self.type_freq()[0] 
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        cond_entropy = Entropy_Zone-Entropy_Types
        return(cond_entropy)
    
    def extract_environments(self,ordr = None,typevec = None):
        """
            returns the categorical distribution of neighbors. Depending on input there could be two uses, 
            if ordr is not None :  returns local neighberhood (distance order on the graph) for all vertices. 
            if typevec is not None : returns local env based on typevec, will return one env for each unique type in typevec
        """
        unqlbl = np.unique(self.Type)
        
        # arrange the indexes for the environments. 
        # if we use ordr this is neighborhood defined by iGraph
        # if we provide types, than indexes of each type. 
        if ordr is not None and typevec is None:
            ind = self.SG.neighborhood(order = ordr)
        elif typevec is not None and ordr is None:
            ind = list()
            for i in range(len(np.unique(typevec))):
                ind.append(np.where(typevec==i))
        else: 
            raise ValueError('either order or typevec must be provided, not both (or none)')
        
        unqlbl = np.unique(self.Type)
        Env = np.zeros((len(ind),len(unqlbl)))
        ndsz = self.node_size.copy().astype(np.int64)
        AllTypes = [self.Type[ix] for ix in ind]
        AllSizes = [ndsz[ix] for ix in ind]
        for i in range(Env.shape[0]):
            Env[i,:]=np.bincount(AllTypes[i],weights = AllSizes[i],minlength=len(unqlbl))
        
        # should be the same as above, but much slower... keeping it here for now till more testing is done. 
        # for i in range(len(ind)):
        #     Env[i,:]=count_values(self.Type[ind[i]],unqlbl,ndsz[ind[i]],norm_to_one = False)
            
        return(Env)
    
    def graph_local_avg(self,VecToSmooth,ordr = 3,kernel = np.array([0.4,0.3,0.2,0.1])):
        Smoothed = np.zeros((len(VecToSmooth),ordr+1))
        Smoothed[:,0] = VecToSmooth
        for i in range(ordr):
            ind = self.SG.neighborhood(order = i+1,mindist=i)
            for j in range(len(ind)):
                ix = np.array(ind[j],dtype=np.int64)
                Smoothed[j,i+1] = np.nanmean(VecToSmooth[ix])

        kernel = kernel[None,:]
        kernel = np.repeat(kernel,len(VecToSmooth),axis=0)
        ix_nan = np.isnan(Smoothed)
        kernel[ix_nan]=0
        sum_of_rows = kernel.sum(axis=1)
        kernel = kernel / sum_of_rows[:, None]
        Smoothed[ix_nan] = 0
        Smoothed = np.multiply(Smoothed,kernel)
        Smoothed = Smoothed.sum(axis=1)
        return(Smoothed)

    
    def watershed(self,CoherenceScores):
        
        # create two directed graphs based on Zone graph
        # first is directed version of the zone graph with edge weights
        # based on local gradient of coherence score
        # second is a copy of the first but keeping only increasing edges
        min_edge_weight = 0
        CellCoherence = -np.log10(CoherenceScores)
        EL = np.asarray(self.SG.get_edgelist()).astype("int")
        G = Graph(n=self.N,edges = EL,directed = False)
        G = G.as_directed()
        EL = np.asarray(G.get_edgelist()).astype("int")
        grad = CellCoherence[EL[:,0]]-CellCoherence[EL[:,1]]
        ix = np.flatnonzero(grad<min_edge_weight)
        edge_tuples = list(map(tuple, EL[ix,:]))
        G2 = G.copy()
        G2.delete_edges(edge_tuples) 
        G2.es['weights']=grad[grad>min_edge_weight]-min_edge_weight
        outdeg = np.array(G2.outdegree(),dtype='int')
        peaks = np.flatnonzero(outdeg==0)
        
        # find peaks and remove small regions with only one cell
        Init = -np.ones(self.N)
        Fixed = np.zeros(self.N)
        Fixed[peaks]=1
        Fixed = Fixed.astype('bool').flatten()
        Init[peaks]=np.arange(len(peaks))
        Init[~Fixed]=-1
        grad = grad-min(grad)
        HoodId = G.community_label_propagation(weights=grad, initial=Init, fixed=Fixed).membership
        HoodId = np.array(HoodId,dtype='int')
        unq,cnt = np.unique(HoodId,return_counts = True)
        peaks = peaks[cnt>1]

        Adj = self.SG.get_adjacency_sparse()
        Dij_min, predecessors,ClosestPeak = dijkstra(Adj, directed=True, 
                                                          indices=peaks, 
                                                          return_predecessors=True, 
                                                          unweighted=False, 
                                                          limit=np.inf, 
                                                          min_only=True)
        # Dij_all = G.shortest_paths(target=peaks,weights = grad)
        # ClosestPeak = np.argmin(Dij_all,axis=1)
        # Dij_min = np.min(Dij_all,axis=1)
        
        # renumber all closest peak continously
        u,ix_rev = np.unique(ClosestPeak, return_inverse=True)
        ClosestPeak=u[ix_rev]
        
        # relabel HoodId in case we have a heterozone that was split along the way
        # by contracting and expanding where each contracted zone gets a unique ID we
        # basically garantuee that Ntypes = N (for now...)
        CG = self.contract_graph(TypeVec = ClosestPeak)
        Id = np.arange(CG.N)
        ClosestPeak = Id[CG.UpstreamMap]
        
        return (ClosestPeak,Dij_min)
        
#         # find fixed points, i.e. local minima (lower scores are higher 
#         hood = self.SG.neighborhood(order=hood_size,mindist=1)
#         Init = -np.ones(len(hood))
#         Fixed = np.zeros(len(hood))

#         for i in range(len(hood)):
#             if np.all(NodeWeight[i] <= NodeWeight[hood[i]]):
#                 Init[i] = i
#                 Fixed[i] = 1

#         Fixed = Fixed.astype('bool').flatten()
#         u,ix_rev = np.unique(Init, return_inverse=True)
#         Init=u[ix_rev]
#         Init[~Fixed]=-1
        
#         if only_find_peaks: 
#             return Init
#         else:
#             # label_prop is stochastic and does not guarantee continuous communities. 
#             # so after label props break any split community into parts. Using similar logic as in contract_graph() method
#             HoodId = self.SG.community_label_propagation(weights=None, initial=Init, fixed=Fixed).membership
#             EL = np.asarray(self.SG.get_edgelist()).astype("int")
#             EL = EL[np.take(HoodId,EL[:,0]) == np.take(HoodId,EL[:,1]),:]
#             hoodG = Graph(n=self.N, edges=EL)
#             hoodG = hoodG.as_undirected().simplify()
#             cmp = hoodG.components()
#             HoodId = np.array(cmp.membership)

#             return HoodId
                           
#     def calc_graph_env_coherence(self,Env,f_dist,*args):
#         # find for each edges the two environments that it connects
#         EL = np.asarray(self.SG.get_edgelist()).astype("int")
        
#         Env1 = Env[EL[:,0],:]
#         Env2 = Env[EL[:,1],:]
        
#         EdgeWeight = f_dist(Env1,Env2,*args)
        
#         # create node weights: 
#         df = pd.DataFrame(np.hstack((EdgeWeight,EdgeWeight)))
#         df['type']=np.hstack((EL[:,0],EL[:,1]))
#         avg = df.groupby(['type']).mean()
#         NodeWeight = np.array(avg).flatten()
        
#         return (EdgeWeight,NodeWeight)

    def calc_graph_env_coherence(self,Env,f_dist,*args):
    
        ind = np.array(self.SG.neighborhood(order = 4),dtype='object')

        # find for each edges the two environments that it connects
        EL = np.asarray(self.SG.get_edgelist()).astype("int")

        EnvIx1 = ind[EL[:,0]]
        EnvIx2 = ind[EL[:,1]]

        for i in range(EnvIx1.shape[0]):
            shared = np.intersect1d(EnvIx1[i],EnvIx2[i])
            EnvIx1[i] = np.setdiff1d(EnvIx1[i],shared)
            EnvIx2[i] = np.setdiff1d(EnvIx2[i],shared)

        unqlbl = np.unique(self.Type)
        Env1 = np.zeros((EnvIx1.shape[0],len(unqlbl)))
        Env2 = np.zeros((EnvIx2.shape[0],len(unqlbl)))
        ndsz = self.node_size.copy().astype(np.int64)
        AllTypes1 = [self.Type[ix] for ix in EnvIx1]
        AllSizes1 = [ndsz[ix] for ix in EnvIx1]
        AllTypes2 = [self.Type[ix] for ix in EnvIx2]
        AllSizes2 = [ndsz[ix] for ix in EnvIx2]
        for i in range(Env1.shape[0]):
            Env1[i,:]=np.bincount(AllTypes1[i],weights = AllSizes1[i],minlength=len(unqlbl))
            Env2[i,:]=np.bincount(AllTypes2[i],weights = AllSizes2[i],minlength=len(unqlbl))

        EdgeWeight = f_dist(Env1,Env2,*args)

        # create node weights: 
        df = pd.DataFrame(np.hstack((EdgeWeight,EdgeWeight)))
        df['type']=np.hstack((EL[:,0],EL[:,1]))
        avg = df.groupby(['type']).mean()
        NodeWeight = np.array(avg).flatten()

        return (EdgeWeight,NodeWeight)
    
       
    def calc_entropy_at_different_Leiden_resolutions(self): 
        Rvec = np.logspace(-1,2.5,100)
        Ent = np.zeros(Rvec.shape)
        Ntypes = np.zeros(Rvec.shape)
        for i in range(len(Rvec)):
            TypeVec = self.FG.community_leiden(resolution_parameter=Rvec[i],objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Ent[i] = self.contract_graph(TypeVec).cond_entropy()
            Ntypes[i] = len(np.unique(TypeVec))
            
        df = pd.DataFrame(data = {'Entropy' : Ent, 'Ntypes' : Ntypes, 'Resolution' : Rvec})     
        return df
    
    def multilayer_Leiden_with_cond_entropy(self): 
        """
            Find optimial clusters by peforming clustering on two-layer graph. 
            
            Input
            -----
            TG : A TissueGraph that has matching SpatialGraph (SG) and FeatureGraph (SG)
            optimization is done on resolution parameter  
            
        """

        def ObjFunLeidenRes(res):
            """
            Basic optimization routine for Leiden resolution parameter. 
            Implemented using igraph leiden community detection
            """
            TypeVec = self.FG.community_leiden(resolution_parameter=res,objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Entropy = self.contract_graph(TypeVec).cond_entropy()
            return(-Entropy)

        print(f"Calling initial optimization")
        sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
                                               method='bounded',
                                               options={'xatol': 1e-2, 'disp': 3})
        initRes = sol['x']
        TypeVec = self.FG.community_leiden(resolution_parameter=initRes,objective_function='modularity').membership
        TypeVec = np.array(TypeVec).astype(np.int64)
        print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {-sol['fun']} number of evals: {sol['nfev']}")
        return TypeVec

