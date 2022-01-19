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

from scipy.spatial.distance import jensenshannon, pdist, squareform

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
        cmb = np.array(list(itertools.combinations(np.arange(M1.shape[0]), r=2)))
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
    
    # update number of neighbors (x accuracy) to increase accuracy
    # at the same time checks if we have enough rows 
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
            

        return None
    
    def save(self,fullfilename):
        """ pickle and save
        """
        pickle.dump(self,open(fullfilename,'wb'),recurse=True)
        
    def create_cell_and_zone_layers(self,XY,PNMF): 
        """
        Creating cell and zone layers. 
        Cell layer is unique as it's the only one where spatial information is directly used with Voronoi
        Zone layer is calculated based on optimization of cond entropy betweens zones and types. 
        """
        # creating first layer - cell tissue graph
        TG = TissueGraph()
        TG.layer_type = "cells"
        TG.build_spatial_graph(XY)
        TG.FG = buildgraph(PNMF,metric = 'cosine')
        
        # cluster cell types optimally
        celltypes = TG.multilayer_Leiden_with_cond_entropy()
        
        # add types and key data
        TG.Type = celltypes
        TG.feature_mat = PNMF
        
        # add layer
        self.Layers.append(TG)
        
        # contract and create the Zone graph
        NewLayer = self.Layers[-1].contract_graph(celltypes)
        self.Layers.append(NewLayer)
        
        # add feature graph - for zones, this graph will have one node per type. 
        self.Layers[-1].FG = self.Layers[-2].FG.copy()
        self.Layers[-1].FG.contract_vertices(celltypes)
        self.Layers[-1].FG.simplify()
        self.Layers[-1].layer_type = 'isozones'
        
        df = pd.DataFrame(data = PNMF)
        df['type']=self.Layers[-1].UpstreamMap
        self.Layers[-1].feature_mat = np.array(df.groupby(['type']).mean())
        df = pd.DataFrame(data = PNMF)
        df['type']=celltypes
        self.Layers[-1].feature_type_mat = np.array(df.groupby(['type']).mean())
        
        return None
    
    def create_communities_and_region_layers(self,min_neigh_size = 10,ordr = 3,hood_size = 1,metric_kwds = {}):
        """
        Creating community and region layers. 
        Community layer is created based on watershed using local env coherence. 
        Region layer is calculated based on optimization of cond entropy betweens community and regions. 
        """
        # find existing environment 
        Env = self.Layers[-1].extract_environments(ordr=ordr)
        
        # perform watershed with seeds as "peaks" in NodeWeight graph
        HoodId = self.Layers[-1].watershed(Env,hood_size = hood_size)
                
        # add Community layers: 
        NewLayer = self.Layers[-1].contract_graph(HoodId)
        self.Layers.append(NewLayer)
        self.Layers[-1].layer_type = 'neighberhoods'
                
        # calculate true environments vector for all communities: 
        WatershedEnvs = self.Layers[-2].extract_environments(typevec = HoodId)
        
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
        
        self.fill_holes(3,min_neigh_size)
        # self.add_contracted_layer_using_type(WatershedEnvs,commtypes)
        
        # add feature graph - for regions this graph will have one node per type. 
        self.Layers[-1].FG = self.Layers[-2].FG.copy()
        self.Layers[-1].FG.contract_vertices(commtypes)
        self.Layers[-1].FG.simplify()
        self.Layers[-1].layer_type = 'regions'
        
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

        
    def fill_holes(self,lvl_to_fill,min_node_size):

        # get types (with holes)
        region_types = self.Layers[lvl_to_fill].Type.copy()

        # mark all verticies with small node size
        region_types[self.Layers[3].node_size < min_node_size]=-1
        fixed = region_types>-1

        # renumber region types in case removing some made numbering discontinuous
        unq,ix = np.unique(region_types[fixed],return_inverse=True)
        cont_region_types = region_types.copy()
        cont_region_types[fixed] = ix

        # map to level-0 to do the label propagation at the cell level: 
        fixed_celllvl = self.map_to_cell_level(3,fixed)
        contregion_celllvl = self.map_to_cell_level(3,cont_region_types)

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
        keys = list(vor.ridge_dict.keys())
        for i in range(len(keys)):
            if keys[i][0]<=keys[i][1]:
                keys[i]=(keys[i][1],keys[i][0])

        self.Geoms['line'] = dict(zip(keys, segs))
        
        # Geom Points are easy, just use XY, the order is correct :)
        self.Geoms['point'] = XY
            
    def add_view(self,view):
        # add the view to the view dict
        self.Views[view.name]=view
    

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
        
        self.layer_type = None
        self.Type = None # Type vector
        self.nbrs = None
        self.feature_mat = None # feature - one per vertex in SG
        self.feature_type_mat = None # features - one per type (i.e. vertex in FG)
                                     # only avaliable for even layers (i.e. iso-zones & regions)
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
        for i in range(len(ind)):
            Env[i,:]=count_values(self.Type[ind[i]],unqlbl,self.node_size[ind[i]],norm_to_one = False)
            
        return(Env)
    
    def smooth(self,VecToSmooth,ordr):
        ind = self.SG.neighborhood(order = ordr)
        Smoothed = np.zeros(VecToSmooth.shape)
        for i in range(len(ind)):
            ix = np.array(ind[i],dtype=np.int64)
            Smoothed[i] = np.mean(VecToSmooth[ix])
            
        return(Smoothed)
    
    def watershed(self,Env,hood_size = 1, smooth_ordr = 0, only_find_peaks = False):
        
        # get edge and node spatial coherence scores
        # Using JSD
        (EdgeWeight,NodeWeight) = self.calc_graph_env_coherence(Env,dist_jsd)
        
        if smooth_ordr>0: 
            NodeWeight = self.smooth(NodeWeight,smooth_ordr)
        
        # find fixed points, i.e. local minima (lower scores are higher 
        hood = self.SG.neighborhood(order=hood_size,mindist=1)
        Init = -np.ones(len(hood))
        Fixed = np.zeros(len(hood))

        for i in range(len(hood)):
            if np.all(NodeWeight[i] <= NodeWeight[hood[i]]):
                Init[i] = i
                Fixed[i] = 1

        Fixed = Fixed.astype('bool').flatten()
        u,ix_rev = np.unique(Init, return_inverse=True)
        Init=u[ix_rev]
        Init[~Fixed]=-1
        
        if only_find_peaks: 
            return Init
        else:
            # label_prop is stochastic and does not guarantee continuous communities. 
            # so after label props break any split community into parts. Using similar logic as in contract_graph() method
            HoodId = self.SG.community_label_propagation(weights=None, initial=Init, fixed=Fixed).membership
            EL = np.asarray(self.SG.get_edgelist()).astype("int")
            EL = EL[np.take(HoodId,EL[:,0]) == np.take(HoodId,EL[:,1]),:]
            hoodG = Graph(n=self.N, edges=EL)
            hoodG = hoodG.as_undirected().simplify()
            cmp = hoodG.components()
            HoodId = np.array(cmp.membership)

            return HoodId
                           
    def calc_graph_env_coherence(self,Env,f_dist,*args):
        # find for each edges the two environments that it connects
        EL = np.asarray(self.SG.get_edgelist()).astype("int")
        
        Env1 = Env[EL[:,0],:]
        Env2 = Env[EL[:,1],:]
        
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

