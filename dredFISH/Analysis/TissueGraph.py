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
from scipy.stats import entropy, mode
from scipy.stats.contingency import crosstab
from scipy.special import rel_entr
from scipy.interpolate import interp2d, interp1d


import sklearn
import sklearn.decomposition
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score 

from numpy.random import default_rng
rng = default_rng()

from sklearn.preprocessing import normalize
from MERFISH_Objects.FISHData import *

import functools
import ipyparallel as ipp

from collections import Counter

import numpy as np
import numpy
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

def lda_fit(n_topics,Env = None):
    lda = sklearn.decomposition.LatentDirichletAllocation(n_components=n_topics)
    B = lda.fit_transform(Env)
    return B

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
        self.layers_graph = list() # a list of tuples that keep track of the relationship between different layers 
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
        self.dview=None
        pickle.dump(self,open(fullfilename,'wb'),recurse=True)

    def load_and_normalize_data(self,base_path,dataset):

        fishdata = FISHData(os.path.join(base_path,'fishdata'))
        data = fishdata.load_data('h5ad',dataset=dataset)
        data.obs_names_make_unique()

        data.X = data.layers['total_vectors']
        data = data[np.isnan(data.X.max(1))==False]

        data.X = data.X/data.obs['total_signal'][:,None]
        data.X = data.X - np.array([np.percentile(data.X[:,i],25) for i in range(data.X.shape[1])])
        data.X = data.X / np.array([np.percentile(data.X[:,i],75) for i in range(data.X.shape[1])])
        data.X = normalize(data.X)

        XY = np.asarray([data.obs['stage_y'], data.obs['stage_x']])
        XY = np.transpose(XY)
        return (XY,data.X)

        
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
        
        self.layers_graph.append((0,1))
        
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
    
    def find_topics(self,ordr=4,max_num_of_topics = 120,use_parallel = True):
        # get local environments per cell and calculate local environment cell type frequencies 
        Env = self.Layers[0].extract_environments(ordr=ordr)
        row_sums = Env.sum(axis=1)
        row_sums = row_sums[:,None]
        Env = Env/row_sums
        Ntopics = np.arange(2,max_num_of_topics)  
        
        if use_parallel: 
            # start parallel engine
            rc = ipp.Client()
            self.dview = rc[:]
            self.dview.push({'Env': Env})
            with self.dview.sync_imports():
                import sklearn.decomposition
                
            g = functools.partial(lda_fit, Env=Env)
            result = self.dview.map_sync(g,Ntopics)
        else: 
            result = list()
            for i in range(len(Ntopics)):
                lda = LatentDirichletAllocation(n_components=Ntopics[i])
                B = lda.fit_transform(Env)
                result.append(B)
                
        IDs = np.zeros((self.N[0],len(result)))
        for i in range(len(result)):
            IDs[:,i] = np.argmax(result[i],axis=1)
        ID_entropy=np.zeros(IDs.shape[1])
        Type_entropy = np.zeros(IDs.shape[1])
        for i in range(IDs.shape[1]):
            _,cnt = np.unique(IDs[:,i],return_counts=True)
            cnt=cnt/cnt.sum()
            Type_entropy[i] = entropy(cnt,base=2) 
            
        topics = IDs[:,np.argmax(Type_entropy)]
        unq,ix = np.unique(topics,return_inverse=True)
        id = np.arange(len(unq))
        topics = id[ix]
        return topics

    
    def create_region_layer(self,topics,min_neigh_size = 10):
        """
        Function gets vector of topicid (one per cell) and uses it 
        to add regions to the TMG
        """
    
        # create region layers through graph contraction
        NeihLayers = self.Layers[0].contract_graph(topics)
        self.Layers.append(NeihLayers)
        current_layer_id = len(self.Layers)-1
        self.layers_graph.append((0,current_layer_id))
        
        # replace type of heterozones that are very small with local region type (labelprop) 
        # self.fill_holes(current_layer_id,min_neigh_size)

        # create the feature_mat and feature_type_mat 
        regions_mapped_to_cells = self.map_to_cell_level(current_layer_id,self.Layers[current_layer_id].Type)
        feature_type_mat = self.Layers[0].extract_environments(typevec = regions_mapped_to_cells)
        self.Layers[current_layer_id].feature_type_mat = feature_type_mat
        
        row_sums = feature_type_mat.sum(axis=1)
        row_sums=row_sums[:,None]
        feature_type_mat=feature_type_mat/row_sums
        
        D = dist_emd(feature_type_mat,None,self.Layers[1].Dtype)
        self.Layers[current_layer_id].Dtype = squareform(D)
        self.Layers[current_layer_id].FG = buildgraph(self.Layers[current_layer_id].Dtype,metric = 'precomputed',n_neighbors = 5)
        
        self.Layers[current_layer_id].calc_type2()
        self.Layers[current_layer_id].layer_type = 'region'

        
    def fill_holes(self,lvl_to_fill,min_node_size):
        update_feature_mat_flag=False
        if self.Layers[lvl_to_fill].feature_mat is not None: 
            feature_mat = self.Layers[lvl_to_fill].feature_mat.copy()
            update_feature_mat_flag = True

        # get types (with holes)
        region_types = self.Layers[lvl_to_fill].Type.copy()

        # mark all verticies with small node size
        region_types[self.Layers[lvl_to_fill].node_size < min_node_size]=-1
        fixed = region_types > -1

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
        upstream_layer_ix = self.find_upstream_layer(lvl_to_fill)
        NewLayer = self.Layers[lvl_to_fill-1].contract_graph(new_type)
        self.Layers[lvl_to_fill] = NewLayer
        if update_feature_mat_flag:
            self.Layers[lvl_to_fill].feature_mat = feature_mat[fixed,:]

    def find_upstream_layer(self,layer_id):
        upstream_layer_id=0
        return upstream_layer_id
    
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
            
        # if needed (i.e. not already at cell level) expand indexing backwards in layers following layers_graph
        ix=self.Layers[lvl].UpstreamMap
        while lvl>0:
            lvl = self.find_upstream_layer(lvl)
            ix=ix[self.Layers[lvl].UpstreamMap]
        
        VecToMap = VecToMap[ix].flatten()
        
        if return_ix: 
            return (VecToMap,ix)
        else: 
            return VecToMap
    
    def find_regions_edge_level(self):
        """
        determine whch cell edges (layer 0) are also edges between regions (layer 2) 
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
        region_id = self.map_to_cell_level(2,np.arange(self.N[2]))
        np_edge_list = np_edge_list[region_id[np_edge_list[:,0]] != region_id[np_edge_list[:,1]],:]
        region_edge_list = [(np_edge_list[i,0],np_edge_list[i,1]) for i in range(np_edge_list.shape[0])]            
        return region_edge_list
    
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
        
        self.SG = None # spatial graph
        self.feature_mat = None # feature - one per vertex in SG
        
        # only avaliable for even layers (i.e. iso-zones & regions)
        self.feature_type_mat = None # features - one per type (i.e. vertex in FG)
        self.Dtype = None # pairwise distace vector storing type similarities (used for Viz)
        self.Type2 = None # classification of the unique types into coarse level types (used for Viz)
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
        Env = np.zeros((len(ind),len(unqlbl)),dtype=np.int64)
        ndsz = self.node_size.copy().astype(np.int64)
        int_types = self.Type.astype(np.int64)
        AllTypes = [int_types[ix] for ix in ind]
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

    
    def watershed(self,CellCoherence):
        is_peak = np.zeros(CellCoherence.shape).astype('bool')
        ind = self.SG.neighborhood(order = 1,mindist=1)
        for i in range(len(ind)):
            is_peak[i] = np.all(CellCoherence[i]>CellCoherence[ind[i]])
        peaks = np.flatnonzero(is_peak)  

        Adj = self.SG.get_adjacency_sparse()
        Dij_min, predecessors,ClosestPeak = dijkstra(Adj, directed=False, 
                                                          indices=peaks, 
                                                          return_predecessors=True, 
                                                          unweighted=False, 
                                                          limit=np.inf, 
                                                          min_only=True)
        
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
        

    def calc_entropy_at_different_Leiden_resolutions(self,Rvec = np.logspace(-1,2.5,100)): 
        Ent = np.zeros(Rvec.shape)
        Ntypes = np.zeros(Rvec.shape)
        for i in range(len(Rvec)):
            TypeVec = self.FG.community_leiden(resolution_parameter=Rvec[i],objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Ent[i] = self.contract_graph(TypeVec).cond_entropy()
            Ntypes[i] = len(np.unique(TypeVec))
            
        self.cond_entropy_df = pd.DataFrame(data = {'Entropy' : Ent, 'Ntypes' : Ntypes, 'Resolution' : Rvec})     
    
    def multilayer_Leiden_with_cond_entropy(self,opt_params = {'iters' : 10, 'n_consensus' : 50}): 
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
            EntropyVec = np.zeros(opt_params['iters'])
            for i in range(opt_params['iters']):
                TypeVec = self.FG.community_leiden(resolution_parameter=res,objective_function='modularity').membership
                TypeVec = np.array(TypeVec).astype(np.int64)
                EntropyVec[i] = self.contract_graph(TypeVec).cond_entropy()
            Entropy = EntropyVec.mean()
            return(-Entropy)

        print(f"Calling initial optimization")
        sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
                                               method='bounded',
                                               options={'xatol': 1e-2, 'disp': 3})
        initRes = sol['x']
        
        # consensus clustering
        TypeVec = np.zeros((self.N,opt_params['n_consensus']))
        for i in range(opt_params['n_consensus']):
            tv = self.FG.community_leiden(resolution_parameter=initRes,objective_function='modularity').membership
            TypeVec[:,i] = np.array(tv).astype(np.int64)
            
        if opt_params['n_consensus']>1:
            cmb = np.array(list(itertools.combinations(np.arange(opt_params['n_consensus']), r=2)))
            rand_scr = np.zeros(cmb.shape[0])
            for i in range(cmb.shape[0]):
                rand_scr[i] = adjusted_rand_score(TypeVec[:,cmb[i,0]],TypeVec[:,cmb[i,1]])
            rand_scr = squareform(rand_scr)
            total_rand_scr = rand_scr.sum(axis=0)
            TypeVec = TypeVec[:,np.argmax(total_rand_scr)]
                                                  

        print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {-sol['fun']} number of evals: {sol['nfev']}")
        return TypeVec
    
    def gradient_magnitude(self,V):
        EL = self.SG.get_edgelist()
        EL = np.array(EL,dtype='int')
        XY = self.XY
        dV = V[EL[:,1]]-V[EL[:,0]]
        dX = XY[EL[:,1],0]-XY[EL[:,0],0]
        dY = XY[EL[:,1],1]-XY[EL[:,0],1]                      
        alpha = np.arctan(dX/dY)
        dVdX = dV*np.cos(alpha)/dX
        dVdY = dV*np.sin(alpha)/dY 
        dVdXY = np.sqrt(dVdX**2 + dVdY**2)
        df = pd.DataFrame({'dVdXY' : np.hstack((dVdXY,-dVdXY))})
        df['type']=np.hstack((EL[:,0],EL[:,1]))
        gradmag = np.array(df.groupby(['type']).mean())
        gradmag = np.array(gradmag)
        return(gradmag)
    
    def graph_local_median(self,VecToSmooth,ordr = 3):
        ind = self.SG.neighborhood(order = ordr)
        Smoothed = np.zeros(VecToSmooth.shape)
        for j in range(len(ind)):
            ix = np.array(ind[j],dtype=np.int64)
            Smoothed[j] = np.nanmedian(VecToSmooth[ix])
        return(Smoothed)
    

        

