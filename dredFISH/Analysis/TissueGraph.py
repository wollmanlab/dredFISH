"""TissueGraph module. 

The module contains main classes required for maximally infomrative biocartography (MIB)
MIB includes the following key classes: 

TissueMultiGraph: the key organizing class used to create and manage graphs across layers (hence multi)

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
    * add_contracted_layer_using_type - the actual addition of layer after contraction, used in each pair of layers that are added. 
    * add_geoms
    * add_view
    
Query: 
    * map_to_cell_level -
    * find_max_edge_level - 
    * N
    * Ntypes


 

TissueGraph: the core class used to analyze tissues using Graph representation is TissueGraph 

Tissue graphs can be created either using XY possition of centroids OR by contracting existing graph
to create coarser zone. 

main methods:

    * build_spatial_graph - construct graph based on Delauny neighbors
    * contract_graph - find zones, i.e. spatially continous areas in the graph the same type
    * CondEntopy - calculates the conditional entropy of a graph given types (or uses defaults existing types)
    * watershed - devide into regions based on watershed
    * calc_graph_env_coherence_using_treenomial - estimate spatial local coherence. 
    
Many other accessory functions that are used to set/get TissueGraph information.


    
"""

# dependeices
from igraph import *

from scipy.spatial import Delaunay,Voronoi
from collections import Counter

import numpy as np
import pandas as pd
import torch

import itertools
import dill as pickle

from scipy.spatial.distance import squareform

from matplotlib.collections import LineCollection, PolyCollection
# from descartes import PolygonPatch
from shapely.geometry import Polygon


# for debuding mostly
import warnings
import time
from IPython import embed

# might move these to Viz, currently used in scatter
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from shapely.geometry import MultiPolygon, LineString, MultiLineString, Polygon

from dredFISH.Analysis.Taxonomy import *

from dredFISH.Visualization.vor import bounding_box_sc, voronoi_polygons
from dredFISH.Visualization.cell_colors import *
from dredFISH.Visualization.vor import * 


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

def treenomialJSD(Q1,Q2):
    """
    calcualtes mean JSD of bernulli represenation of taxonomy tree. 
    """
    M = (Q1+Q2)/2

    # Q includes all the p and 1-p for each tree branch, so already doing "double" 
    KL1 = Q1*(torch.log2(Q1/M))
    KL1[Q1==0]=0
    KL2 = Q2*(torch.log2(Q2/M))
    KL2[Q2==0]=0
    JSD = (KL1+KL2)
    
    JSDavg = torch.mean(JSD,axis=1)
    
    return(JSDavg.cpu().detach().numpy())

###### Main TissueGraph classes

class TissueMultiGraph: 
    """
       TisseMultiGraph - responsible for storing multiple layers (each a TissueGraph) and 
                         the relationships between them. 
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
        TG.LayerType = "cells"
        TG.build_spatial_graph(XY)
        
        # cluster cell types optimally
        celltypes = Taxonomy.RecursiveLeidenWithTissueGraphcond_entropy(PNMF,TG,metric = 'cosine',single_level=True)
        
        # add types and key data
        TG.Type = celltypes
        TG.TX.data = PNMF
        
        # build a tree
        TG.TX.BuildTree()
        
        # add layer
        self.Layers.append(TG)
        
        # contract and create the Zone graph
        self.add_contracted_layer_using_type(PNMF,celltypes)
        self.Layers[-1].LayerType = 'isozones'
        
        return None
    
    def create_communities_and_region_layers(self,ordr=3):
        """
        Creating community and region layers. 
        Community layer is created based on watershed using local env coherence. 
        Region layer is calculated based on optimization of cond entropy betweens community and regions. 
        """
        # find existing environment 
        Env = self.Layers[-1].extract_environments(ordr=ordr)
        
        # get edge and node spatial coherence scores
        (EdgeWeight,NodeWeight) = self.Layers[-1].calc_graph_env_coherence_using_treenomial(Env)
        
        # perform watershed with seeds as "peaks" in NodeWeight graph
        HoodId = self.Layers[-1].watershed(EdgeWeight,NodeWeight)
                
        # add Community layers: 
        self.add_contracted_layer_using_type(Env,HoodId)
        self.Layers[-1].LayerType = 'isozones'
        
        # calculate true environments vector for all communities: 
        WatershedEnvs = np.zeros((self.Layers[-1].N,self.Layers[-2].Ntypes))
        ZoneTypes = self.Layers[-2].Type.astype(np.int64)
        unqTypes = np.unique(ZoneTypes)
        for i in range(self.Layers[-1].N):
            ix = np.where(self.Layers[-1].UpstreamMap==i)
            WatershedEnvs[i,:] = count_values(ZoneTypes[ix].astype(np.int64),unqTypes,self.Layers[-2].node_size[ix],norm_to_one=False)
            
        # calcualte pairwise distances between environments using treenomialJSD
        cmb = np.array(list(itertools.combinations(np.arange(WatershedEnvs.shape[0]), r=2)))
        WatershedEnvs = torch.from_numpy(WatershedEnvs)
        cmb = torch.from_numpy(cmb)
        D = treenomialJSD(WatershedEnvs[cmb[:,0],:],WatershedEnvs[cmb[:,1],:])
        Dsqr = squareform(D)
        commtypes = Taxonomy.RecursiveLeidenWithTissueGraphcond_entropy(Dsqr,self.Layers[-1],metric = 'precomputed',single_level=True)
                
        self.add_contracted_layer_using_type(WatershedEnvs,commtypes)
    
    def add_contracted_layer_using_type(self,Env,TypeVec):
        """
        add_contracted_layer_using_type - uses TypeVec to contract the last graph in self (i.e. TMG) 
                                      calcualtes a few more things it needs and append the new layer to self
        """
        # create the graph
        EG = self.Layers[-1].contract_graph(TypeVec)
        df = pd.DataFrame(data = Env)
        df['type']=EG.UpstreamMap
        avg = df.groupby(['type']).mean()
        
        EG.TX.data = np.array(avg)
        EG.TX.BuildTree()
              
        # add layer
        self.Layers.append(EG)
                    
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
            return(VecToMap)
    
    def find_max_edge_level(self):
        """
        determine the maximal level that an edges between two cell still exists in, i.e. it was not contracted
        returns a dict with sorted edge tuples as keys and max level is values.   
        """
        
        # create edge list with sorted tuples (the geom convention)
        edge_list = list()
        for e in self.Layers[0].G.es:
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
        
        # Bounding box with adding bounding box based on cell XY
        # initialize bounding box
        XY = self.Layers[0].XY
        _, bb = bounding_box_sc(XY)
        self.Geoms['BoundingBox'] = Polygon(bb)

        # 
        diameter = np.linalg.norm(bb.ptp(axis=0))
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
        
        # Next section deals with edge line between any two voroni polygons. 
        # The key actions are to 
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
        
        # Points are easy, just use XY, the order is correct :)
        self.Geoms['point'] = XY
            
    def add_view(self,view):
        # add the view to the view dict
        self.Views[view.name]=view
    

class TissueGraph:
    """TissueGraph - main class responsible for maximally informative biocartography.
    """
    def __init__(self):
        
        self.LayerType = None
        
        self.nbrs = None
        self._G = None
        self.UpstreamMap = None
        self.TX = Taxonomy()

        return None
        
    @property
    def N(self):
        """Size of the tissue graph
            internally stored as igraph size
        """
        if self._G is not None:
            return len(self._G.vs)
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with build_spatial_graph')
    
    @property
    def G(self):
        if self._G is not None:
            return self._G
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with build_spatial_graph')
    
    @property
    def node_size(self):
        if self._G is not None:
            return np.asarray(self._G.vs['Size'])
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with build_spatial_graph or contract_graph methods')
    
    @node_size.setter
    def node_size(self,Nsz):
        self._G.vs["Size"]=Nsz
    
    @property
    def XY(self):
        """
            XY : dependent property - will query info from Graph and return
        """
        XY=np.zeros((self.N,2))
        XY[:,0]=self._G.vs["X"]
        XY[:,1]=self._G.vs["Y"]
        return(XY)
        
    @property    
    def X(self):
        """
            X : dependent property - will query info from Graph and return
        """
        return(self._G.vs["X"])
        
    @property
    def Y(self):
        """Y : dependent property - will query info from Graph and return
        """
        return(self._G.vs["Y"])
    
    @property
    def Type(self):
        """Type: returns the types of all vertices.
                internally as vertices attributes
                warns when None. 
        """
        # validate that Type attribute was initialized 
        if self.TX.leaflabels is None:
            warnings.warn("Type was not initalized - returning None")
            return(None)
        
        # if we are here, then there is a Type attribute. Just return it,
        TypeVec = np.asarray(self.TX.leaflabels)
        return TypeVec
        
    
    @Type.setter
    def Type(self, TypeVec):
        """
            Assign Type values, stores them using igraph attributes. 
        """
        # stores Types inside the taxonomy object 
        self.TX.leaflabels=np.asarray(TypeVec)
        return
                             
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

        self._G = Graph(n=XY.shape[0],edges=EL,directed=False).simplify()
        self._G.vs["X"]=XY[:,0]
        self._G.vs["Y"]=XY[:,1]
        self._G.vs["Size"]=np.ones(len(XY[:,1]))
        
        # set up names
        self._G.vs["name"]=list(range(self.N))
        return(self)

    
    def contract_graph(self,TypeVec = None):
        """contract_graph : reduce graph size by merging neighbors of same type. 
            Given a vector of types, will contract the graph to merge vertices that are 
            both next to each other and of the same type. 
        
        Input: TypeVec - a vector of Types for each node. 
               If TypeVec is not provided will attempty to use the Type property of the graph itself. 
        
        Output: a new TissueGraph after vertices merging. 
        """
         # get edge list - work with names and not indexes in case things shift around (they shouldn't),     
        EL = np.asarray(self._G.get_edgelist()).astype("int")
        nm=self._G.vs["name"]
        EL[:,0] = np.take(nm,EL[:,0])
        EL[:,1] = np.take(nm,EL[:,1])
        
        # Figure out which type to use
        if TypeVec is None: 
            TypeVec = self.Type
            
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
        ZoneGraph._G = self._G.copy()
        
        comb = {"X" : "mean",
               "Y" : "mean",
               "Type" : "ignore",
               "name" : "ignore"}
        
        ZoneGraph._G.contract_vertices(IxMapping,combine_attrs=comb)
        ZoneGraph._G.simplify()
        ZoneGraph._G.vs["Size"] = ZoneSize
        ZoneGraph._G.vs["name"] = ZoneName
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
    
    def extract_environments(self,ordr = 4):
        """
            returns the categorical distribution of neighbors for each vertex in the graph 
            distance is determined by ordr parameters that is passed to igraph neighberhood method. 
        """
        unqlbl = np.unique(self.Type)
        ind = self.G.neighborhood(order = ordr)
        
        unqlbl = np.unique(self.Type)
        Env = np.zeros((self.N,len(unqlbl)))
        for i in range(self.N):
            Env[i,:]=count_values(self.Type[ind[i]],unqlbl,norm_to_one = False)
        
        return(Env)
    
    def watershed(self,EdgeWeight,NodeWeight):
        hood = self.G.neighborhood(order=1,mindist=1)
        Init = -np.ones(len(hood))
        Fixed = np.zeros(len(hood))
        NodeAtt = -np.log10(NodeWeight)
        for i in range(len(hood)):
            if np.all(NodeAtt[i] <= NodeAtt[hood[i]]):
                Init[i] = i
                Fixed[i] = 1

        Fixed = Fixed.astype('bool').flatten()
        u,ix_rev = np.unique(Init, return_inverse=True)
        Init=u[ix_rev]
        Init[~Fixed]=-1
        HoodId = self.G.community_label_propagation(weights=EdgeWeight, initial=Init, fixed=Fixed).membership
        HoodId = np.array(HoodId)
        return(HoodId)
        
    def calc_graph_env_coherence_using_treenomial(self,Env):
        # create the treenomial distribution for all envs
        (treemat,nrmtreemat) = self.TX.TreeAsMat(return_nrm = True)
        q_no = np.matmul(Env,treemat.T)
        q_deno = np.matmul(Env,nrmtreemat.T)
        q = q_no/q_deno
        q[np.isnan(q)]=0
        
        # find for each edges the two environments that it connects
        EL = np.asarray(self.G.get_edgelist()).astype("int")
        Q1 = torch.from_numpy(q[EL[:,0],:])
        Q2 = torch.from_numpy(q[EL[:,1],:])
        EdgeWeightTreenomial = treenomialJSD(Q1,Q2)
        
        # create node weights: 
        df = pd.DataFrame(np.hstack((EdgeWeightTreenomial,EdgeWeightTreenomial)))
        df['type']=np.hstack((EL[:,0],EL[:,1]))
        avg = df.groupby(['type']).mean()
        NodeWeightTreenomial = np.array(avg).flatten()
        
        return (EdgeWeightTreenomial,NodeWeightTreenomial)
        
    def calc_entropy_at_different_Leiden_resolutions(self,G): 
        Rvec = np.logspace(-1,3,250)
        Ent = np.zeros(Rvec.shape)
        for i in range(len(Rvec)):
            TypeVec = G.community_leiden(resolution_parameter=Rvec[i],objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Ent[i] = self.contract_graph(TypeVec).cond_entropy()
            
        return (Rvec,Ent)
    
class View:
    def __init__(self,TMG,name=None):
        # each view needs a unique name
        self.name = name
        self.TMG = TMG
        
        # Fundamentally, a view keeps tab of all the type for different geoms
        # and a dictionary that maps these ids to color/shape etc. 
        
        # types, maps each points, line, or polygon to a specific type key. 
        # in these dataframes, the index must match the TMG Geom and different columns store different attributes
        self.line_style = pd.DataFrame()
        self.point_style = pd.DataFrame()
        self.polygon_style = pd.DataFrame()
        self.boundingbox_style = pd.DataFrame()
        
        # colormap is avaliable in case some derived Views needs to use it (for coloring PolyCollection for example)
        self.clrmp = None
        
    def is_empty(self):
        return(self.line_style.empty and self.point_style.empty and self.polygon_style.empty)
    
    def set_view(self): 
        """
        Key abstract method - has to be implemented in the subclass
        signature should always include the TMG (and other stuff if needed)
        """
        raise NotImplementedError()
        
    
    def plot_boundingbox(self): 
        xy=np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T
        plt.plot(xy[:,0],xy[:,1],color=self.boundingbox_style['color'])
    
    
    def plot_points(self): 
        plt.scatter(x=self.Layers[0].X,
                    y=self.Layers[0].Y,
                    s=self.point_style['size'],
                    c=self.point_style['color'])
        
    def plot_polys(self): 
        p = PolyCollection(self.TMG.Geoms['poly'],cmap=self.clrmp)
        p.set_array(self.polygon_style['scalar'])
        ax = plt.gca()
        ax.add_collection(p)
        
    def plot_lines(self): 
        # get lines sorted by key (which is by convention internally sorted)
        segs = [s[1] for s in sorted(self.TMG.Geoms['line'].items())]
        line_segments = LineCollection(segs,
                                       linewidths=self.line_style['width'],
                                       colors=self.line_style['color'])
        ax = plt.gca()
        ax.add_collection(line_segments)
    
    def plot(self,return_fig = False):
        """
        plot the View. 

        (optionally) return the generated fig. 
        
        This method will not be used directly by this View as 
        Out: 
        fig
        """
        
        if self.is_empty():
            raise TypeError('View was not initalized with set_view')
        
        # plotting of different geometries depending on styles that exist in the view
        # current supported geoms are: BoundingBox, lines, poly, points 
        # in each case, there are some assumption about the columns that exists in the view style. 
        # these assumptions are no enforced, so take care! 

        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(111)
        
        if not self.boundingbox_style.empty:
            self.plot_boundingbox()
        
        if not self.point_style.empty: 
            self.plot_points()
            
        if not self.polygon_style.empty:
            self.plot_polys()
            
        if not self.line_style.empty:
            self.plot_lines()
        
        # set up limits and remove frame
        mx = np.max(np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        mn = np.min(np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        ax.set_xlim(mn[0],mx[0])
        ax.set_ylim(mn[1],mx[1])
        ax.axis('off')
        
        if return_fig:
            return fig
        else:
            return None

# for any new view, we derive the class so we have lots of views, each with it's own class so we can keep key attributes and 
# rewrite the different routines for each type of views

class RandomPolygonColorByType(View):
    def __init__(self,TMG,name = "polygons / random colors",lvl = 0):
        super().__init__(TMG,name = f"{name} / level-{lvl}")
        self.lvl = lvl
        
    def set_view(self,TMG):
        cell_types = TMG.map_to_cell_level(self.lvl)
        # create scalar mapping by just using cell_type id
        scalar_mapping = cell_types/np.max(cell_types)
        self.polygon_style['scalar'] = scalar_mapping
        
        # create the colormap
        self.clrmp = ListedColormap(np.random.rand(len(np.unique(cell_types)),3))
        

class RandomPolygonColorByTypeWithLines(RandomPolygonColorByType):
    def __init__(self,TMG,name = "polygons and edges / random colors",lvl = 0):
        super().__init__(TMG,name = name, lvl = lvl)

    def set_view(self):
        # start with polygons in random colors
        super().set_view(TMG)
        edge_lvls = TMG.find_max_edge_level()
        edge_width = [e[1] for e in sorted(edge_lvls.items())]
        self.line_style['width'] = edge_width
        self.line_style['color'] = np.repeat('#48434299',len(edge_width))

class OnlyLines(View):
    def __init__(self,TMG,name = "only lines"):
        super().__init__(TMG,name = name)
        self.edge_width = None
    
    def set_view(self):
        edge_lvls = self.TMG.find_max_edge_level()
        ew = np.array([float(e[1]) for e in sorted(edge_lvls.items())],dtype = 'float')
        self.edge_width = ew
        self.line_style['width'] = self.edge_width
        self.line_style['color'] = np.repeat('#48434299',len(self.edge_width))
        
class RandomPointColorByType(View):
    def __init__(self,TMG,name = "points / random colors",lvl = 0):
        super().__init__(TMG,name = f"{name} / level-{lvl}")
        self.lvl = lvl
        
    def set_view(self):
        
        cell_types = self.TMG.map_to_cell_level(lvl)
        
        # in this simple view, the only thing we do colors points by type with random color
        unqcelltypes = np.unique(cell_types)
        
        # create unique color per type, shuffle, and expand to color per cell
        colors = cm.rainbow(np.linspace(0, 1, len(unqcelltypes)))
        colors = colors[np.random.permutation(colors.shape[0]),:]
        colors = colors[cell_types,:]
        
        # convert to hex and set style
        f_rgb2hex = lambda rgb: '#%02x%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255),int(rgb[3]*255))
        hex_colors = [f_rgb2hex(colors[i,:]) for i in range(colors.shape[0])]
        self.point_style['color'] = hex_colors
        self.point_style['size'] = 2
        self.point_style.index = np.arange(len(cell_types))