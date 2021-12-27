"""TissueGraph module. 

The module contains two main classes: TissueGraph and TissueMultiGraph plus some accessory functions. 

TissueGraph: the core class used to analyze tissues using Graph representation is TissueGraph 

Tissue graphs can be created either using XY possition of centroids OR by contracting existing graph
to create coarser zone. 

main methods:

    * BuildSpatialGraph - construct graph based on Delauny neighbors
    * ContractGraph - find zones, i.e. spatially continous areas in the graph the same type
    * CondEntopy - calculates the conditional entropy of a graph given types (or uses defaults existing types)
    * watershed - devide into regions based on watershed
    * calcGraphEnvCoherenceUsingTreenomial - estimate spatial local coherence. 
    
Many other accessory functions that are used to set/get TissueGraph information.

TissueMultiGraph: the key organizing class used to create and manage graphs across layers (hence multi)

main methods: 

    * save (and load during __init__ if fullfilename is provided)
    * createCellAndZoneLayers - creates the first pair of Cell and Zone layers of the TMG
    * createCommunitiesAndRegionsLayers - create the second pair of layers (3 and 4) of the TMG
    * addContractedLayerUsingType - the actual addition of layer after contraction, used in each pair of layers that are added. 
    * MapToCellLevel
    
"""

# dependeices
from igraph import *

from scipy.spatial import Delaunay,Voronoi
from collections import Counter

import numpy as np
import pandas as pd
import torch

<<<<<<< HEAD
import itertools
import dill as pickle

from scipy.spatial.distance import squareform
=======
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from shapely.geometry import Polygon

import matplotlib.cm as cm
>>>>>>> 3e75b79bc7ff8b9b075f58714e2df1988c5dc287

# for debuding mostly
import warnings
import time
from IPython import embed

# might move these to Viz, currently used in scatter
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from dredFISH.Analysis.Taxonomy import *
from dredFISH.Visualization.vor import bounding_box_sc, voronoi_polygons

from dredFISH.Visualization.cell_colors import *
from dredFISH.Visualization.vor import * 


##### Some simple accessory funcitons
def CountValues(V,refvals,sz = None,norm_to_one = True):
    """
    CountValues - simple tabulation with default values (refvals)
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
        if fullfilename is not None:
            TMGload = pickle.load(open(fullfilename,'rb'))
            self.Layers = TMGload.Layers
            self.VertexDict = TMGload.VertexDict
            self.EdgeDict = TMGload.EdgeDict
            self.ScalarEnvSize = TMGload.ScalarEnvSize
            self.Kvec = TMGload.Kvec
        else: 
            self.Layers = list()
            self.VertexDict = None
            self.EdgeDict = None
            self.ScalarEnvSize = 10
            self.Kvec = np.ceil(np.power(1.5,np.arange(start=6,stop=16,step=0.25))).astype(np.int64)
        return None
    
    def save(self,fullfilename):
        """ pickle and save
        """
        pickle.dump(self,open(fullfilename,'wb'),recurse=True)
        
    def createCellAndZoneLayers(self,XY,PNMF): 
        """
        Creating cell and zone layers. 
        Cell layer is unique as it's the only one where spatial information is directly used with Voronoi
        Zone layer is calculated based on optimization of cond entropy betweens zones and types. 
        """
        # creating first layer - cell tissue graph
        TG = TissueGraph()
        TG.LayerType = "cells"
        TG.BuildSpatialGraph(XY)
        
        # cluster cell types optimally
        celltypes = Taxonomy.RecursiveLeidenWithTissueGraphCondEntropy(PNMF,TG,metric = 'cosine',single_level=True)
        
        # add types and key data
        TG.Type = celltypes
        TG.TX.data = PNMF
        
        # build a tree
        TG.TX.BuildTree()
        
        # add layer
        self.Layers.append(TG)
        
        # contract and create the Zone graph
        self.addContractedLayerUsingType(PNMF,celltypes)
        self.Layers[-1].LayerType = 'isozones'
        
        return None
    
    def createCommunitiesAndRegionLayers(self,ordr=3):
        """
        Creating community and region layers. 
        Community layer is created based on watershed using local env coherence. 
        Region layer is calculated based on optimization of cond entropy betweens community and regions. 
        """
        # find existing environment 
        Env = self.Layers[-1].extractEnvironments(ordr=ordr)
        
        # get edge and node spatial coherence scores
        (EdgeWeight,NodeWeight) = self.Layers[-1].calcGraphEnvCoherenceUsingTreenomial(Env)
        
        # perform watershed with seeds as "peaks" in NodeWeight graph
        HoodId = self.Layers[-1].watershed(EdgeWeight,NodeWeight)
                
        # add Community layers: 
        self.addContractedLayerUsingType(Env,HoodId)
        self.Layers[-1].LayerType = 'isozones'
        
        # calculate true environments vector for all communities: 
        WatershedEnvs = np.zeros((self.Layers[-1].N,self.Layers[-2].Ntypes))
        ZoneTypes = self.Layers[-2].Type.astype(np.int64)
        unqTypes = np.unique(ZoneTypes)
        for i in range(self.Layers[-1].N):
            ix = np.where(self.Layers[-1].UpstreamMap==i)
            WatershedEnvs[i,:] = CountValues(ZoneTypes[ix].astype(np.int64),unqTypes,self.Layers[-2].NodeSize[ix],norm_to_one=False)
            
        # calcualte pairwise distances between environments using treenomialJSD
        cmb = np.array(list(itertools.combinations(np.arange(WatershedEnvs.shape[0]), r=2)))
        WatershedEnvs = torch.from_numpy(WatershedEnvs)
        cmb = torch.from_numpy(cmb)
        D = treenomialJSD(WatershedEnvs[cmb[:,0],:],WatershedEnvs[cmb[:,1],:])
        Dsqr = squareform(D)
        commtypes = Taxonomy.RecursiveLeidenWithTissueGraphCondEntropy(Dsqr,self.Layers[-1],metric = 'precomputed',single_level=True)
                
        self.addContractedLayerUsingType(WatershedEnvs,commtypes)
    
    def addContractedLayerUsingType(self,Env,TypeVec):
        """
        addContractedLayerUsingType - uses TypeVec to contract the last graph in self (i.e. TMG) 
                                      calcualtes a few more things it needs and append the new layer to self
        """
        # create the graph
        EG = self.Layers[-1].ContractGraph(TypeVec)
        df = pd.DataFrame(data = Env)
        df['type']=EG.UpstreamMap
        avg = df.groupby(['type']).mean()
        
        EG.TX.data = np.array(avg)
        EG.TX.BuildTree()
              
        # add layer
        self.Layers.append(EG)
                    
    def MapToCellLevel(self,lvl,VecToMap = None,return_ix = False):
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
    
    @property
    def N(self):
        return([L.N for L in self.Layers])
    
    @property
    def Ntypes(self):
        return([L.Ntypes for L in self.Layers])
    
    @property
    def CondEntropy(self):
        return([L.CondEntropy() for L in self.Layers])
    
    def scatter(self,lvl = None):
        """
        Simple scatter using cell level XY colored by actual level type. 
        """
        
        if lvl is None:
            for i in range(len(self.Layers)):
                self.scatter(i)
            return None
        
        myenvtype = self.MapToCellLevel(lvl)
        unqenv = np.unique(myenvtype)
        colors = cm.rainbow(np.linspace(0, 1, len(unqenv)))
        colors = colors[np.random.permutation(colors.shape[0]),:]
        envcolors = colors[myenvtype,:]
        plt.figure(figsize=(15, 15))

        plt.scatter(x=self.Layers[0].X,y=self.Layers[0].Y,s=2,c=envcolors)
        plt.show()
        
    
class TissueGraph:
    """TissueGraph - main class responsible for maximally informative biocartography.
                     class 
    
            
    """
    def __init__(self):
        self.LayerType = None
        self.MaxEnvSize = 500
        self.MinEnvSize = 10
        
        self.EnvSize = None
        
        self.nbrs = None
        self._G = None
        self.UpstreamMap = None
        self.TX = Taxonomy()
        
        self.BoundingBox = None 
        self.Tri = None 

        return None
        
    @property
    def N(self):
        """Size of the tissue graph
            internally stored as igraph size
        """
        if self._G is not None:
            return len(self._G.vs)
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with BuildSpatialGraph')
    
    @property
    def G(self):
        if self._G is not None:
            return self._G
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with BuildSpatialGraph')
    
    @property
    def NodeSize(self):
        if self._G is not None:
            return np.asarray(self._G.vs['Size'])
        else: 
            raise ValueError('Graph was not initalized, please build a graph first with BuildSpatialGraph or ContractGraph methods')
    
    @NodeSize.setter
    def NodeSize(self,Nsz):
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
    
    def BuildSpatialGraph(self,XY):
        """
        BuildSpatialGraph will create an igrah using Delaunay triangulation

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
<<<<<<< HEAD
      
=======
        
        # initialize bounding box
        _, bb = bounding_box_sc(XY)
        self.BoundingBox = Polygon(bb)

        diameter = np.linalg.norm(bb.ptp(axis=0))
        vp = list(voronoi_polygons(Voronoi(XY), diameter))
        vp = [p.intersection(self.BoundingBox) for p in vp]
        self.Tri = {pdx: p for pdx, p in enumerate(vp)}
        
>>>>>>> 3e75b79bc7ff8b9b075f58714e2df1988c5dc287
        # set up names
        self._G.vs["name"]=list(range(self.N))
        return(self)
    
    def plot(self, color_dict={}, cell_type=None, fpath=None, graph_params=None):
        """
        Plot cell type or zone as a Voronoi diagram

        Input
        -----
        cell_type : list of cell labels 
        color_dict : mapping of cell type to color space
        graph_params : current matplotlib plot parameters passed via a dictionary
            figsize
            border_color, border_alpha, border_lw, border_ls
            poly_alpha
            inner, inner_alpha, inner_lw, inner_ls
            scatter
            scatter_color, scatter_alpha, scatter_size
        fpath : file path for figure to be saved to 

        Output
        ------
        fig : graphic for plotting if not provided with file path for saving plot 
        """

        # default graphing parameters 
        if graph_params is None:
            graph_params = {}

        graph_param_defaults = {}
        graph_param_defaults["figsize"] = (12,8)
        graph_param_defaults["border_color"] = "black"
        graph_param_defaults["border_alpha"] = 1
        graph_param_defaults["border_lw"] = 1
        graph_param_defaults["border_ls"] = "solid"
        graph_param_defaults["poly_alpha"] = 0.5
        graph_param_defaults["inner"] = False
        graph_param_defaults["inner_alpha"] = 1
        graph_param_defaults["inner_lw"] = 0.75
        graph_param_defaults["inner_ls"] = "dotted"
        graph_param_defaults["scatter"] = False
        graph_param_defaults["scatter_color"] = "black"
        graph_param_defaults["scatter_alpha"] = 0.05
        graph_param_defaults["scatter_size"] = 3

        for x in graph_param_defaults:
            if x not in graph_params:
                graph_params[x] = graph_param_defaults[x]

        if cell_type is None:
            cell_type = self.Type

        faces = []
        colors = []
        color_segments = []
        patches = []
        segments = []

        for k in self.Tri:
            p = self.Tri[k]
            
            if p.area == 0:
                continue
            if hasattr(p, "geoms"):
                for subp in p.geoms:
                    coords = subp.exterior.coords
                    faces += [coords]
                    colors += [color_dict[cell_type[k]]]
            else:
                coords = p.exterior.coords
                faces += [coords]
                colors += [color_dict[cell_type[k]]]

        # helper function to convert Polygon coordinates to segments 
        get_segments = lambda x, y: [[[x[i], y[i]], [x[i + 1], y[i + 1]]] for i in range(len(x) - 1)]

        for i in range(len(faces)):
            patches.append(PolygonPatch(Polygon(faces[i]), fc=colors[i], ec=colors[i], lw=0.2, alpha=graph_params["poly_alpha"], zorder=1))
            x, y = zip(*faces[i])
            new_segments = get_segments(x, y)
            segments += new_segments
            color_segments += [colors[i] for idx in range(len(new_segments))]

        bord_faces = []
        bord_segments = []
        if self.UpstreamMap is not None and graph_params["inner"]:

            if self.UpstreamMap is None:
                raise ValueError("Cannot plot inner-enviroments for base level")

            for env in range(max(self.UpstreamMap) + 1):
                polygon_idx = np.where(self.UpstreamMap == env)[0]

                if len(polygon_idx) == 0:
                    continue

                aggr_p = self.Tri[polygon_idx[0]]
                for k in polygon_idx:
                    p = self.Tri[k]
                    aggr_p = aggr_p.union(p)

                if aggr_p.area == 0:
                    continue

                if hasattr(aggr_p, "geoms"):
                    for subp in aggr_p.geoms:
                        coords = subp.exterior.coords
                        bord_faces += [coords]
                else:
                    coords = aggr_p.exterior.coords
                    bord_faces += [coords]

            for i in range(len(bord_faces)):
                x, y = zip(*bord_faces[i])
                bord_segments += get_segments(x, y)

        if graph_params["scatter"]:
            graph_params["scatter_alpha"] = 0

        fig, ax = plt.subplots(figsize=graph_params["figsize"])
        ax.add_collection(PatchCollection(patches, match_original=True))
        ax.scatter(x=self.XY[:,0], y=self.XY[:,1],
            c=graph_params["scatter_color"],
            s=graph_params["scatter_size"],
            alpha=graph_params["scatter_alpha"])

        if graph_params["inner"]:
            # outter lines 
            ax.add_collection(LineCollection(bord_segments,
                                colors=graph_params["border_color"],
                                lw=graph_params["border_lw"],
                                alpha=graph_params["border_alpha"],
                                linestyle=graph_params["border_ls"]))
            # inner lines
            ax.add_collection(LineCollection(segments,
                                colors=color_segments,
                                lw=graph_params["inner_lw"],
                                alpha=graph_params["inner_alpha"],
                                linestyle=graph_params["inner_ls"])) 
            
        else:
            # outer lines 
            ax.add_collection(LineCollection(segments,
                                            colors=graph_params["border_color"],
                                            lw=graph_params["border_lw"],
                                            alpha=graph_params["border_alpha"],
                                            linestyle=graph_params["border_ls"]))

        if fpath != None:
            fig.savefig("fpath")
        else:
            return fig
    
    def ContractGraph(self,TypeVec = None):
        """ContractGraph : reduce graph size by merging neighbors of same type. 
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
        df = pd.DataFrame(data = self.NodeSize)
        df['type'] = IxMapping
        ZoneSize = df.groupby(['type']).sum()
        ZoneSize = np.array(ZoneSize).flatten()
        
         
        # create a new Tissue graph by copying existing one, contracting, and updating XY
        ZoneGraph = TissueGraph()
        ZoneGraph._G = self._G.copy()
        ZoneGraph.BoundingBox = copy(self.BoundingBox)
        ZoneGraph.Tri = copy(self.Tri)
        
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
                             

                             
    def TypeFreq(self): 
        """
            TypeFreq: return the catogorical probability for each type
        """
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count frequencies")
        unqTypes = np.unique(self.Type)
        Ptypes = CountValues(self.Type,unqTypes,self.NodeSize)
        
        return Ptypes,unqTypes
    
    # TODO: this should use NodeSize in case we are calculating this on the contracted graph
    
                             
    def CondEntropy(self):
        """
        CondEntropy: calculate conditional entropy of the tissue graph
                     cond entropy is the difference between graph entropy based on pagerank and type entropy
        """
        Pzones = self.NodeSize
        Pzones = Pzones/np.sum(Pzones)
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        # validate that type exists
        if self.Type is None: 
            raise ValueError("Can't calculate cond-entropy without Types, please check")
            
        Ptypes = self.TypeFreq()[0] 
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        CondEntropy = Entropy_Zone-Entropy_Types
        return(CondEntropy)
    
    def extractEnvironments(self,ordr = 4):
        """
            returns the categorical distribution of neighbors for each vertex in the graph 
            distance is determined by ordr parameters that is passed to igraph neighberhood method. 
        """
        unqlbl = np.unique(self.Type)
        ind = self.G.neighborhood(order = ordr)
        
        unqlbl = np.unique(self.Type)
        Env = np.zeros((self.N,len(unqlbl)))
        for i in range(self.N):
            Env[i,:]=CountValues(self.Type[ind[i]],unqlbl,norm_to_one = False)
        
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
        
    def calcGraphEnvCoherenceUsingTreenomial(self,Env):
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
        
    def calcEntAtDifferentLeidenRes(self,G): 
        Rvec = np.logspace(-1,3,250)
        Ent = np.zeros(Rvec.shape)
        for i in range(len(Rvec)):
            TypeVec = G.community_leiden(resolution_parameter=Rvec[i],objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Ent[i] = self.ContractGraph(TypeVec).CondEntropy()
            
        return (Rvec,Ent)