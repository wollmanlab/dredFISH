"""TissueGraph 

Main class used to analyze tissues using Graph representation

Tissue graphs can be created either using XY possition of centroids OR by contracting existing graph
to create coarser zone. 

This file should be imported as a module and contains the following
main functions:

    * BuildSpatialGraph - construct graph based on Delauny neighbors
    * ContractGraph - find zones, i.e. spatially continous areas in the graph the same type
    * CondEntopy - calculates the conditional entropy of a graph given types (or uses defaults existing types)
    
Many other accessory functions that are used to set/get TissueGraph information. 
    
"""

# dependeices
from igraph import *
from scipy.spatial import Delaunay,Voronoi
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize_scalar
import scanpy as sc

from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from shapely.geometry import Polygon

import matplotlib.cm as cm

from IPython import embed

from sklearn.neighbors import NearestNeighbors

from dredFISH.Analysis.Taxonomy import *
from dredFISH.Visualization.vor import bounding_box_sc, voronoi_polygons

from scipy.special import rel_entr
import torch
import time
import pdb

from dredFISH.Visualization.cell_colors import *
from dredFISH.Visualization.vor import * 

##### Some simple accessory funcitons
def CountValues(V,refvals,sz = None):
    Cnt = Counter(V)
    if sz is not None: 
        for i in range(len(V)): 
            Cnt.update({V[i] : sz[i]-1}) # V[i] is already represented, so we need to subtract 1 from sz[i]  
            
    cntdict = dict(Cnt)
    missing = list(set(refvals) - set(V))
    cntdict.update(zip(missing, np.zeros(len(missing))))
    Pv = np.array([cntdict.get(k) for k in sorted(cntdict.keys())])
    Pv=Pv/np.sum(Pv)
    return(Pv)



###### Main TissueGraph classes

class TissueMultiGraph: 
    """
       TisseMultiGraph - responsible for storing multiple layers (each a TissueGraph) and 
                         the relationships between them. 
    """
    def __init__(self):
        self.Layers = list()
        self.VertexDict = None
        self.EdgeDict = None
        self.ScalarEnvSize = 10
        self.Kvec = np.ceil(np.power(1.5,np.arange(start=6,stop=16,step=0.25))).astype(np.int64)
        return None
    
    def createCellAndZoneLayers(self,XY,PNMF,initRes = None): 
        
        # creating first layer - cell tissue graph
        TG = TissueGraph()
        TG.BuildSpatialGraph(XY)
        
        # cluster cell types optimally
        celltypes = Taxonomy.RecursiveLeidenWithTissueGraphCondEntropy(PNMF,TG,metric = 'cosine',single_level=True,initRes = initRes)
        TG.Type = celltypes
        TG.TX.data = PNMF
        
        # build a tree
        TG.TX.BuildTree()
        
        # add layer
        self.Layers.append(TG)
        
        # contract and create the Zone graph
        ZG = TG.ContractGraph()
        
        # add average of PNMF as data for each zone. 
        df = pd.DataFrame(data = PNMF)
        df['type']=ZG.UpstreamMap
        avg = df.groupby(['type']).mean()
        unqTypes = avg.index
        ZG.TX.data = np.array(avg)
        
        ZG.TX.linkage = TG.TX.linkage
        ZG.MaxEnvSize = max(self.Kvec)
        
        # groundwork toward environments: find optimal env size for each isozone: 
        # (KLdiff,KL2g,KLsampling) = ZG.calcSpatialCoherencePerVertex(self.Kvec)
        ZG.EnvSize = self.ScalarEnvSize
        self.Layers.append(ZG)
        return None
        
    def addEnvironmentLayers(self,n=0,verbose = True):    
        """ add multiple layers, either until we are out of entropy, or predefined number n
        """
        cnt=0
        start = time.time()
        if n<=0: 
            Ent = self.Layers[-1].CondEntropy()
            while Ent>0:
                self.addEnvironmentLayer()
                Ent = self.Layers[-1].CondEntropy()
                cnt += 1
                if verbose:
                    print(f"Env Layer: {cnt} Entropy: {Ent} Time: {time.time()-start:.2f}")
        else: 
            for i in range(n):
                self.addEnvironmentLayer()
                cnt += 1
                if verbose:
                    Ent = self.Layers[-1].CondEntropy()
                    print(f"Env Layer: {cnt} Entropy: {Ent} Time: {time.time()-start:.2f}")
        
    def addEnvironmentLayer(self,Cvec = np.arange(start=5,stop=16,step=5)): 
        # check that cell/zone was already created. 
        if len(self.Layers)==0: 
            raise ValueError('Please initialize MultiLayer graph by creating Cell and isozone layers first')
        
        # Extract environments from existing last graph
        Env = self.Layers[-1].extractEnvironments(multilevel_freq = False)
        Env = Env*self.ScalarEnvSize
        Env = np.round(Env)
        Env[Env>self.ScalarEnvSize]=self.ScalarEnvSize
        
        # cluster using this environment
        envtypes = Taxonomy.TreenomialMixtureModelWithCondEntropy(Env,self.Layers[-1],Cvec)
        
        # create the graph
        EG = self.Layers[-1].ContractGraph(envtypes)
        
        # add average of Env as data for each zone. 
        df = pd.DataFrame(data = Env)
        df['type']=EG.UpstreamMap
        avg = df.groupby(['type']).mean()
        EG.TX.data = np.array(avg)
        
        EG.TX.BuildTree()

        # find environments
        # (KLdiff,KL2g,KLsampling) = EG.calcSpatialCoherencePerVertex(self.Kvec)
        EG.EnvSize = self.ScalarEnvSize
        
        # add layer
        self.Layers.append(EG)
        return None
        
    def MapTypeToCellLevel(self,lvl):
        TypeVec = self.Layers[lvl].Type.astype(np.int64)
        if lvl>0:
            ix=self.Layers[lvl].UpstreamMap
            for i in np.arange(lvl-1,0,-1):
                ix=ix[self.Layers[i].UpstreamMap]
            TypeVec = TypeVec[ix]
        
        return(TypeVec)
    
    
    def scatter(self,lvl):
        myenvtype = self.MapTypeToCellLevel(lvl)
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
    def State(self):
        """
            returns the state of each node. State could either be a transcriptional basis (PNMF) of cells 
            or the local frequency of types around it (environments)
        """
        if self._G is None: 
            raise ValueError('Graph was not initalized, please build a graph first with BuildSpatialGraph or ContractGraph methods')
        
        X = self._G.vs["State"]
        return(X)
    
    @State.setter
    def State(self,X): 
        """ Updates the transcriptional data (basis)
        """
        
        # validate that graph is initialized
        if self._G is None: 
            raise ValueError('Graph was not initalized, please build a graph first with BuildSpatialGraph or ContractGraph methods')
            
        if self.N != X.shape[0]:
            raise ValueError('Basis matrix must have the same number of rows as nodes in the cell graph')
            
        self._G.vs["State"]=X
    
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
        
        # initialize bounding box
        _, bb = bounding_box_sc(XY)
        self.BoundingBox = Polygon(bb)

        diameter = np.linalg.norm(bb.ptp(axis=0))
        vp = list(voronoi_polygons(Voronoi(XY), diameter))
        vp = [p.intersection(self.BoundingBox) for p in vp]
        self.Tri = {pdx: p for pdx, p in enumerate(vp)}
        
        # set up names
        self._G.vs["name"]=list(range(self.N))
        return(self)
    
    def plot2(self, color_dict={}, cell_type=None, fpath=None, graph_params=None):
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



    def plot(self, XY=None, cell_type=None, color_dict={}, fpath=None, graph_params=None):
        """
        Plot cell type or zone as a Voronoi diagram

        Input
        -----
        XY : coordinates of each cell
        cell_type : list of cell labels 
        color_dict : mapping of cell type to color space
        graph_params : current matplotlib plot parameters passed via a dictionary
            figsize
            border_color, border_alpha, border_lw, border_ls
            poly_alpha
            inner, inner_alpha
            scatter
            scatter_color, scatter_alpha, scatter_size
        fpath : file path for figure to be saved to 

        Output
        ------
        fig : graphic for plotting if not provided with file path for saving plot 

        """

        # default graphing parameters 
        if graph_params is None:
            graph_params={}

        graph_param_defaults ={}
        graph_param_defaults["figsize"]=(12,8)
        graph_param_defaults["border_color"]="black"
        graph_param_defaults["border_alpha"]=1
        graph_param_defaults["border_lw"]=1
        graph_param_defaults["border_ls"]="solid"
        graph_param_defaults["poly_alpha"]=0.5
        graph_param_defaults["inner"]=False
        graph_param_defaults["inner_alpha"]=1
        graph_param_defaults["inner_lw"]=0.75
        graph_param_defaults["inner_ls"]="dotted"
        graph_param_defaults["scatter"]=False
        graph_param_defaults["scatter_color"]="black"
        graph_param_defaults["scatter_alpha"]=1
        graph_param_defaults["scatter_size"]=3

        for x in graph_param_defaults:
            if x not in graph_params:
                graph_params[x]=graph_param_defaults[x]

        if XY is None:
            XY = self.XY
        if cell_type is None:
            cell_type=self.Type

        # generate Voronoi polygons that intersect bounding box
        vp = voronoi_intersect_box(XY)

        fig, ax = plt.subplots(figsize=graph_params["figsize"])

        def plot_single_poly(coords, ax, graph_params, color="blue", inner=False):
            """
            Helper function to plot a polygon using graph params

            Input
            -----
            coords : coordinates of polygon edges
            ax : figure axis for plotting  
            graph_params : graph parameters as a dictionary
            color : color of polygon
            inner : should inner polygons be colored (specific to zone plots)
            """

            nofill = False
            if inner:
                nofill = True
                border_color = color
                border_alpha = graph_params["inner_alpha"]
                border_lw = graph_params["inner_lw"]
                border_ls = graph_params["inner_ls"]
            else:
                border_color = graph_params["border_color"]
                border_alpha = graph_params["border_alpha"]
                border_lw = graph_params["border_lw"]
                border_ls = graph_params["border_ls"]
                poly_color = color
                poly_alpha = graph_params["poly_alpha"]

            x, y = zip(*coords)
            ax.plot(x, y, 
                color=border_color,
                alpha=border_alpha,
                linewidth=border_lw,
                linestyle=border_ls)

            if not nofill:
                ax.fill(*zip(*coords),
                    color=poly_color,
                    alpha=poly_alpha)


        # cell level
        if self.UpstreamMap is None:
            for i in range(len(vp)):
                p = vp[i]

                if p.area == 0:
                    continue

                if hasattr(p, "geoms"):
                    for subp in p.geoms:
                        coords = subp.exterior.coords
                        plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[i]])
                else:
                    coords = p.exterior.coords
                    plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[i]])
        # zone and beyond level
        else:
            for i in range(len(vp)):
                poly_idx = np.where(self.UpstreamMap==i)[0]

                if len(poly_idx) == 0:
                    continue

                agr_p = p = vp[poly_idx[0]]
                
                # iterate through all polygons within some zone 
                for idx in poly_idx:
                    p = vp[idx]
                    agr_p = agr_p.union(p)

                    if p.area == 0:
                        continue
                    # inner polygon lines 
                    if hasattr(p, "geoms"):
                        for subp in p.geoms:
                            coords = subp.exterior.coords
                            plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[poly_idx[0]]], inner=graph_params["inner"])
                    else:
                        coords = p.exterior.coords
                        plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[poly_idx[0]]], inner=graph_params["inner"])
                
                if agr_p.area == 0:
                    continue
                
                # borders
                if hasattr(agr_p, "geoms"):
                    for subp in agr_p.geoms:
                        coords = subp.exterior.coords
                        plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[poly_idx[0]]])
                else:
                    coords = agr_p.exterior.coords
                    plot_single_poly(coords, ax, graph_params, color=color_dict[cell_type[poly_idx[0]]])

        if graph_params["scatter"]:
            ax.scatter(x=XY[:,0],y=XY[:,1],
                c=graph_params["scatter_color"],
                s=graph_params["scatter_size"],
                alpha=graph_params["scatter_alpha"])
        
        if fpath != None:
            fig.savefig("fpath")
        else:
            return fig
    
    def Cluster(self,data):
        """
            Find optimial clusters (using recursive leiden)
            optimization is done on resolution parameter and hierarchial clustering
            
        """
        verbose = True
        def OptLeiden(res,datatoopt,ix,currcls):
            """Basic optimization routine for Leiden resolution parameter in Scanpy
            """
            # calculate leiden clustering
            sc.tl.leiden(datatoopt,resolution=res)
            TypeVec = np.asarray(datatoopt.obs['leiden'])
            
            # merge TypeVec with full cls vector
            dash = np.array((1,),dtype='object')
            dash[0]='_' 
            newcls = currcls.copy()
            newcls[ix] = newcls[ix]+dash+TypeVec
            
            ZG = self.ContractGraph(newcls)
            Entropy = ZG.CondEntropy()
            return(-Entropy)
        
        sc.pp.neighbors(data, n_neighbors=10, n_pcs=0)
        if verbose: 
            print(f"Calling initial optimization")
        emptycls = np.asarray(['' for _ in range(self.N)],dtype='object')
        sol = minimize_scalar(OptLeiden, args = (data,np.arange(self.N),emptycls),
                                         bounds = (0.1,30), 
                                         method='bounded',
                                         options={'xatol': 1e-1, 'disp': 3})
        initRes = sol['x']
        ent_best = sol['fun']
        if verbose: 
            print(f"Initial entropy was: {-ent_best} number of evals: {sol['nfev']}")
        sc.tl.leiden(data,resolution=initRes)
        cls = np.asarray(data.obs['leiden'])
        if verbose:
            u=np.unique(cls)
            print(f"Initial types found: {len(u)}")
        
        def DescentTree(cls,ix,ent_best):
            
            dash = np.array((1,),dtype='object')
            dash[0]='_' 
            unqcls = np.unique(cls[ix])
            
            if verbose: 
                print(f"descending the tree")
            for i in range(len(unqcls)):
                newcls = cls.copy()
                ix = np.where(cls == unqcls[i])
                smalldata = data[ix]
                sc.pp.neighbors(smalldata, n_neighbors=10, n_pcs=0)
                sol = minimize_scalar(OptLeiden, args = (smalldata,ix,newcls),
                                                 bounds = (0.1,30), 
                                                 method='bounded',
                                                 options={'xatol': 1e-1, 'disp': 2})
                ent = sol['fun']
                if verbose: 
                    print(f"split groun {i} into optimal parts, entropy {-ent}")
                if ent < ent_best:
                    if verbose: 
                        print(f"split improves entropy - descending")
                    ent_best=ent
                    sc.tl.leiden(smalldata,resolution=sol['x'])
                    nxtlvl = np.asarray(smalldata.obs['leiden'])
                    newcls[cls==unqcls[i]] = cls[cls==unqcls[i]]+dash+nxtlvl
                    cls = newcls.copy()
                    cls = DescentTree(cls,ix,ent_best)
                    
            return(cls)
        
        finalCls = DescentTree(cls,np.arange(self.N),ent_best) 
    
        return(finalCls)
    
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
        ZoneSize = np.bincount(self.Type.astype(np.int64), weights=self.NodeSize)
        
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
                             
    @property    
    def Ntypes(self): 
        """ 
            Ntypes: returns number of unique types in the graph
        """ 
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count how many")
        return(len(np.unique(self.Type)))
                             
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
    
    def LocalTypeFreq(self):
        """
        Calculates the type frequency from 1 to self.MaxEnvSize for each node. 
        As this is a lot of probabilities, we store those as sparse matrix (torch.sparse)
        
        This method is similar to what "calcSpatialCoherencePerVertex" is doing but returning 
        the intermediate environment as sprase array where as the "calcSpatialCoherencePerVertex"
        uses these frequencies on the fly to calculate KL and local environment length scales
        """
        # first, create the list of types (unique values) we care about
        Ptypes,UnqTypes = self.TypeFreq()
        
        # create nearest neighbors data
        (distances,indices) = self.SpatialNeighbors
        
        # use these to build local frequencies
        LocalFreq = np.zeros((self.N,len(UnqTypes),self.MaxEnvSize))
        unq,TypeCode = np.unique(self.Type, return_inverse=True)
        TypeCode = TypeCode.astype(np.int_)
        TypeInt = TypeCode[indices]
        
        LF = torch.sparse_coo_tensor(size=(self.N,len(UnqTypes),self.MaxEnvSize))
        Totals = np.zeros((self.N,len(UnqTypes)),dtype="int16")
        for k in range(self.MaxEnvSize): 
            np.add.at(Totals,(np.arange(self.N),TypeInt[:,k]),1)
            SpInd = np.array([np.arange(self.N),TypeInt[:,k],k*np.ones(self.N)]).astype("int64")
            SpVal = Totals[np.arange(self.N),TypeInt[:,k]]/(k+1)
            LF = LF + torch.sparse_coo_tensor(SpInd,SpVal,size=(self.N,len(UnqTypes),self.MaxEnvSize))
        
        return LF
    
    @property
    def SpatialNeighbors(self):
        if self.nbrs is None: 
            # create nearest neighbors data
            mxnbrs = min(self.N,self.MaxEnvSize+1)
            nbrs = NearestNeighbors(n_neighbors=mxnbrs, algorithm='auto').fit(self.XY)
            distances, indices = nbrs.kneighbors(self.XY)
            indices = np.array(indices)
            indices = indices[:,1:indices.shape[1]]
            distances = np.array(distances)
            distances = distances[:,1:distances.shape[1]]
            self.nbrs = {'indices' : indices,
                         'distances' : distances}
        else: 
            indices = self.nbrs['indices']
            distances = self.nbrs['distances']
            
        return (distances,indices)
    
    def calcSpatialCoherencePerVertex(self,Kvec=None): 
        """
        calcSpatialCoherencePerVertex calculates the information score (KL(sample,global) - KL(random,global)) for increasing size of environment. 
                                      within an environment, cells are sorted eclidean distance.  
                                   
                                      Calculations are based on KL between env and all minus KL of permuted
                                      the difference to global is "Signal" and we subtract from it (in log scale) the 
                                      samping noise (binomial and all that...)
                                   
        Inputs: 
                    Kvec - optional vector of distnaces around each cell to test. In not provided will test all distances up to MaxEnvSize 
                    
        Outputs: 
                    KLdiff       : a matrix of scores as function of env size (score = KL to global 0 KL of random to global) 
                   
                   
        Algorithm: 
                    First we establish a baseline of randomness, what is the KL div of random samples of increasing sizes. 
                    Then for each vertex check for increasing distances what is the differene between the "signal" defined as KL to global 
                    and the "noise" defined by KL of random sample (accounting for sample size effectively).  
                    
                    
        """
        verbose = True
        start = time.time()
        if verbose: 
            print("Estimating sampling effect (KLsampling)")
        # global type distribution that we'll compare to
        Ptypes,UnqTypes = self.TypeFreq()
        # include all tree nodes
        Ptypes = Ptypes[np.newaxis,:]
        Ptypes = self.TX.MultilevelFrequency(Ptypes)
        
        # First create an array such that for each node, we get the sampling noise effect 
        iter = 30 # repeat sampling a few times, i.e. expected values over 

        if Kvec is None: 
            Kvec = np.arange(self.MaxEnvSize)
            
        rndKL = np.zeros(len(Kvec))

        for i in range(len(Kvec)):
            k=Kvec[i]
            TypeSample = np.reshape(np.random.choice(self.Type,size = (k+1)*iter) , (-1, iter))
            P = np.apply_along_axis(lambda V : CountValues(V,UnqTypes),axis=0,arr = TypeSample)
            
            # P=P.T
            P = self.TX.MultilevelFrequency(P.T)
            
            mat = rel_entr(P, Ptypes)

            rndKLiter = np.sum(mat, axis=1)/np.log(2)
            rndKL[i]=rndKLiter.mean()
    
        KLsampling = np.broadcast_to(rndKL, (self.N,len(rndKL)))
        if verbose: 
            print(f"Calculation tool {time.time()-start:.2f}")
        
        
        # now find the KL from global for actual data
        Ptypes = Ptypes.reshape(-1,1)
        Ptypes = Ptypes.T
        
        if verbose: 
            print(f"Calculating nearest neighbors up to {self.MaxEnvSize}")
        
        # create nearest neighbors data
        (distances,indices) = self.SpatialNeighbors
            
        unq,TypeCode = np.unique(self.Type, return_inverse=True)
        TypeCode = TypeCode.astype(np.int_)
        TypeInt = TypeCode[indices]
        
        TypeInt = torch.from_numpy(TypeInt)
        
        if verbose: 
            print(f"Calculation tool {time.time()-start:.2f}")
            
        KL2g = np.zeros((self.N,len(Kvec)))
        
        # create the mapping betweeb leaf type and ALL tree types
        treemat = self.TX.TreeAsMat()
        treemap = np.zeros(treemat.shape[1],dtype='object')
        treemapsz = np.zeros(treemap.shape,dtype='object')
        treemapweight = np.zeros(treemap.shape,dtype='object')
        for i in range(treemat.shape[1]):
            treemap[i] = np.flatnonzero(treemat[:,i]).astype(np.int64)
            treemapsz[i] = np.ones(len(treemap[i]),dtype=np.int64)
            treemapweight[i] = treemat[treemap[i],i]
            
        # Convert TypeInt to TypeIntTree that accounts for ALL tree types
        RowTracker = np.zeros(TypeInt.shape,dtype='object')
        TypeIntTree = np.zeros(TypeInt.shape,dtype='object')
        WeightTracker = np.zeros(TypeInt.shape,dtype='object')

        for i in range(TypeIntTree.shape[0]):
            # i_treemapsz = np.array([i*x for x in treemapsz],dtype='object')
            TypeIntTree[i,:] = treemap[TypeInt[i,:]]
            RowTracker[i,:] = i*treemapsz[TypeInt[i,:]]
            WeightTracker[i,:] = treemapweight[TypeInt[i,:]] 
            
        # count by increasing env size by one and each time add 1 to the Total "bean counter"
        # we are adding 1 to the leaf and all "upstream" composite types from the tree
        Totals = torch.zeros((self.N,treemat.shape[0]),dtype=torch.double)

        if verbose: 
            print(f"Calculating KL divergence for all cells per environment size of {TypeInt.shape[1]}")
            
        cnt=0; 
        for i in range(len(Kvec)-1):
            ix=np.arange(start = Kvec[i],stop = Kvec[i+1],step=1)
            print(f"iter: {i} time: {time.time()-start:.2f}")
            ix_type = np.hstack(np.hstack(TypeIntTree[:,ix]))
            ix_type = torch.from_numpy(ix_type)
            ix_rows = np.hstack(np.hstack(RowTracker[:,ix])) 
            ix_rows = torch.from_numpy(ix_rows)
            weights = np.hstack(np.hstack(WeightTracker[:,ix]))
            weights = torch.from_numpy(weights)
            
            Totals.index_put_((ix_rows,ix_type),weights, accumulate=True)
            
            # probabilities (each time renormalize all rows to sum=1)
            row_sums = Totals.sum(axis = 1)
            P = Totals / row_sums[:,None]
            
            # element wise KL divergence (p*log(p/q))
            mat = rel_entr(P, Ptypes)
            
            # get entropy in bits
            KL2g[:,i]=np.sum(mat.numpy(), axis=1)/np.log(2)
            
        KLdiff = KL2g - KLsampling
        
        self.EnvSize = np.zeros(self.N,dtype='int64')
        for i in range(self.N):
            self.EnvSize[i]=np.argmax(KLdiff[i,self.MinEnvSize:])+self.MinEnvSize
        
        return (KLdiff,KL2g,KLsampling)               
        

    def extractEnvironments(self,return_freq = True,multilevel_freq = True):
        """
            returns the categorical distrobution of neighbors for each vertex in teh graph 
            distance is determined by self.EnvSize that can be a scalar or a vector length self.N
            categorical distribution is 
        """
        unqlbl = np.unique(self.Type)
        (distances,indices) = self.SpatialNeighbors
        
        
        EnvSize = self.EnvSize
        if np.isscalar(EnvSize): 
            EnvSize = np.ones(self.N,dtype=np.int64)*EnvSize
        
        
        # make sure it's an int so numpy isn't mad at me
        EnvSize = EnvSize.astype(np.int64)
        
        # two options for output:
        # 1) (default) probability vector adapted to account for tree
        # 2) just the neighbor types to be used "raw"
        if return_freq:
            Env = np.zeros((self.N,len(unqlbl)))
            for i in range(self.N):
                Env[i,:]=CountValues(self.Type[indices[i,0:EnvSize[i]]],unqlbl)
    
            # include multi-level information
            if multilevel_freq:
                Env = self.TX.MultilevelFrequency(Env) 
        else:
            Env = list()
            for i in range(self.N):
                Env.append(self.Type[indices[i,0:EnvSize[i]]])
            Env = np.array(Env)
            
        return(Env)
    
    
    
        
        
        
