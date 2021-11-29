"""TissueGraph functions

A collection of files to analyze tissues using Graph representation

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

from sklearn.neighbors import NearestNeighbors

# from Viz.cell_colors import *
# from Viz.vor import *

##### Some simple accessory funcitons
def funcKL(P,Q): 
    ix = P>0
    return(np.sum(P[ix]*np.log2(P[ix]/Q[ix])))

def funcJSD(P,Q):
    M=0.5*P + 0.5*Q
    return(0.5*funcKL(P,M) + 0.5*funcKL(Q,M))

def CountValues(V,refvals):
    Cnt = Counter(V)
    cntdict = dict(Cnt)
    missing = list(set(refvals) - set(V))
    cntdict.update(zip(missing, np.zeros(len(missing))))
    return(cntdict)



###### Main TissueGraph classes

# class TissueMultiGraph: 
#     """
#        TisseMultiGraph - responsible for storing multiple layers (each a TissueGraph) and 
#                          the relationships between them. 
#     """
#     def __init__(self):
#         self.Layers
#         self.LayerMappings
#         return None
    
class TissueGraph:
    """TissueGraph - main class responsible for maximally informative biocartography.
                     class 
    
            
    """
    def __init__(self):
        self.MaxEnvSize = 1000
        self._G = None
        self.UpstreamMap = None
        self.Corners = None # at least 3 columns: X,Y,iternal/external,type?    
        self.Lines = None # at least 3 columns: i,j (indecies of Corders),iternal/external,type?
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
        if "Type" not in self._G.vs.attribute_names():
            warnings.warn("Type was not initalized - returning None")
            return(None)
        
        # if we are here, then there is a Type attribute. Just return it, 
        return(self._G.vs["Type"])
        
    
    @Type.setter
    def Type(self, TypeVec):
        """
            Assign Type values, stores them using igraph attributes. 
        """
        # stores Types as graph vertex attributes. 
        self._G.vs["Type"]=TypeVec
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
        
        # TODO (Jonathan) 
        # initalize self.Corners and self.Lines
        
        # set up names
        self._G.vs["name"]=list(range(self.N))
        return(self)
    
    def plot(self, XY=None, cell_type=None, color_dict={}, size=(12,8), inner=False, scatter=False):
        """
        Plot cell type or zone 
        """
        if type(XY) == type(None):
            XY = self.XY
        if type(cell_type) == type(None):
            cell_type=self.Type

        vp = voronoi_intersect_box(XY)

        fig, ax = plt.subplots(figsize=size)

        if type(self.UpstreamMap) == type(None):
            for i in range(len(vp)):
                p = vp[i]
                if p.area == 0:
                    continue
                if hasattr(p, "geoms"):
                    for subp in p.geoms:
                        x, y = zip(*subp.exterior.coords)
                        ax.plot(x, y, color="black",alpha=1,linewidth=1,linestyle="solid")
                        ax.fill(*zip(*subp.exterior.coords),color=color_dict[cell_type[i]],alpha=0.5)
                else:
                    x, y = zip(*p.exterior.coords)
                    ax.plot(x, y, color="black",alpha=1,linewidth=1,linestyle="solid")
                    ax.fill(*zip(*p.exterior.coords),color=color_dict[cell_type[i]],alpha=0.5)
                if scatter == True:
                    ax.scatter(x=XY[i][0],y=XY[i][1],c="black",s=10,alpha=1)
        else:
            for i in range(len(vp)):
                poly_idx = np.where(self.UpstreamMap==i)[0]

                if len(poly_idx) == 0:
                    continue

                if inner == True:
                    for idx in poly_idx:
                        p = vp[idx]

                        if p.area == 0:
                            continue
                        if hasattr(p, "geoms"):
                            for subp in p.geoms:
                                x, y = zip(*subp.exterior.coords)
                                ax.plot(x, y, color=color_dict[cell_type[poly_idx[0]]],alpha=0.5,linewidth=0.75,linestyle="dotted")
                        else:
                            x, y = zip(*p.exterior.coords)
                            ax.plot(x, y, color=color_dict[cell_type[poly_idx[0]]],alpha=1,linewidth=0.75,linestyle="dotted")

                p = vp[poly_idx[0]]
                
                for idx, x in enumerate(poly_idx):
                    if idx == 0:
                        continue
                    p = p.union(vp[x])
                    
                if p.area == 0:
                    continue
                
                if hasattr(p, "geoms"):
                    for subp in p.geoms:
                        x, y = zip(*subp.exterior.coords)
                        ax.plot(x, y, color="black",alpha=0.5,linewidth=1,linestyle="solid")
                        ax.fill(*zip(*subp.exterior.coords),color=color_dict[cell_type[poly_idx[0]]],alpha=0.5)
                else:
                    x, y = zip(*p.exterior.coords)
                    ax.plot(x, y, color="black",alpha=0.5,linewidth=1,linestyle="solid")
                    ax.fill(*zip(*p.exterior.coords),color=color_dict[cell_type[poly_idx[0]]],alpha=0.5)
                if scatter == True:
                    for idx in poly_idx:
                        ax.scatter(x=XY[idx][0],y=XY[idx][1],c="black",s=10,alpha=1)

#     def UpdatedSpatialDataOfContractedGraph(self): 
#         # Updates Corners and Lines (mostly internal/external and possibly type and other data)
        
        
# #     def fplot(self):
# #         return 1
#         # voroni graph, colors by type, edges only external make it nice :)     
    
    
    
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
            entropy = ZG.CondEntropy()
            return(-entropy)
        
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
        if any(TypeVec == None): 
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
                
        ZoneName, ZoneSingleIx, ZoneSize = np.unique(IxMapping, return_counts=True,return_index=True)
        
 
        # create a new Tissue graph by copying existing one, contracting, and updating XY
        ZoneGraph = TissueGraph()
        ZoneGraph._G = self._G.copy()
        
        comb = {"X" : "mean",
               "Y" : "mean",
               "Type" : "ignore",
               "name" : "ignore"}
        
        ZoneGraph._G.contract_vertices(IxMapping,combine_attrs=comb)
        ZoneGraph._G.vs["Size"] = ZoneSize
        ZoneGraph._G.vs["name"] = ZoneName
        ZoneGraph._G.vs["Type"] = TypeVec[ZoneSingleIx]
        ZoneGraph.UpstreamMap = IxMapping
        
        return(ZoneGraph)
                             
    @property    
    def Ntypes(self): 
        """ 
            Ntypes: returns number of unique types in the graph
        """ 
        if self.Type == None: 
            raise ValueError("Type not yet assigned, can't count how many")
        return(len(numpy.unique(self.Type)))
                             
    def TypeFreq(self): 
        """
            TypeFreq: return the catogorical probability for each type
        """
        if self.Type == None: 
            raise ValueError("Type not yet assigned, can't count frequencies")
        unqTypes = np.unique(self.Type) 
        cntdict = CountValues(self.Type,unqTypes)
        Ptypes = np.array([cntdict.get(k) for k in sorted(cntdict.keys())])
        Ptypes=Ptypes/np.sum(Ptypes)
        return Ptypes,unqTypes
                             
    def CondEntropy(self):
        """
        CondEntropy: calculate conditional entropy of the tissue graph
                     cond entropy is the difference between graph entropy based on pagerank and type entropy
        """
        Pzones = self.NodeSize
        Pzones = Pzones/np.sum(Pzones)
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        # validate that type exists
        if self.Type == None: 
            raise ValueError("Can't calculate cond-entropy without Types, please check")
            
        Ptypes = self.TypeFreq()[0] 
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        CondEntropy = Entropy_Zone-Entropy_Types
        return(CondEntropy)
    
    def LocalTypeFreq(self):
        # first, create the list of types (unique values) we care about
        Ptypes,UnqTypes = self.TypeFreq()
        
        # create nearest neighbors data
        nbrs = NearestNeighbors(n_neighbors=self.MaxEnvSize, algorithm='auto').fit(self.XY)
        distances, indices = nbrs.kneighbors(self.XY)
        
        # use these to build local frequencies
        LocalFreq = np.zeros((self.MaxEnvSize,self.N,len(UnqTypes)))
        for k in range(self.MaxEnvSize): 
            ix = indices[:,0:k]
            f = lambda ind : funcKL(CountValues(self.Type[ind],UnqTypes),Ptypes)
            LocalFreq[k,:,:]=np.apply_along_axis(f,ix,axis=1)

        return LocalFreq
    
    
    def calcSpatialCoherencePerVertex(self): 
        """
        calcSpatialCoherencePerVertex calculates the information score (KL(sample,global) - KL(random,global)) for increasing size of environment. 
                                      within an environment, cells are sorted eclidean distance.  
                                   
                                      Calculations are based on KL between env and all minus KL of permuted
                                      the difference to global is "Signal" and we subtract from it (in log scale) the 
                                      samping noise (binomial and all that...)
                                   
        Inputs: 
                    none, everything is in self 
                    
        Outputs: 
                    KLdiff       : a matrix of scores as function of env size (score = KL to global 0 KL of random to global) 
                   
                   
        Algorithm: 
                    First we establish a baseline of randomness, what is the KL div of random samples of increasing sizes. 
                    Then for each vertex check for increasing distances what is the differene between the "signal" defined as KL to global 
                    and the "noise" defined by KL of random sample (accounting for sample size effectively).  
                    
                    
        """

        # global type distribution that we'll compare to
        Ptypes,UnqTypes = self.TypeFreq()
                          
        # First create an array such that for each node, we get the sampling noise effect 
        iter = 3 # repeat sampling a few times, i.e. expected values over 
        rndKL = np.zeros(max(avgsz))
        for k in range(max(avgsz)): 
            for i in range(iter): 
                Psample = CountValues(np.random.choice(self.Type,size=k),UnqTypes)
                rndKL[k]=rndKL[k]+funcKL(Psample,Ptypes)
        
        rndKL=rndKL/iter            
    
        # now find the KL from global for actual data
        LocalFreq = self.LocalTypeFreq()
        KL2g = numpy.apply_over_axes(lambda P : funcKL(P,Ptypes), LocalFreq, axes)
        
        KLdiff = KL2g - KLsampling
                       
        return(KLdiff)               
        
        
        

        
        
        
        
