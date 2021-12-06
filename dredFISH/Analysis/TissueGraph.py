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

from IPython import embed

from sklearn.neighbors import NearestNeighbors

from dredFISH.Analysis.Taxonomy import *

from scipy.special import rel_entr
import torch
import time

# from Viz.cell_colors import *
# from Viz.vor import *

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
        self.MinEnvSize = 10
        
        self.EnvSize = None
        
        self.nbrs = None
        self._G = None
        self.UpstreamMap = None
        self.TX = Taxonomy()
        
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
        
        ZoneName, ZoneSingleIx, ZoneSize = np.unique(IxMapping, return_counts=True,return_index=True)
        
        # create a new Tissue graph by copying existing one, contracting, and updating XY
        ZoneGraph = TissueGraph()
        ZoneGraph._G = self._G.copy()
        
        comb = {"X" : "mean",
               "Y" : "mean",
               "name" : "ignore"}
        
        ZoneGraph._G.contract_vertices(IxMapping,combine_attrs=comb)
        ZoneGraph._G.vs["Size"] = ZoneSize
        ZoneGraph._G.vs["name"] = ZoneName
        ZoneGraph.Type = TypeVec[ZoneSingleIx]
        ZoneGraph.TX.linkage = self.TX.linkage
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
            nbrs = NearestNeighbors(n_neighbors=self.MaxEnvSize+1, algorithm='auto').fit(self.XY)
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
        iter = 10 # repeat sampling a few times, i.e. expected values over 
        rndKL = np.zeros(self.MaxEnvSize)

        for k in range(self.MaxEnvSize):
            TypeSample = np.reshape(np.random.choice(self.Type,size = (k+1)*iter) , (-1, iter))
            P = np.apply_along_axis(lambda V : CountValues(V,UnqTypes),axis=0,arr = TypeSample)
            
            # P=P.T
            P = self.TX.MultilevelFrequency(P.T)
            
            # Ptypes_mat = np.broadcast_to(Ptypes,P.shape)
            
            mat = rel_entr(P, Ptypes)

            rndKLiter = np.sum(mat, axis=1)/np.log(2)
            rndKL[k]=rndKLiter.mean()
    
        KLsampling = np.broadcast_to(rndKL, (self.N,len(rndKL)))
        if verbose: 
            print(f"Calculation tool {time.time()-start:.2f}")
        
        # for debuging purposes...
        # return (None,None,KLsampling)  
                
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
            
        KL2g = np.zeros((self.N,self.MaxEnvSize))
        
        # create the mapping betweeb leaf type and ALL tree types
        treemat = self.TX.TreeAsMat()
        treemap = np.zeros(treemat.shape[1],dtype='object')
        treemapsz = np.zeros(treemap.shape,dtype='object')
        treemapweight = np.zeros(treemap.shape,dtype='object')
        for i in range(treemat.shape[1]):
            treemap[i] = np.flatnonzero(treemat[:,i])
            treemapsz[i] = np.ones(len(treemap[i]))
            treemapweight[i] = treemat[treemap[i],i]
            
        # Convert TypeInt to TypeIntTree that accounts for ALL tree types
        RowTracker = np.zeros(TypeInt.shape,dtype='object')
        TypeIntTree = np.zeros(TypeInt.shape,dtype='object')
        WeightTracker = np.zeros(TypeInt.shape,dtype='object')

        for i in range(TypeIntTree.shape[0]):
            i_treemapsz = np.array([i*x for x in treemapsz],dtype='object')
            TypeIntTree[i,:] = treemap[TypeInt[i,:]]
            RowTracker[i,:] = i_treemapsz[TypeInt[i,:]]
            WeightTracker[i,:] = treemapweight[TypeInt[i,:]] 
            
        # count by increasing env size by one and each time add 1 to the Total "bean counter"
        # we are adding 1 to the leaf and all "upstream" composite types from the tree
        Totals = torch.zeros((self.N,treemat.shape[0]),dtype=torch.double)

        if verbose: 
            print(f"Calculating KL divergence for all cells per environment size of {TypeInt.shape[1]}")
        
        for k in range(TypeInt.shape[1]): 
            if k % 50 ==0: 
                print(f"iter: {k} time: {time.time()-start:.2f}")
            # bean counting

            ix_type = torch.from_numpy(np.hstack(TypeIntTree[:,k]).astype(dtype=np.int64))
            ix_rows = torch.from_numpy(np.hstack(RowTracker[:,k]).astype(dtype=np.int64)) 
            weights = torch.from_numpy(np.hstack(WeightTracker[:,k])) 
    
            Totals.index_put_((ix_rows,ix_type),weights, accumulate=True)
                             
            # probabilities (each time renormalize all rows to sum=1)
            row_sums = Totals.sum(axis = 1)
            P = Totals / row_sums[:,None]
            
            # element wise KL divergence (p*log(p/q))
            mat = rel_entr(P, Ptypes)
            # get entropy in bits
            KL2g[:,k]=np.sum(mat.numpy(), axis=1)/np.log(2)
        
        KLdiff = KL2g - KLsampling
        
        self.EnvSize = np.zeros(self.N,dbtype='int64')
        for i in range(self.N):
            self.EnvSize[i]=np.argmax(KLdiff[i,self.MinEnvSize:])+self.MinEnvSize
        
        return (KLdiff,KL2g,KLsampling)               
        

    def extractEnvironments(self):
        unqlbl = np.unique(self.Type)
        (distances,indices) = self.SpatialNeighbors
        Env = np.zeros((self.N,len(unqlbl)))
        for i in range(self.N):
            Env[i,:]=CountValues(self.Type[indices[i,0:self.EnvSize[i]]],unqlbl)
    
        # get treemat
        treemat = self.TX.TreeAsMat() 
        Env = np.matmul(Env,treemat.T)
    
        return(Env)
        
        
        
