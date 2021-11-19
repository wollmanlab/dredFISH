"""TissueGraph functions

A collection of files to analyze tissues using Graph representation

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
import warnings

class TissueGraph: 
    def __init__(self): 
        self._G = None
        self.UpstreamMap = None
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

        # create adjacency matrix
        AdjMat=np.zeros([XY.shape[0],XY.shape[0]]).astype(np.bool_)
        for x in dd.simplices:
            AdjMat[x[0],x[1]]=True
            AdjMat[x[1],x[0]]=True
            AdjMat[x[0],x[2]]=True
            AdjMat[x[2],x[0]]=True
            AdjMat[x[1],x[2]]=True
            AdjMat[x[2],x[1]]=True
    
        self._G = Graph.Adjacency(AdjMat,mode="undirected").simplify()
        self._G.vs["X"]=XY[:,0]
        self._G.vs["Y"]=XY[:,1]
        
        # set up names
        self._G.vs["name"]=list(range(self.N))
        return(self)

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
  
    
    def ContractGraph(self,TypeVec=None):
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
        # decompose is a method from igraph that returns a list of all individual components (i.e. zones) 
        DecomposedZones = Graph.decompose(IsoZonesGraph) 
        
        IxMapping = np.empty(self.N)
        ZoneTypes = []
        ZoneXY=np.empty((len(DecomposedZones),2))
        
        for i in range(len(DecomposedZones)):
            ix=DecomposedZones[i].vs.get_attribute_values("name")
            IxMapping[ix]=i
            ZoneTypes.append(self.Type[ix[0]])
            ZoneXY[i,:]=np.mean(self.XY[ix,:],axis=0)

        
        ZoneGraph = TissueGraph()
        ZoneGraph.BuildSpatialGraph(ZoneXY)
        ZoneGraph.Type = ZoneTypes
        ZoneGraph._G.vs["Size"] = [x.vcount() for x in DecomposedZones]
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
        Ptypes=np.array(list(Counter(self.Type).values()))
        Ptypes=Ptypes/np.sum(Ptypes)
        return(Ptypes)
                             
    def CondEntropy(self):
        """
        CondEntropy: calculate conditional entropy of the tissue graph
                     cond entropy is the difference between graph entropy based on pagerank and type entropy
        """
        Pzones = self._G.pagerank()
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        # validate that type exists
        if self.Type == None: 
            raise ValueError("Cann't calculate cond-entropy without Types, please check")
            
        Ptypes = self.TypeFreq(); 
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        CondEntropy = Entropy_Zone-Entropy_Types
        return(CondEntropy)
    
    def FindLocalMicroenvironments(self,MinEnvSize): 
        """
        FindLocalMicroenvironments identifies most informative local microenvironment length-scale (um) for each vertex in the graph
                                   Calculations are based on KL between env and all minus KL of permuted 
        """
        
        
        
    
        
            
           