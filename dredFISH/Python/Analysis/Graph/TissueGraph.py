"""TissueGraph functions

A collection of files to analyze tissues using Graph representation

This file should be imported as a module and contains the following
functions:

    * BuildSpatialGraph - construct graph based on Delauny neighbors
    * AddTypeInfo - adds a type attribute to all vertices (cells / zones / environments)
    
"""

# dependeices
from igraph import *
from scipy.spatial import Delaunay,Voronoi
import numpy as np
import warnings

class TissueGraph: 
    def __init__(self): 
        self._G=None
        self.N = None
        return None
        
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
            
        # update number of items
        self.N = XY.shape[0]
        
        # start with triangulation
        dd=Delaunay(XY)

        # create adjacency matrix
        AdjMat=np.zeros([self.N,self.N]).astype(np.bool_)
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
        XY=np.zeros(self.N,2)
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
        if "Type" not in G.vs.attribute_names():
            warnings.warn("Type was not initalized - returning None")
            return(None)
        
        # if we are here, then there is a Type attribute. Just return it, 
        return(self._G.vs["Type"])
        
    
    @Type.setter
    def Type(self, TypeVec):
        
        self._G.vs["Type"]=TypeVec
        return
  
    
    def ContractGraph(self,TypeVec):
        """ContractGraph : reduce graph size by merging neighbors of same type. 
            Given a vector of types, will contract the graph to merge vertices that are 
            both next to each other and of the same type. 
        
        Input: a vector of Types for each node. If TypeVec is not provided will attempty to use the Type property of the graph itself. 
        
        Output: a new TissueGraph after vertices merging. 
        """
         # get edge list    
        EL = np.asarray(G.get_edgelist()).astype("int")
        nm=G.vs["name"]
        EL[:,0] = np.take(nm,EL[:,0])
        EL[:,1] = np.take(nm,EL[:,1])

        SameTypeIX = self.Type[SpatialNeighberhood[,1]] == self.Type[SpatialNeighberhood[,2]]
        
        
    
    def CondEntropy(self):
        """
        CondEntropy: calculate conditional entropy of the tissue graph
                     cond entropy is the difference between graph entropy based on pagerank and type entropy
        """
        Pzones = self._G.pagerank()
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        Ptypes=np.array(list(Counter(self.Type).values()))
        Ptypes=Ptypes/np.sum(Ptypes)
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        CondEntropy = Entropy_Zone-Entropy_Types
        return(CondEntropy)
            
           