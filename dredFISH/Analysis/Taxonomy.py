"""Taxonomy

Class to represent a taxonomy (binary tree) of types. 

Taxonomies are created using two inputs: 
    1. data matrix (either PNMF or composition)
    2. TissueGraph object

Creating a taxonomy involves two parts: 
    1. finding high-res clusters - this assigns a label for each row in input data. 
    2. building a binary tree of relationship between clusters. Tree starts with cluster average of the high-res clusters
    
"""

from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import igraph as ig
import pynndescent
import time 
import numpy as np
import torch
import pandas as pd
from scipy.optimize import minimize_scalar

from IPython import embed


def buildgraph(X,n_neighbors = 15,metric = 'correlation',accuracy = 4):
    
    # update number of neighbors (x accuracy) to increase accuracy
    # at the same time checks if we have enough rows 
    n_neighbors = min(X.shape[0]-1,n_neighbors)
    n_neighbors_with_extras = min(X.shape[0]-1,n_neighbors * accuracy)
    
    # perform nn search (using accuracy x number of neighbors to improve accuracy)
    knn = pynndescent.NNDescent(X,n_neighbors = n_neighbors_with_extras ,metric=metric,diversify_prob=0.5)
    
    # get indices and remove self. 
    (indices,distances) = knn.neighbor_graph
    indices=indices[:,1:n_neighbors+1]
    
    id_from = np.tile(np.arange(indices.shape[0]),indices.shape[1])
    id_to = indices.flatten(order='F')
    
    # build graph
    edgeList = np.vstack((id_from,id_to)).T
    G = ig.Graph(n=X.shape[0], edges=edgeList)
    return G

class Taxonomy: 
    
    def __init__(self):
        self.data = None
        self.leaflabels = None
        self.linkage = None
        self.treeasmat = None
        self.verbose = True
        return None
    
    
    def RecursiveLeidenWithTissueGraphCondEntropy(self,X,TG,metric = 'correlation',single_level = False): 
        """
            Find optimial clusters (using recursive leiden)
            optimization is done on resolution parameter and allows for recursive descent to 
            further break clusters into subclusters as long as graph conditional entropy is increased. 
            
        """
        # update self.data
        self.data = X
        
        # used for output if verbose = true
        start = time.time()
        
        def OptLeiden(res,agraph,ix,currcls):
            """
            Basic optimization routine for Leiden resolution parameter. 
            Implemented using igraph leiden 
            """
            # calculate leiden clustering
            TypeVec = agraph.community_leiden(resolution_parameter=res,objective_function='modularity').membership
            TypeVec = np.asarray(TypeVec).astype(str)
            # merge TypeVec with full cls vector
            dash = np.array((1,),dtype='object')
            dash[0]='_' 
            newcls = currcls.copy()
            newcls[ix] = newcls[ix]+dash+TypeVec
            
            CG = TG.ContractGraph(newcls)
            Entropy = CG.CondEntropy()
            return(-Entropy)
        
        # we start by performing first clustering
        if self.verbose: 
            print(f"Build similarity graph ")
        fullgraph = buildgraph(X,metric = metric)
        if self.verbose:
            print(f"calculation took: {time.time()-start:.2f}")
        
        if self.verbose: 
            print(f"Calling initial optimization")
            
        emptycls = np.asarray(['' for _ in range(TG.N)],dtype='object')
        sol = minimize_scalar(OptLeiden, args = (fullgraph,np.arange(TG.N),emptycls),
                                         bounds = (0.1,30), 
                                         method='bounded',
                                         options={'xatol': 1e-2, 'disp': 3})
        
        if self.verbose:
            print(f"calculation took: {time.time()-start:.2f}")
        
        initRes = sol['x']
        ent_best = sol['fun']
        if self.verbose: 
            print(f"Initial entropy was: {-ent_best} number of evals: {sol['nfev']}")
        
        cls = fullgraph.community_leiden(resolution_parameter=initRes,objective_function='modularity').membership
        cls = np.asarray(cls).astype(str)

        if self.verbose:
            u=np.unique(cls)
            print(f"Initial types found: {len(u)}")
            
        if single_level: 
            return cls
        
        def DescentTree(cls,ix,ent_best):
            
            dash = np.array((1,),dtype='object')
            dash[0]='_' 
            unqcls = np.unique(cls[ix])
            
            if self.verbose: 
                print(f"descending the tree")
            for i in range(len(unqcls)):
                newcls = cls.copy()
                ix = np.flatnonzero(np.asarray(cls == unqcls[i]))
                
                subgraph = buildgraph(X[ix,:])
                sol = minimize_scalar(OptLeiden, args = (subgraph,ix,newcls),
                                                 bounds = (0.1,30), 
                                                 method='bounded',
                                                 options={'xatol': 1e-2, 'disp': 2})
                ent = sol['fun']
                if self.verbose: 
                    print(f"split groun {i} into optimal parts, entropy {-ent}")
                if ent < ent_best:
                    if self.verbose: 
                        print(f"split improves entropy - descending")
                    ent_best=ent
                    
                    nxtlvl = fullgraph.community_leiden(resolution_parameter=initRes,objective_function='modularity').membership
                    nxtlvl = np.asarray(nxtlvl).astype(str)
                    newcls[cls==unqcls[i]] = cls[cls==unqcls[i]]+dash+nxtlvl
                    cls = newcls.copy()
                    cls = DescentTree(cls,ix,ent_best)
                    
            return(cls)
        
        cls = DescentTree(cls,np.arange(TG.N),ent_best) 
        return None
    
    def BuildTree(self,method='ward'): 
        """ perform hierarchial clustering (wraps around scipy.cluster.hierarchy) 
        """
        (X,unqTypes) = self.avgPerType()
        self.linkage = linkage(X,method)
        return None
    
    def avgPerType(self): 
        """ aggregates full data matrix based on high-res clusters
        """
        df = pd.DataFrame(data = self.data)
        df['type']=self.leaflabels
        avg = df.groupby(['type']).mean()
        unqTypes = avg.index
        avg = np.array(avg)
        return (avg,unqTypes)
    
    def TreeAsMat(self):
        """
           Converts scipy linkage (self.linkage) into a nodes x leafs matrix. 
           for N types, the matrix size will be (2*N-2,N) 
           where each row has 1 if a leaf is under this ode and 0 otherwide
        """
        if self.treeasmat is not None:
            return self.treeasmat
        
        # define recursive function 
        def get_children(node):
            if node.is_leaf():
                return np.array(node.get_id())
            return np.append(get_children(node.get_left()), get_children(node.get_right()))
        
        # init mat and get children of all nodes
        rootnode, nodelist = to_tree(self.linkage, rd=True)
        
        treemat = np.zeros((len(nodelist)-1,len(np.unique(self.leaflabels))))
        for i in range(len(nodelist)-1):
            ix=get_children(nodelist[i])
            treemat[i,ix]=1
        
        # nornmalize treemat by row so that each node has the same influence, no matter if it's a leaf or internal
        # row_sums = treemat.sum(axis = 1)
        # treemat = treemat / row_sums[: , np.newaxis]
        
        # normalize by spreading each leaf weight to all upstream internal nodes 
        row_sums = treemat.sum(axis = 0)
        treemat = treemat / row_sums[np.newaxis,:]
        
        self.treeasmat = treemat
        return treemat 
    
    def MultilevelFrequency(self,P):
        """
            converts the probability vector (matrix) P with shape (:,Ntypes)
            to a multi-level probability vector that includes all nodes alont the tree
            
            for performance, try to keep things as torch
        """
        
        mat = self.TreeAsMat()
            
        P_ml = np.matmul(P,mat.T)
              
        row_sums = P_ml.sum(axis = 1)
        P_ml = P_ml / row_sums[: , np.newaxis]
        return(P_ml)
    