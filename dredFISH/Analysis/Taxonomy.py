"""Taxonomy

Class to represent a taxonomy (binary tree) of types. 

Taxonomies are created using two inputs: 
    1. data matrix (either PNMF or Env)
    2. TissueGraph object

Creating a taxonomy involves two parts: 
    1. finding high-res clusters - this assigns a label for each row in input data. 
    2. building a binary tree of relationship between clusters. Tree starts with cluster average of the high-res clusters
    
"""

import numpy as np
import torch
import pandas as pd

import time 
from IPython import embed

from numpy.random import default_rng
rng = default_rng()

from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, fcluster, cut_tree
import pynndescent 

from igraph import *

class Taxonomy: 
    
    def __init__(self):
        self.data = None
        self.leaflabels = None
        self.linkage = None
        self.treeasmat = None
        self.nrmtreemat = None
        self.verbose = True
        return None
 
    def convert_multinomial_to_treenomial(self,Env):
         # create the treenomial distribution for all envs
        (treemat,nrmtreemat) = self.TreeAsMat(return_nrm = True)
        q_no = np.matmul(Env,treemat.T)
        q_deno = np.matmul(Env,nrmtreemat.T)
        q = q_no/q_deno
        q[np.isnan(q)]=0
        return(q)
    
    
    def BuildTree(self,method='average'): 
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
    
    def TreeAsMat(self,return_nrm = False):
        """
           Converts scipy linkage (self.linkage) into a nodes x leafs matrix. 
           for N types, the matrix size will be (2*N-2,N) 
           where each row has 1 if a leaf is under this ode and 0 otherwide
        """
        if self.treeasmat is not None:
            if return_nrm:
                return (self.treeasmat,self.nrmtreemat)
            else: 
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
        
        nrmtreemat = np.zeros(treemat.shape)
        Zlr = np.array(self.linkage[:,0:2]).astype(np.int64)
        for i in range(self.linkage.shape[0]):
            nrmtreemat[Zlr[i,0],:]=treemat[Zlr[i,0],:]+treemat[Zlr[i,1],:]
            nrmtreemat[Zlr[i,1],:]=treemat[Zlr[i,0],:]+treemat[Zlr[i,1],:]
        
        self.treeasmat = treemat
        self.nrmtreemat = nrmtreemat
        
        if return_nrm:
            return (self.treeasmat,self.nrmtreemat)
        else: 
            return self.treeasmat 
    

    



