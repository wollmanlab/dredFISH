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

from numpy.random import default_rng
rng = default_rng()

from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon

from scipy.stats import multinomial
from scipy.stats import binom

from torch.distributions.multinomial import Multinomial
import torch

from IPython import embed
from os.path import exists


def buildgraph(X,n_neighbors = 15,metric = 'correlation',accuracy = 4):
     
    # update number of neighbors (x accuracy) to increase accuracy
    # at the same time checks if we have enough rows 
    n_neighbors = min(X.shape[0]-1,n_neighbors)
    
    if metric == 'jsd':
        knn = NearestNeighbors(n_neighbors = n_neighbors,metric = jensenshannon)
        knn.fit(X)
        (distances,indices) = knn.kneighbors(X, n_neighbors = n_neighbors)
    else:
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
        self.nrmtreemat = None
        self.verbose = True
        return None
    
    @staticmethod
    def TreenomialMixtureModelWithCondEntropy(X,EG,Cvec):
        # deal with case where Cvec is scalar
        if len(np.array(Cvec).shape)==0:
            iter=1
            Cvec = np.array([Cvec,0])
        else:
            iter=len(Cvec)
            
        Ent = np.zeros(iter)
        Cls = np.zeros((X.shape[0],iter))
        
        for i in range(iter):
            c = Cvec[i]
            k = EG.Ntypes
            n = EG.EnvSize
            (treemat,nrmtreemat) = EG.TX.TreeAsMat(return_nrm = True)
            TM = TreenomialMixture(c,k,n,treemat,nrmtreemat)
            Cls[:,i] = TM.fit(X)
            Ent[i] = EG.ContractGraph(Cls[:,i]).CondEntropy()
            print(f"Cvec: {Cvec[i]} Cond Entropy: {Ent[i]:.2f}")
        
        ix = np.argmax(Ent)
        cls = Cls[:,ix]
            
        return(cls)
    
    @staticmethod
    def RecursiveLeidenWithTissueGraphCondEntropy(X,TG,metric = 'correlation',single_level = False,initRes = None): 
        """
            Find optimial clusters (using recursive leiden)
            optimization is done on resolution parameter and allows for recursive descent to 
            further break clusters into subclusters as long as graph conditional entropy is increased. 
            
        """
        
        verbose = True
        
        # used for output if verbose = true
        start = time.time()
        
        def OptLeiden(res,agraph,ix,currcls):
            """
            Basic optimization routine for Leiden resolution parameter. 
            Implemented using igraph leiden community detection
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
        if verbose: 
            print(f"Build similarity graph ")
        
        if initRes is not None and exists('G_fordev.pkl'): 
            fullgraph = ig.Graph.Read_Pickle('G_fordev.pkl')
        else: 
            fullgraph = buildgraph(X,metric = metric)
            
        if verbose:
            print(f"calculation took: {time.time()-start:.2f}")
        
        if verbose: 
            print(f"Calling initial optimization")
            
        emptycls = np.asarray(['' for _ in range(TG.N)],dtype='object')
        if initRes is None:
            sol = minimize_scalar(OptLeiden, args = (fullgraph,np.arange(TG.N),emptycls),
                                             bounds = (0.1,30), 
                                             method='bounded',
                                             options={'xatol': 1e-2, 'disp': 3})
            initRes = sol['x']
            ent_best = sol['fun']
            if verbose: 
                print(f"Initial entropy was: {-ent_best} number of evals: {sol['nfev']}")

        if verbose:
            print(f"calculation took: {time.time()-start:.2f}")
        
        cls = fullgraph.community_leiden(resolution_parameter=initRes,objective_function='modularity').membership
        cls = np.asarray(cls).astype(str)

        if verbose:
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
        return cls
    
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
        
        # nornmalize treemat by row so that each node has the same influence, no matter if it's a leaf or internal
        # row_sums = treemat.sum(axis = 1)
        # treemat = treemat / row_sums[: , np.newaxis]
        
        # normalize by spreading each leaf weight to all upstream internal nodes 
        # row_sums = treemat.sum(axis = 0)
        # treemat = treemat / row_sums[np.newaxis,:]
        
        self.treeasmat = treemat
        self.nrmtreemat = nrmtreemat
        
        if return_nrm:
            return (self.treeasmat,self.nrmtreemat)
        else: 
            return self.treeasmat 
    
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
    



class TreenomialMixture: 
    
    def __init__(self,c,k,n,treemat,nrmtreemat): 
        """
        Multinomial mixture model
        c: the number of hidden states
        k: number of leaf (final categories)
        n: number of trials in each "draw", i.e. environment size (assuming the same for all)
        treemat : representation of binary tree as sparse binary matrix mapping leaves to nodes
        nrmtreemat : similar to treemat, but providing normalization, i.e. binary matrix of parents for each node
        
        """
        
        self.C = c
        self.K = k
        self.N = n
        self.smoothing = 0.001  # Laplace smoothing (?)
        # for simplicity, we keep track of all 2*(n-1) parameters seperatly 
        # in practive, a lot of them are dependent (i.e. q, (1-q) for each param
        self.Q = np.zeros((self.C, 2*self.K-2))
        self.freq = np.ones(self.C)/self.C
        self.treemat = treemat
        self.nrmtreemat = nrmtreemat
        
    @property
    def treemap(self):
        treemap = np.zeros(self.treemat.shape[1],dtype='object')
        for i in range(self.treemat.shape[1]):
            treemap[i] = np.flatnonzero(self.treemat[:,i]).astype(np.int64)
        
        return(treemap)
    
    def inferTreeParams(self,X):
        """
        infers the (dependent) 2n-2 params for a single tree. 
        X is a m x k count matrix 
        """
        if len(X.shape)==1:
            X=X[None,:]
        
        # sum all count data if we get more them one environment 
        X = X.sum(axis=0)
        
        q_no = np.matmul(X,self.treemat.T)
        q_deno = np.matmul(X,self.nrmtreemat.T)
        q = q_no/q_deno
        q[np.isnan(q)]=0
        return(q)
    
    def leafProb(self):
        """
        transforms the probabilities to leaf based, i.e. multiples all the tree conditional probabilities
        """
        P = np.zeros((self.C,self.K))
        for i in range(self.C): 
            P[i,:] = [np.prod(self.Q[i,self.treemap[j]]) for j in range(len(self.treemap))]
        
        return(P)
    
    def loglik(self,X):
        """
        calculates the log likelihood matrix of all mixtures for the data
        returns a m x C matrix of log likelihoods. 
        
        Internaly, transforms to multinomial with leaf probabilities and calculates likelihood using scipy multinomial
        """
        # Xsplt = np.split(X,X.shape[0],axis=0)
        loglik_mat = torch.zeros((X.shape[0],self.C))
        P = self.leafProb();
        P = torch.from_numpy(P)
        X = torch.from_numpy(X)
        tfreq = torch.from_numpy(self.freq)
        for i in range(self.C):
            RV = Multinomial(self.N,P[i,:])
            loglik_mat[:,i] = RV.log_prob(X) + torch.log(tfreq[i])
            # loglik_mat[:,i] = RV.logpmf(Xsplt).flatten() + np.log(self.freq[i])
        loglik_mat = loglik_mat.numpy()
        return(loglik_mat)
    
    def fit(self, X,cls = None, threshold=0, max_epochs=10):
        """
        Training with Expectation Maximisation (EM) Algorithm
        :param X: the target labels as frequencies (NxK)
        :param threshold: stopping criterion based on the % decrease variation off the likelihood
        :param max_epochs: maximum number of epochs
        """  
        
        # initlize cls randomly if not provided
        if cls is None: 
            cls = rng.integers(low=0, high=self.K, size=X.shape[0])
        
        # EM Algorithm
        likelihood_list = list()
        current_epoch = 0
        old_likelihood = - np.inf
        delta = np.inf
        
        start = time.time()
        print("Treenomial mixture model training:")
        while current_epoch <= max_epochs and delta > threshold:
            
            # M-step - estimate parameters based on current cls
            for i in range(self.C):
                self.Q[i,:] = self.inferTreeParams(X[cls==i,:])
                self.freq[i] = sum(cls==i)
            
            # E-step
            loglik_mat = self.loglik(X)
                    
            loglik = loglik_mat.max(axis=1)
            cls = loglik_mat.argmax(axis=1)
            
            likelihood_list.append(sum(loglik))
            # delta = abs((sum(loglik) - old_likelihood)/old_likelihood)
            old_likelihood = sum(loglik)

            print(f"epoch: {current_epoch} likelihood: {sum(loglik):.2f} time: {time.time()-start:.2f}")
            current_epoch += 1
        
        return cls
    
    def cluster(self,X):
        loglik_mat = self.loglik(X)
        cls = loglik_mat.argmax(axis=1)
        return(cls)