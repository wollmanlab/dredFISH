"""Classification of biospatial units in TMG analysis

This module contains the main classes related to the task of classification of biospatial units (i.e. cells, isozones, regions) into type. 
The module is composed of a class hierarchy of classifiers. The "root" of this heirarchy is an abstract base class "Classifier" that has two 
abstract methods (train and classify) that any subclass will have to implement. 

"""

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pynndescent import NNDescent
from tqdm import tqdm
import pandas as pd
from collections import Counter
import logging
import abc
from signal import valid_signals
import string
import numpy as np
import itertools
from datetime import datetime
from multiprocessing import Pool
from scipy.stats import norm
from scipy.stats import entropy, mode, median_abs_deviation
from scipy.spatial.distance import squareform
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pynndescent 
import anndata
import os
import torch
import math
import matplotlib.colors as mcolors
from dredFISH.Analysis import TissueGraph
from dredFISH.Utils import fileu
from dredFISH.Utils import tmgu
from dredFISH.Utils import pathu
from dredFISH.Utils import basicu
from dredFISH.Utils import celltypeu
from functools import partial
from scipy.ndimage import gaussian_filter
class Classifier(metaclass=abc.ABCMeta): 
    """Interface for classifiers 

    This is an abstract class that defines the interface (i.e. methods and attributes) that all different classifiers must have. 
        
    Attributes
    ----------
    tax : Taxonomy
        A taxonomy system that the classifier classifies into. 
    """
    
    def __init__(self, tax=None): 
        """Create a classifier
        
        Parameters
        ----------
        tax : Taxnonomy system
            The taxonomical system that is used for classification. 
        """
        if tax is None: 
            tax = TissueGraph.Taxonomy(name='clusters')
        self.tax = tax
    
    @abc.abstractmethod
    def train(self):
        """Trains a classifier
        """
        pass
    
    @abc.abstractmethod
    def classify(self,data):
        """Classifies input into types
        """
        pass

class Unsupervized(Classifier):
    """
    Unsupervized "pass-through", i.e. the ref data is the answer
    only check if number of cells matches
    """
    def __init__(self):
        super().__init__()

    def train(self,ref_data,ref_label_id): 
        # update the taxonomy
        self.tax.add_types(ref_label_id,ref_data)
        self._ref_label_id = ref_label_id
        self._ref_data = ref_data

    def classify(self, data = None):
        if data is None or data.shape[0]==self._ref_data.shape[0]:
            return self._ref_label_id
        else: 
            raise ValueError("Number of rows in data doesn't match training ref")
        
class KNNClassifier(Classifier): 
    """k-nearest-neighbor classifier
    
    Attributes
    ----------
    k : int (default = 1)
        The number of local neighbors to consider
    approx_nn : int (default = 30)
        Number of NN used by pynndescent to construct the approximate knn graph
    metric : str (default = 'correlation')
        What metric to use to determine similarity
        
    """
    def __init__(self,k=1,approx_nn= 30,metric = 'correlation', tax = None):
        """Define a KNN classifier
        
        Parameters
        ----------
        k : int
            the K in KNN
        approx_nn : int
            number of nearest neighbors used in pynndescent calculation of the knn graph
        metric : str
            what distance metric to use? 
        """
        super().__init__(tax = tax)
        self.k = k
        self.approx_nn = approx_nn
        self.metric = metric

    def train(self, ref_data, ref_label_id):
        """ Builds approx. KNN graph for queries
        """
        self._knn = pynndescent.NNDescent(ref_data.astype(np.float32),n_neighbors = self.approx_nn, metric = self.metric)
        self._knn.prepare()
        self._ref_label_id = np.array(ref_label_id)
        
        # update the taxonomy
        self.tax.add_types(ref_label_id,ref_data)

    def classify(self, data): 
        """Find class of rows in data based on approx. KNN
        """
        # get indices and remove self. 
        indices,distances = self._knn.query(data.astype(np.float32),k=self.k)
        if self.k==1:
            ids = self._ref_label_id[indices]
        else:
            kids = self._ref_label_id[indices]
            ids = mode(kids,axis=1)
        return ids.flatten()
    
class OptimalLeidenUnsupervized(Unsupervized):
    """Classifiy cells based on unsupervized Leiden with optimal resolution 
    
    This classifiers is trained using unsupervized learning. 
    Uses Unsupervized (passthrough) classifiers
    """ 
    def __init__(self, TG):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
        """
        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("OptimalLeidenUnsupervized requires a TissueGraph object as input")
            
        #set up the KNN portion
        super().__init__()
        self._TG = TG   
    
    def train(self, opt_res=None, opt_params={'iters' : 10, 'n_consensus' : 50}):
        """training for this class is performing optimal leiden clustering 
        
        Optimization is done over resolution parameter trying to maximize the TG conditional entropy. 
        The analysis uses the two graphs (spatial and features) and finds a resolution parameters that will maximize 
        the conditional entropy, H(Zone|Type). 
        
            
        Parameters
        ----------
        opt_params : dict
            Few parameters that control optimizaiton. 
            'iter' is the number of repeats in each optimization round. 
            'n_consensus' is the number of times to do the final clustering with optimal resolution.   
            
        """
        def ObjFunLeidenRes(res, return_types=False):
            """
            Basic optimization routine for Leiden resolution parameter. 
            Implemented using igraph leiden community detection
            """
            EntropyVec = np.zeros(opt_params['iters'])
            for i in range(opt_params['iters']):
                TypeVec = self._TG.FG.community_leiden(resolution_parameter=res,
                                                   objective_function='modularity').membership
                TypeVec = np.array(TypeVec).astype(np.int64)
                EntropyVec[i] = self._TG.contract_graph(TypeVec,return_useful_layer = False).cond_entropy()
            Entropy = EntropyVec.mean()
            if return_types: 
                return (-Entropy, TypeVec)
            else:
                return (-Entropy)

        if opt_res is None: 
            print(f"Calling initial optimization")
            sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
                                                method='bounded',
                                                options={'xatol': 1e-2, 'disp': 3})
            self._opt_res = sol['x']
            ent = sol['fun']
            evls = sol['nfev']
        else: 
            self._opt_res = opt_res
            ent = -1
            evls = 0
        
        # consensus clustering
        TypeVec = np.zeros((self._TG.N,opt_params['n_consensus']))
        for i in range(opt_params['n_consensus']):
            ent, TypeVec[:,i] = ObjFunLeidenRes(self._opt_res, return_types=True)
            
        if opt_params['n_consensus']>1:
            cmb = np.array(list(itertools.combinations(np.arange(opt_params['n_consensus']), r=2)))
            rand_scr = np.zeros(cmb.shape[0])
            for i in range(cmb.shape[0]):
                rand_scr[i] = adjusted_rand_score(TypeVec[:,cmb[i,0]],TypeVec[:,cmb[i,1]])
            rand_scr = squareform(rand_scr)
            total_rand_scr = rand_scr.sum(axis=0)
            TypeVec = TypeVec[:,np.argmax(total_rand_scr)]
                                                  
        print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {ent} number of evals: {evls}")

        # update the Taxonomy and create
        # self.tax.add_types(self._TG.feature_mat,TypeVec)
        # self.TypeVec = TypeVec
        
        # train the KNN classifier using the types we found
        super().train(self._TG.feature_mat, TypeVec)

class OptimalLeidenKNNClassifier(KNNClassifier):
    """Classifiy cells based on unsupervized Leiden with optimal resolution 
    
    This classifiers is trained using unsupervized learning with later classification done by knn (k=1)
    the implementations uses the same classify method as KNNClassifier.  
    The overladed train method does the unsupervized learning before calling super().train to create the KNN index.
    """
    def __init__(self, TG):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
        """
        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("OptimalLeidenKNNClassifier requires a TissueGraph object as input")
            
        #set up the KNN portion
        super().__init__(k=1,approx_nn = 10,metric = 'correlation')
        self._TG = TG
    
    def train(self, opt_res=None, opt_params={'iters' : 10, 'n_consensus' : 50}):
        """training for this class is performing optimal leiden clustering 
        
        Optimization is done over resolution parameter trying to maximize the TG conditional entropy. 
        The analysis uses the two graphs (spatial and features) and finds a resolution parameters that will maximize 
        the conditional entropy, H(Zone|Type). 
        
            
        Parameters
        ----------
        opt_params : dict
            Few parameters that control optimizaiton. 
            'iter' is the number of repeats in each optimization round. 
            'n_consensus' is the number of times to do the final clustering with optimal resolution.   
            
        """
        def ObjFunLeidenRes(res, return_types=False):
            """
            Basic optimization routine for Leiden resolution parameter. 
            Implemented using igraph leiden community detection
            """
            EntropyVec = np.zeros(opt_params['iters'])
            for i in range(opt_params['iters']):
                TypeVec = self._TG.FG.community_leiden(resolution_parameter=res,
                                                   objective_function='modularity').membership
                TypeVec = np.array(TypeVec).astype(np.int64)
                EntropyVec[i] = self._TG.contract_graph(TypeVec,return_useful_layer = False).cond_entropy()
            Entropy = EntropyVec.mean()
            if return_types: 
                return (-Entropy, TypeVec)
            else:
                return (-Entropy)

        if opt_res is None: 
            print(f"Calling initial optimization")
            sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
                                                method='bounded',
                                                options={'xatol': 1e-2, 'disp': 3})
            self._opt_res = sol['x']
            ent = sol['fun']
            evls = sol['nfev']
        else: 
            self._opt_res = opt_res
            ent = -1
            evls = 0
        
        # consensus clustering
        TypeVec = np.zeros((self._TG.N,opt_params['n_consensus']))
        for i in range(opt_params['n_consensus']):
            ent, TypeVec[:,i] = ObjFunLeidenRes(self._opt_res, return_types=True)
            
        if opt_params['n_consensus']>1:
            cmb = np.array(list(itertools.combinations(np.arange(opt_params['n_consensus']), r=2)))
            rand_scr = np.zeros(cmb.shape[0])
            for i in range(cmb.shape[0]):
                rand_scr[i] = adjusted_rand_score(TypeVec[:,cmb[i,0]],TypeVec[:,cmb[i,1]])
            rand_scr = squareform(rand_scr)
            total_rand_scr = rand_scr.sum(axis=0)
            TypeVec = TypeVec[:,np.argmax(total_rand_scr)]
                                                  
        print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {ent} number of evals: {evls}")

        # update the Taxonomy and create
        # self.tax.add_types(self._TG.feature_mat,TypeVec)
        # self.TypeVec = TypeVec
        
        # train the KNN classifier using the types we found
        super().train(self._TG.feature_mat, TypeVec)
         

class TopicClassifier(Classifier): 
    """Uses Latent Dirichlet Allocation to classify cells into regions types. 

    Decision on number of topics comes from maximizing overall entropy. 
    """
    def __init__(self,TG, ordr=4):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
        """
        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("TopicClassifier requires a (cell level) TissueGraph object as input")

        super().__init__(tax = None)
        self.tax.name = 'topics'
        self._TG = TG
        Env = self._TG.extract_environments(ordr=ordr)
        row_sums = Env.sum(axis=1)
        row_sums = row_sums[:,None]
        Env = Env/row_sums
        self.Env = Env

    def lda_fit(self, n_topics):
        """This cannot be nested inside
        """
        Env = self.Env

        lda = LatentDirichletAllocation(n_components=n_topics)
        B = lda.fit(Env)
        T = lda.transform(Env)
        return (B,T)

    def train(self, 
        n_topics_list=np.geomspace(2,100,num=10).astype(int), 
        n_procs=1):
        """fit multiple LDA models and chose the one with highest type entropy
        """
        # define function handle
        if n_procs == 1: 
            self.update_user("Running LDA in serial") 
            results = [self.lda_fit(n_topics) for n_topics in n_topics_list]
        else:
            n_procs = max(n_procs, len(n_topics_list))
            self.update_user(f"Running LDA in parallel with {n_procs} cores")
            with Pool(n_procs) as pl:
                results = pl.map(self.lda_fit, n_topics_list)

        IDs = np.zeros((self._TG.N,len(results)))
        for i in range(len(results)):
            IDs[:,i] = np.argmax(results[i][1],axis=1)
        Type_entropy = np.zeros(IDs.shape[1])
        for i in range(IDs.shape[1]):
            _,cnt = np.unique(IDs[:,i],return_counts=True)
            cnt=cnt/cnt.sum()
            Type_entropy[i] = entropy(cnt,base=2) 

        # some of the topics might be empty (weren't used in the fit)
        # so need to find the correct number and refit
        curr_lda = results[np.argmax(Type_entropy)][0]
        topics_prob = curr_lda.transform(self.Env)
        topics = np.argmax(topics_prob, axis=1)
        # train one last LDA after we found the correct number of topics
        n_topics = topics.max()
        self._lda = self.lda_fit(n_topics)[0]
        topics_prob = self._lda.transform(self.Env)
        topics = np.argmax(topics_prob, axis=1)
        self.tax.add_types(topics,self.Env)
        return self

    def classify(self,data):
        """classify based on fitted LDA"""
        topics_prob = self._lda.transform(data)
        topics = np.argmax(topics_prob, axis=1)

        return topics
    
class KnownCellTypeClassifier(Classifier): 
    """
    """
    def __init__(self, TG, 
        tax_name='known_celltypes',
        ref='allen_smrt_dpnmf', 
        ref_levels=['class_label', 'neighborhood_label', 'subclass_label', 'cluster_label'], 
        model='knn',
        ):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
            The TG is required to have a `TG.adata.layers['raw']` matrix
        ref : 
            tag of a reference dataset, or path to a reference dataset (.h5ad)
        """
        # specific packages to this task

        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("This Classifier requires a (cell level) TissueGraph object as input")

        super().__init__(tax=None)
        self.tax.name = tax_name
        self._TG = TG # data
        self.ref_path = pathu.get_path(ref, check=True) # refdata
        self.ref_levels = ref_levels 

        # model could be a string or simply sklearn classifier (many kinds)
        if isinstance(model, str):
            if model == 'knn':
                self.model = KNeighborsClassifier(n_neighbors=15, metric='correlation') # metric 'cosine' is also good
            elif model == 'svm':
                self.model = svm.SVC(kernel='rbf')
            else:
                raise ValueError("Need to choose from: `knn`, `svm`")
        else:
            self.model = model

        #### TODO: subselect data by brain regions

        #### preproc data
        self.update_user("Loading and preprocessing data")
        # get ref data features
        ref_adata = anndata.read(self.ref_path)
        self.ref_ftrs = np.array(ref_adata.X)
        self.ref_lbls = ref_adata.obs[self.ref_levels].values # multiple levels

        # pre normalize to get features
        self.ftrs = basicu.normalize_fishdata(self._TG.adata.layers['raw'], norm_cell=True, norm_basis=False) 


    def train(self, 
        run_func='run', # baseline
        run_kwargs_perlevel=[dict(norm='per_bit_equalscale', ranknorm=False, n_cells=100)], 
        verbose=True,
        ignore_internal_failure=False,
        ):
        """
        """
        lbls = celltypeu.iterative_classify(
                        self.ref_ftrs,
                        self.ref_lbls,
                        self.ftrs,
                        levels=self.ref_levels, # levels of cell types
                        run_func=run_func, # baseline
                        run_kwargs_perlevel=run_kwargs_perlevel, 
                        model=self.model,
                        verbose=verbose,
                        ignore_internal_failure=ignore_internal_failure,
        )
        self.TypeVec = lbls
        
    def classify(self):
        return self.TypeVec
    

class SpatialPriorAssistedClassifier(Classifier): 
    """
    """
    def __init__(self, TG, 
        tax_name='spatial_prior_assited',
        ref='allen_wmb_tree', 
        spatial_ref='spatial_reference',
        ref_levels=['class', 'subclass','supertype','cluster'], 
        model='nb',
        ):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
            The TG is required to have a `TG.adata.layers['raw']` matrix
        ref : 
            tag of a reference dataset, or path to a reference dataset (.h5ad)
        spatial_ref : 
            tag of a spatial reference dataset, or path to a spatial reference dataset (.h5ad)
        """
        # specific packages to this task

        # if not isinstance(TG, TissueGraph.TissueGraph): 
        #     raise ValueError("This Classifier requires a (cell level) TissueGraph object as input")

        super().__init__(tax=None)
        self.tax.name = tax_name
        self._TG = TG # data
        self.ref_path = pathu.get_path(ref, check=True) # refdata
        self.spatial_ref_path = pathu.get_path(spatial_ref, check=True) # refdata
        self.ref_levels = ref_levels 

        # model could be a string or simply sklearn classifier (many kinds)
        if isinstance(model, str):
            if model == 'nb':
                self.model = GaussianNB()
            elif model == 'mlp':
                self.model = MLPClassifier(
                            hidden_layer_sizes=(50,),  # Adjust number and size of hidden layers as needed
                            activation='relu',  # Choose a suitable activation function
                            max_iter=500,  # Set maximum iterations for training
                            verbose=False,
                            )
            else:
                raise ValueError("Need to choose from: `nb`, `mlp`")
        else:
            self.model = model

        self.measured = TG.copy()#self._TG.adata.copy()

    def initalize(self):
        #### preproc data
        self.update_user("Loading and preprocessing data")

        base_level = self.ref_levels[-1]

        # get ref data features
        self.reference = anndata.read(self.ref_path)
        self.reference.layers['raw'] = self.reference.X.copy()
        self.spatial_reference = anndata.read(self.spatial_ref_path)

        shared_var = list(self.reference.var.index.intersection(self.measured.var.index))
        self.reference = self.reference[:,np.isin(self.reference.var.index,shared_var)].copy()
        self.measured = self.measured[:,np.isin(self.measured.var.index,shared_var)].copy()

        self.Nbases = self.measured.shape[1]

        """ Build Spatial Proportions Balanced"""
        self.update_user("Building Spatial Proportions Balanced")
        cts = []
        ccs = []
        for ct,cc in Counter(self.spatial_reference.obs[base_level]).items():
            cts.append(ct)
            ccs.append(cc)
        proportions = pd.DataFrame(ccs,index=cts,columns=['counts'])
        proportions['percentage'] = proportions['counts']/proportions['counts'].sum()
        proportions = proportions.sort_values(by='percentage', ascending=False)

        idxes = []
        total_cells = 1000000
        for idx,row in proportions.iterrows():
            n_cells = int(row['percentage']*total_cells)
            if n_cells>1:
                temp = np.array(self.reference.obs[self.reference.obs[base_level]==idx].index)
                if temp.shape[0]>0:
                    idxes.extend(list(np.random.choice(temp,n_cells)))
        self.spatial_balanced_reference = self.reference[idxes,:].copy()

        """ Perform Initial harmonization """
        self.normalize()

        """ Build Class Balanced """
        self.update_user("Building Class Balanced")
        idxes = []
        total_cells = 1000000
        cts = np.unique(self.reference.obs[base_level])
        for idx in cts:
            n_cells = int(total_cells/cts.shape[0])
            if n_cells>1:
                temp = np.where(self.reference.obs[base_level]==idx)[0]
                if temp.shape[0]>0:
                    idxes.extend(list(np.random.choice(temp,n_cells)))
        self.class_balanced_reference = self.reference[idxes,:].copy()


        """ Calculate Spatial Priors """
        self.update_user("Building Spatial Tree")
        self.max_distance = 0.2 # PARAMETER
        reference_coordinates_right = np.array(self.spatial_reference.obs[['ccf_z','ccf_y','ccf_x']])
        reference_coordinates_left = reference_coordinates_right*np.array([-1,1,1])
        reference_coordinates = np.concatenate([reference_coordinates_right,reference_coordinates_left])
        f = '/scratchdata1/AllenWMB_SpatialTree.pkl'
        import pickle
        try:
            with open(f,'rb') as f:
                self.tree = pickle.load(f)
        except:
            self.tree = NNDescent(reference_coordinates, metric='euclidean', n_neighbors=100, n_trees=1) #FIX EPSILON
            with open(f,'wb') as f:
                pickle.dump(self.tree,f)
        self.tree.prepare()

        self.update_user("Querying Spatial Tree")
        f = f"/scratchdata1/spatial_priors_{self.measured.obs['dataset'].iloc[0].split('_')[0]}_baselevel_{base_level}_window_{self.max_distance}.csv"
        print(f)
        try:
            priors = pd.read_csv(f,index_col=0)
        except:
            cell_labels = np.array(self.spatial_reference.obs[base_level])
            cell_labels = np.concatenate([cell_labels,cell_labels])
            cts = np.unique(cell_labels)
            priors = pd.DataFrame(np.zeros([self.measured.shape[0],cts.shape[0]]),index=self.measured.obs.index,columns=cts)
            n = 1000
            idxes = self.measured.obs.index
            for start_idx in tqdm(np.arange(0,idxes.shape[0],n)):
                if start_idx+n>idxes.shape[0]:
                    end_idx = idxes.shape[0]
                else:
                    end_idx = start_idx+n
                temp_idx = idxes[start_idx:end_idx]
                measured_coordinates = np.array(self.measured.obs.loc[temp_idx,['ccf_z','ccf_y','ccf_x']])
                neighbors,distances = self.tree.query(measured_coordinates,k=1000)
                neighbor_types = cell_labels[neighbors]
                neighbor_types[distances>self.max_distance] = np.nan # turn into a weighted average by distance
                for idx,cell_type in enumerate(cts):
                    priors.loc[temp_idx,cell_type] = np.sum(1*(neighbor_types==cell_type),axis=1)
            priors.to_csv(f)
        self.base_priors = priors
        self.update_user("Calculating Spatial Priors")
        self.priors = {}
        self.class_converters = {}
        for level in self.ref_levels:
            self.level = level
            class_converter = self.class_balanced_reference.obs.loc[:,[base_level,level]]
            class_converter = class_converter.drop_duplicates()
            unique_labels = np.unique(class_converter[level])
            prior = pd.DataFrame(np.zeros([self.base_priors.shape[0],unique_labels.shape[0]]),index=self.base_priors.index,columns=unique_labels)
            if level!=base_level:
                self.class_converters[level] = dict(zip(class_converter[base_level],class_converter[level]))
            else:
                self.class_converters[level] = dict(zip(unique_labels,unique_labels))
            for column in self.base_priors.columns:
                prior.loc[:,self.class_converters[level][column]] = prior.loc[:,self.class_converters[level][column]] + self.base_priors.loc[:,column]
            self.priors[level] = prior

        """ Calculate Centers """
        self.update_user("Calculating Reference Centers")
        self.reference_centers = {}
        self.unq_types = {}
        for level in self.ref_levels:
            self.level = level
            self.unq_types[self.level] = self.class_balanced_reference.obs[self.level].unique()
            self.reference_centers[self.level] = self.calc_centers(self.class_balanced_reference.layers['harmonized'],labels=self.class_balanced_reference.obs[self.level])
    
    def update_user(self,message):
        print(f"{message} - {datetime.now().strftime('%Y %m %d %H %M %S')}")
        logging.info(message)

    def normalize(self):
        """ Initial Normalization""" 
        self.update_user("Performing Initial Normalization")
        self.measured.layers['normalized'] = self.measured.layers['raw'].copy()
        self.reference.layers['normalized'] = self.reference.layers['raw'].copy()
        self.spatial_balanced_reference.layers['normalized'] = self.spatial_balanced_reference.layers['raw'].copy()

        """  Correct For Staining Efficiency Assume that there is a well scalar for each bit (assume median cell is roughly the same across wells )"""
        """ Dont perform for seq as the number of genes present should smooth out this value across bits """
        batch_key = 'section_index'
        for i,bit in enumerate(self.measured.var.index):
            median = np.median(self.spatial_balanced_reference.layers['normalized'][:,i])
            for batch in self.measured.obs[batch_key].unique():
                m = self.measured.obs[batch_key]==batch
                scalar = (median/np.clip(np.median(self.measured.layers['normalized'][m,i]),1,None))
                self.measured.layers['normalized'][m,i] = self.measured.layers['normalized'][m,i]*scalar

        """ Third Assume that each Cell may have their own scalar that sum represents well """
        """ Convert to Roys more robust version TODO """
        standard = np.median(np.sum(self.spatial_balanced_reference.layers['normalized'],axis=1,keepdims=True))
        scalar = standard/np.clip(np.sum(self.measured.layers['normalized'],axis=1,keepdims=True),1,None)
        self.measured.layers['normalized'] = self.measured.layers['normalized'] * scalar
        self.measured.layers['harmonized'] = self.measured.layers['normalized'].copy()

        scalar = standard/np.clip(np.sum(self.reference.layers['normalized'],axis=1,keepdims=True),1,None)
        self.reference.layers['normalized'] = self.reference.layers['normalized'] * scalar
        self.reference.layers['harmonized'] = self.reference.layers['normalized'].copy()

        scalar = standard/np.clip(np.sum(self.spatial_balanced_reference.layers['normalized'],axis=1,keepdims=True),1,None)
        self.spatial_balanced_reference.layers['normalized'] = self.spatial_balanced_reference.layers['normalized'] * scalar
        self.spatial_balanced_reference.layers['harmonized'] = self.spatial_balanced_reference.layers['normalized'].copy()

        """ Fourth Assume that the median and std for each dataset should be preserved across bits """
        """ Quantile Match? TODO """
        for i,bit in enumerate(self.measured.var.index):
            standard_median = np.median(self.spatial_balanced_reference.layers['normalized'][:,i])
            standard_std = np.std(self.spatial_balanced_reference.layers['normalized'][:,i])

            median = np.median(self.measured.layers['normalized'][:,i])
            std = np.std(self.measured.layers['normalized'][:,i])
            self.measured.layers['normalized'][:,i] = (((self.measured.layers['normalized'][:,i]-median)/std)+standard_median)*standard_std

            median = np.median(self.reference.layers['normalized'][:,i])
            std = np.std(self.reference.layers['normalized'][:,i])
            self.reference.layers['normalized'][:,i] = (((self.reference.layers['normalized'][:,i]-median)/std)+standard_median)*standard_std

            median = np.median(self.spatial_balanced_reference.layers['normalized'][:,i])
            std = np.std(self.spatial_balanced_reference.layers['normalized'][:,i])
            self.spatial_balanced_reference.layers['normalized'][:,i] = (((self.spatial_balanced_reference.layers['normalized'][:,i]-median)/std)+standard_median)*standard_std
    
    def calc_centers(self,X, P = None, labels = None):
        """
        calculate the type centers. 
        must supply either probability matrix P or label list
        """
        if P is None and labels is None: 
            raise "Must supply either P or labels"

        if P is not None and labels is not None: 
            raise "Only supply either P or labels - not both"    

        centers = np.zeros((self.unq_types[self.level].shape[0],self.Nbases))
        if labels is not None: 
            for i,name in enumerate(self.unq_types[self.level]):
                ix = np.flatnonzero(labels==name)
                centers[i,:] = X[ix,:].mean(axis=0)
        
        if P is not None: 
            Pnrm = np.sum(P, axis=0)[:, np.newaxis]
            Pnrm[Pnrm==0]=1
            centers = np.dot(P.T, X) / Pnrm

        return centers

    def train(self):
        self.update_user("Training")
        self.likelihood_model = {}
        X = np.array(self.class_balanced_reference.layers['harmonized'])
        for level in self.ref_levels:
            Y = np.array(self.class_balanced_reference.obs[level].copy())
            self.likelihood_model[level] = self.model.fit(X, Y)
                
    def classify(self,iter=5):
        self.update_user("Classifying")
        self.likelihoods = {}
        self.posteriors = {}
        self.measured_centers = {}
        self.measured.layers["initial_harmonized"]  = self.measured.layers['harmonized'].copy()
        for level in self.ref_levels:
            self.level = level
            self.update_user(level)
            for i in range(iter):
                self.update_user(f"Iteration: {str(i)}")
                self.likelihoods[self.level] = pd.DataFrame(self.likelihood_model[self.level].predict_proba(self.measured.layers['harmonized']),
                                                            index = self.priors[self.level].index,columns=self.likelihood_model[self.level].classes_)
                self.likelihoods[self.level][self.likelihoods[self.level].isna()] = 0
                self.priors[self.level][self.priors[level].isna()] = 0
                self.posteriors[self.level] = self.likelihoods[self.level] * self.priors[level]
                self.posteriors[self.level][self.posteriors[self.level].isna()] = 0

                nrm_vec = self.posteriors[self.level].sum(axis=1)
                nrm_vec[nrm_vec==0]=1
                self.posteriors[self.level] = self.posteriors[self.level] / np.array(nrm_vec)[:,None]

                self.measured.obs[self.level] = self.likelihood_model[self.level].classes_[np.argmax(self.posteriors[self.level],axis=1)]

                """ Add Posterior weighting """
                Pnrm = np.sum(np.array(self.posteriors[self.level]), axis=1)[:, np.newaxis]
                Pnrm[Pnrm==0]=1
                centers = self.calc_centers(self.measured.layers['harmonized'],P=Pnrm)
                ref_centers = self.reference_centers[self.level].copy()
                center_shifts = centers-ref_centers
                cell_shifts = np.dot(self.posteriors[self.level],center_shifts)
                self.measured.layers['harmonized'] = self.measured.layers['harmonized'] - cell_shifts

            self.measured.layers[f"{self.level}_harmonized"]  = self.measured.layers['harmonized'].copy()
        return self.measured
    
from tqdm import trange
import gc
from sklearn.feature_selection import f_classif


class KNN(object):
    def __init__(self,train_k=50,predict_k=500,max_distance=np.inf,metric='euclidean',verbose=False,weighted=False):
        self.train_k = train_k
        self.predict_k = predict_k
        self.max_distance = max_distance
        self.metric = metric
        self.verbose = verbose
        self.weighted = weighted

    def fit(self,X,y):
        if self.weighted:
            F, p = f_classif(X, y)
            self.weights = F
            X = X*self.weights
        self.feature_tree_dict = {}
        self.feature_tree_dict['labels'] = y
        self.feature_tree_dict['tree'] = NNDescent(X, metric=self.metric, n_neighbors=self.train_k,n_trees=10,verbose=self.verbose)
        self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))
        self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
        self.feature_tree_dict['labels_index'] = np.array([self.converter[i] for i in y])

    def predict(self,X,y=None):
        if self.weighted:
            X = X*self.weights
        if not isinstance(y,type(None)):
            self.feature_tree_dict['labels'] = y
            self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
            self.feature_tree_dict['labels_index'] = np.array([self.converter[i] for i in y])
        self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))

        neighbors,distances = self.feature_tree_dict['tree'].query(X,k=self.predict_k)
        neighbor_types = self.feature_tree_dict['labels_index'][neighbors]
        neighbor_types[distances>self.max_distance]==-1
        likelihoods = torch.zeros([X.shape[0],self.cts.shape[0]])
        for cell_type in self.cts:
            likelihoods[:,self.converter[cell_type]] = torch.sum(1*torch.tensor(neighbor_types==self.converter[cell_type]),axis=1)
        return self.cts[likelihoods.max(1).indices]
    
    def predict_proba(self,X,y=None):
        if self.weighted:
            X = X*self.weights
        if not isinstance(y,type(None)):
            self.feature_tree_dict['labels'] = y
            self.converter = dict(zip(self.cts,np.array(range(self.cts.shape[0]))))
            self.feature_tree_dict['labels_index'] = np.array([self.converter(i) for i in y])
        self.cts = np.array(sorted(np.unique(self.feature_tree_dict['labels'])))

        neighbors,distances = self.feature_tree_dict['tree'].query(X,k=self.predict_k)
        neighbor_types = self.feature_tree_dict['labels_index'][neighbors]
        neighbor_types[distances>self.max_distance]==-1
        likelihoods = torch.zeros([X.shape[0],self.cts.shape[0]])
        for cell_type in self.cts:
            likelihoods[:,self.converter[cell_type]] = torch.sum(1*torch.tensor(neighbor_types==self.converter[cell_type]),axis=1)
        total = torch.sum(likelihoods,axis=1,keepdims=True)
        total[total==0] = 1
        likelihoods = likelihoods/total
        return likelihoods.numpy()
    
    @property
    def classes_(self):
        return self.cts

class SpatialAssistedLabelTransfer(Classifier): 
    """
    """
    def __init__(self, adata, 
        tax_name='iterative_spatial_assisted_label_transfer',
        ref='allen_wmb_tree', 
        spatial_ref='spatial_reference',
        ref_levels=['class', 'subclass','supertype','cluster'], 
        model='knn',
        out_path='',
        batch_name='section_index',save_fig=False,neuron=None,weighted=False,verbose=True,
        ):
        """
        Parameters
        ----------
        adata : TissueGraph or adata
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
            The TG is required to have a `TG.adata.layers['raw']` matrix
        ref : 
            tag of a reference dataset, or path to a reference dataset (.h5ad)
        spatial_ref : 
            tag of a spatial reference dataset, or path to a spatial reference dataset (.h5ad)
        """
        # specific packages to this task

        if isinstance(adata, TissueGraph.TissueGraph): 
            adata = adata.adata.copy()

        super().__init__(tax=None)
        self.save_fig = save_fig
        self.tax.name = tax_name
        self.ref_path = pathu.get_path(ref, check=True) # refdata
        self.spatial_ref_path = pathu.get_path(spatial_ref, check=True) # refdata
        self.ref_levels = ref_levels 
        self.out_path = out_path
        self.batch_name = batch_name
        self.neuron=neuron
        self.verbose=verbose

        # model could be a string or simply sklearn classifier (many kinds)
        if isinstance(model, str):
            if model == 'nb':
                self.model = GaussianNB()
            elif model == 'mlp':
                self.model = MLPClassifier(
                            hidden_layer_sizes=(50,),  # Adjust number and size of hidden layers as needed
                            activation='relu',  # Choose a suitable activation function
                            max_iter=500,  # Set maximum iterations for training
                            verbose=False
                            )
            elif model == 'knn':
                self.model = KNN(train_k=15,predict_k=100,max_distance=np.inf,metric='correlation')
            else:
                raise ValueError("Need to choose from: `nb`,`mlp`,`knn`")
        else:
            self.model = model
        self.measured = adata.copy()
        self.dataset = self.measured.obs['dataset'].iloc[0]

    def update_user(self,message):
        fileu.update_user(message,verbose=self.verbose)

    def train(self):
        self.update_user("Initializing")

        self.update_user("Training Spatial Model")
        kdesp = KDESpatialPriors(ref_levels=self.ref_levels,neuron=self.neuron)
        kdesp.train()
        self.update_user("Generating Spatial Priors")
        self.priors = {}
        priors,types = kdesp.classify(self.measured, level=self.ref_levels[-1],dim_labels=['ccf_x','ccf_y','ccf_z'])
        priors[np.sum(priors,axis=1)==0,:] = 1 # if all zeros make it uniform
        # priors[np.sum(priors,axis=1)==0,:] = np.median(priors,axis=0) # if all zeros make it the average
        self.priors[self.ref_levels[-1]] = {'columns':types,'indexes':np.array(self.measured.obs.index),'matrix':priors.astype(np.float16)}
        for level in self.ref_levels:
            if level == self.ref_levels[-1]:
                continue
            level_priors,level_types = kdesp.convert_priors(priors,level)
            self.priors[level] = {'columns':level_types,'indexes':np.array(self.measured.obs.index),'matrix':level_priors.astype(np.float16)}
        del kdesp,priors
        gc.collect()
        self.update_user("Generating Spatial Balanced Reference")
        self.reference = anndata.read(self.ref_path)
        self.reference.layers['raw'] = self.reference.X.copy()
        shared_var = list(self.reference.var.index.intersection(self.measured.var.index))

        # bad_bits = ['RS0109_cy5','RSN9927.0_cy5','RS0468_cy5','RS643.0_cy5','RS156.0_cy5','RS0237_cy5']
        # shared_var = [i for i in shared_var if not i in bad_bits]

        self.reference = self.reference[:,np.isin(self.reference.var.index,shared_var)].copy()
        self.measured = self.measured[:,np.isin(self.measured.var.index,shared_var)].copy()
        self.Nbases = self.measured.shape[1]
        level = self.ref_levels[-1]
        idxes = []
        total_cells = np.min([self.measured.shape[0],500000])
        weights = np.mean(self.priors[level]['matrix'],axis=0)
        weights = weights/weights.sum()
        for i,label in enumerate(self.priors[level]['columns']):
            n_cells = int(total_cells*weights[i])
            if n_cells>1:
                m = self.reference.obs[level]==label
                temp = np.array(self.reference.obs[m].index)
                if temp.shape[0]>0:
                    if np.sum(m)>n_cells:
                        idxes.extend(list(np.random.choice(temp,n_cells,replace=False)),)
                    else:
                        idxes.extend(list(np.random.choice(temp,n_cells)))
        self.reference = self.reference[idxes,:].copy()

        self.update_user("Performing Initial Normalization")
        """ Assume that 'normalized from measured has already been size corrected """
        if not 'classification_space' in self.measured.layers.keys():
            self.measured.layers['classification_space'] = self.measured.layers['normalized'].copy()

        """ Assume some size correction has been done to reference """
        if not 'classification_space' in self.reference.layers.keys():
            self.reference.layers['classification_space'] = self.reference.layers['raw'].copy()
        gc.collect()

    def classify(self):
        # self.train()
        self.update_user("Classifying")
        self.likelihoods = {}
        self.posteriors = {}
        for level in self.ref_levels:
            self.likelihoods[level] = {'columns':self.priors[level]['columns'],'indexes':self.priors[level]['indexes'],'matrix':np.zeros_like(self.priors[level]['matrix'])}
            self.posteriors[level] = {'columns':self.priors[level]['columns'],'indexes':self.priors[level]['indexes'],'matrix':np.zeros_like(self.priors[level]['matrix'])}

        measured_coordinates = self.measured.layers['classification_space'].copy()
        reference_coordinates = self.reference.layers['classification_space'].copy()
        level = self.ref_levels[-1]
        self.update_user(f"{level} Classification")
        measured_coordinates = basicu.robust_zscore(measured_coordinates)
        reference_coordinates = basicu.robust_zscore(reference_coordinates)
        self.update_user(f"{level} Training Feature Model")
        gc.collect()
        self.model.fit(reference_coordinates,np.array(self.reference.obs[level]))
        del reference_coordinates
        gc.collect()
        self.update_user(f"{level} Predicting From Features")
        likelihoods = self.model.predict_proba(measured_coordinates).astype(np.float16)
        for idx,ct in enumerate(self.model.classes_):
            jidx = np.where(self.likelihoods[level]['columns']==ct)[0][0]
            self.likelihoods[level]['matrix'][:,jidx] = likelihoods[:,idx]
        del self.model
        gc.collect()
                    
        """ Propagate likelihoods to higher levels """
        for temp_level in self.ref_levels:
            self.update_user(f"{temp_level} Propogating {self.ref_levels[-1]} Likelihoods")
            if temp_level != self.ref_levels[-1]:
                converter = dict(zip(np.array(self.reference.obs[self.ref_levels[-1]]),np.array(self.reference.obs[temp_level])))
                for ct in self.likelihoods[self.ref_levels[-1]]['columns']:
                    if not ct in converter.keys():
                        converter[ct] = np.nan
                for idx,ct in enumerate(self.likelihoods[temp_level]['columns']):
                    ct_idxes = [i for i,c in enumerate(self.likelihoods[self.ref_levels[-1]]['columns']) if converter[c]==ct]
                    self.likelihoods[temp_level]['matrix'][:,idx] = np.sum(self.likelihoods[self.ref_levels[-1]]['matrix'][:,ct_idxes],axis=1)
            self.posteriors[temp_level]['matrix'] = self.likelihoods[temp_level]['matrix']*self.priors[temp_level]['matrix']
            """ if all zeros just use priors """
            self.posteriors[temp_level]['matrix'][np.max(self.posteriors[temp_level]['matrix'],axis=1)==0,:] = self.priors[temp_level]['matrix'][np.max(self.posteriors[temp_level]['matrix'],axis=1)==0,:] 
            """ if all zeros just use likelihoods """
            self.posteriors[temp_level]['matrix'][np.max(self.posteriors[temp_level]['matrix'],axis=1)==0,:] = self.likelihoods[temp_level]['matrix'][np.max(self.posteriors[temp_level]['matrix'],axis=1)==0,:] 
        
            if self.verbose:
                level = temp_level

                metrics = {'priors':self.priors,'likelihoods':self.likelihoods,'posteriors':self.posteriors}
                for metric_label,metric in metrics.items():
                    self.measured.obs[level] = metric[level]['columns'][np.argmax(metric[level]['matrix'],axis=1)]
                    bit = f"Supervised : {level} {metric_label}"
                    n_columns = np.min([6,self.measured.obs[self.batch_name].unique().shape[0]])
                    n_rows = math.ceil(self.measured.obs[self.batch_name].unique().shape[0]/n_columns)
                    fig,axs = plt.subplots(n_columns,n_rows,figsize=[n_rows*3,n_columns*3])
                    fig.patch.set_facecolor('black')
                    fig.suptitle(f"{self.dataset.split('_')[0]} {bit}", color='white')
                    if self.measured.obs[self.batch_name].unique().shape[0]==1:
                        axs = [axs]
                    else:
                        axs = axs.ravel()
                    for ax in axs:
                        ax.axis('off')
                    pallette = dict(zip(self.reference.obs[level], self.reference.obs[level+'_color']))
                    for ct in self.measured.obs[level].unique():
                        if not ct in pallette.keys():
                            pallette[ct] = 'black'
                        else:
                            color = pallette[ct]
                            if not isinstance(color,str):
                                pallette[ct] = 'black'
                    self.measured.obs[level+'_color'] = self.measured.obs[level].map(pallette)
                    for i,section in tqdm(enumerate(sorted(self.measured.obs[self.batch_name].unique())),desc=f"{level} Visualization"):
                        m = (self.measured.obs[self.batch_name]==section)
                        temp_data = self.measured[m,:].copy()
                        c = np.array(temp_data.obs[level+'_color'])
                        ax = axs[i]
                        ax.set_title(section, color='white')
                        ax.axis('off')
                        ax.axis('off')
                        im = ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_y'],c=c,s=0.01,marker='x')
                    if self.save_fig:
                        figure_path = os.path.join(self.out_path, 'Figure')
                        if not os.path.exists(figure_path):
                            os.makedirs(figure_path)
                        plt.savefig(os.path.join(figure_path,f"{bit}.png"))
                    plt.show()
            pallette = dict(zip(self.reference.obs[temp_level], self.reference.obs[temp_level+'_color']))
            self.measured.obs[temp_level] = self.posteriors[temp_level]['columns'][np.argmax(self.posteriors[temp_level]['matrix'],axis=1)]
            self.measured.obs[temp_level+'_color'] = self.measured.obs[temp_level].map(pallette)
            self.measured.obs[temp_level+'_score'] = np.max(self.posteriors[temp_level]['matrix'],axis=1)
            gc.collect()
            """ Add metrics for downstream QC """
        return self.measured

class SpatialOnlyLabelTransfer(Classifier): 
    """
    Predicts label using only X,Y,Z position to transfer label between two TG objects
    """
    def __init__(self, ref_TG, label_to_transfer,k=15):
        self.model =  KNN(train_k=k,predict_k=k,metric='euclidean')
        self.ref_XYZ = np.abs(np.hstack((ref_TG.XY,ref_TG.Z)))
        self.ref_label = ref_TG.adata.obs["label_to_transfer"]
        

    def train(self):
        self.model.fit(self.ref_XYZ,self.ref_label)

    def classify(self,new_TG):
        new_XYZ = np.abs(np.hstack((new_TG.XY,new_TG.Z)))
        labels = self.model.predict(new_XYZ)
        return labels

class KDESpatialPriors(Classifier):
    def __init__(self,
    ref='/scratchdata1/MouseBrainAtlases/Allen',
    ref_levels=['class', 'subclass'],neuron=None):
        if isinstance(ref,str):
            self.ref = TissueGraph.TissueMultiGraph(basepath = ref, input_df = None, redo = False).Layers[0].adata
        else:
            self.ref = ref
        self.ref_levels = ref_levels
        self.neuron = neuron

    def train(self,dim_labels=['x_ccf','y_ccf','z_ccf'],border=1,binsize=0.1):
        XYZ = np.array(self.ref.obs[dim_labels])
        gates = []
        bins = []
        for dim in range(3):
            vmin  = binsize*int((np.min(XYZ[:,dim])-border)/binsize)
            vmax = binsize*int((np.max(XYZ[:,dim])+border)/binsize)
            g = np.linspace(vmin,vmax,int((vmax-vmin)/binsize)+1)
            gates.append(g)
            bins.append(g[:-1]+binsize/2)

        labels = np.array(self.ref.obs[self.ref_levels[-1]])
        types = np.unique(labels)

        if isinstance(self.neuron,bool):
            if self.neuron:
                print('Using Only Neurons')
                types = np.array([i for i in types if not 'NN' in i])
            else:
                print('Using Only Non Neurons')
                types = np.array([i for i in types if 'NN' in i])
            # print(f" Using these Types only {types}")
        typedata = np.zeros([bins[0].shape[0],bins[1].shape[0],bins[2].shape[0],types.shape[0]],dtype=np.float16)
        for i in trange(types.shape[0],desc='Calculating Spatial KDE'):
            label = types[i]
            m = labels==label
            if np.sum(m)==0:
                continue
            hist, edges = np.histogramdd(XYZ[m,:], bins=gates)
            # stk = gaussian_filter(hist,(0.5/binsize,0.25/binsize,0.25/binsize))
            stk = gaussian_filter(hist,(0.25/binsize,0.1/binsize,0.1/binsize))
            typedata[:,:,:,i] = stk
        density = np.sum(typedata,axis=-1,keepdims=True)
        density[density==0] = 1
        typedata = typedata/density
        self.typedata = typedata
        self.bins = bins
        self.types = types
        self.converters = {}
        for level in self.ref_levels:
            if level==self.ref_levels[-1]:
                continue
            self.converters[level] = dict(zip(self.ref.obs[self.ref_levels[-1]],self.ref.obs[level]))

    def convert_priors(self,priors,level):
        converter = self.converters[level]
        types = np.unique([item for key,item in converter.items()])
        updated_priors = np.zeros([priors.shape[0],types.shape[0]])
        for i,label in enumerate(types):
            m = np.array([converter[key]==label for key in self.types])
            updated_priors[:,i] = np.sum(priors[:,m],axis=1)
        return updated_priors,types
        
    def classify(self,measured,level='subclass',dim_labels=['ccf_x','ccf_y','ccf_z']):
        XYZ = np.array(measured.obs[dim_labels])
        XYZ_coordinates = XYZ.copy()
        for dim in range(3):
            XYZ_coordinates[:,dim] = (XYZ_coordinates[:,dim]-self.bins[dim][0])/(self.bins[dim][1]-self.bins[dim][0])
        XYZ_coordinates = XYZ_coordinates.astype(int)

        priors = self.typedata[XYZ_coordinates[:,0],XYZ_coordinates[:,1],XYZ_coordinates[:,2],:]
        types = self.types

        if level!=self.ref_levels[-1]:
            priors,types = self.convert_priors(priors,level)
        return priors,types
    
class LabelTransfer(Classifier): 
    """
    """
    def __init__(self, ref_adata, adata, 
        tax_name='label_transfer',
        ref_levels=['class', 'subclass','supertype','cluster'], 
        model='knn',
        out_path='',
        dataset='infer',
        ):
        """
        Parameters
        ----------
        ref_adata : TissueGraph or adata
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
            The TG is required to have a `TG.adata.layers['raw']` matrix
        adata : TissueGraph or adata
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
            The TG is required to have a `TG.adata.layers['raw']` matrix
        """
        # specific packages to this task

        if isinstance(ref_adata, TissueGraph.TissueGraph): 
            ref_adata = ref_adata.adata.copy()
        if isinstance(adata, TissueGraph.TissueGraph): 
            adata = adata.adata.copy()

        super().__init__(tax=None)
        self.tax.name = tax_name
        self.ref_levels = ref_levels 
        self.out_path = out_path

        # model could be a string or simply sklearn classifier (many kinds)
        if isinstance(model, str):
            if model == 'nb':
                self.model = GaussianNB()
            elif model == 'mlp':
                self.model = MLPClassifier(
                            hidden_layer_sizes=(50,),  # Adjust number and size of hidden layers as needed
                            activation='relu',  # Choose a suitable activation function
                            max_iter=500,  # Set maximum iterations for training
                            verbose=False,
                            )
            elif model == 'knn':
                self.model = KNN(train_k=15,predict_k=50,max_distance=np.inf,metric='euclidean')
            else:
                raise ValueError("Need to choose from: `nb`,`mlp`,`knn`")
        else:
            self.model = model
        self.reference = ref_adata.copy()
        self.measured = adata.copy()
        if dataset=='infer':
            self.dataset = self.measured.obs['dataset'].iloc[0]
        else:
            self.dataset = dataset

    def update_user(self,message):
        time = datetime.now().strftime("%Y %B %d %H:%M:%S")
        print(f"{time} | {message}")
        logging.info(message)

    def train(self):
        self.update_user("Initializing")

        self.update_user("Performing Initial Normalization")
        self.reference.layers['harmonized'] = self.reference.layers['normalized'].copy()
        self.measured.layers['harmonized'] = self.measured.layers['normalized'].copy()
        self.reference.layers['harmonized'] = np.sqrt(np.clip(self.reference.layers['harmonized'].copy(),1,None))
        self.measured.layers['harmonized'] = np.sqrt(np.clip(self.measured.layers['harmonized'].copy(),1,None))
        self.reference.layers['harmonized'] = basicu.normalize_fishdata_robust_regression(self.reference.layers['harmonized'])
        self.measured.layers['harmonized'] = basicu.normalize_fishdata_robust_regression(self.measured.layers['harmonized'])
        self.measured.layers['initial_harmonized'] = self.measured.layers['harmonized'].copy()
        self.reference.layers['initial_harmonized'] = self.reference.layers['harmonized'].copy()

        self.update_user("Training Feature Model")
        self.model.fit(self.reference.layers['harmonized'].copy(),np.array(self.reference.obs[self.ref_levels[-1]]))

    def classify(self):
        self.train()
        self.update_user("Classifying")
        self.likelihoods = {}
        idxes = np.array(self.measured.obs.index)
        for level in self.ref_levels:
            cts = np.unique(np.array(self.reference.obs[level]))
            self.likelihoods[level] = {'columns':cts,'indexes':idxes,'matrix':np.zeros([idxes.shape[0],cts.shape[0]])}
        for i,level in enumerate(self.ref_levels):
            if i==0:
                self.update_user(f"{level} Performing Quantile Normalization on all cells")
                self.measured.layers['harmonized'] = basicu.quantile_matching(np.array(self.reference.layers['initial_harmonized']).copy(),np.array(self.measured.layers['initial_harmonized']).copy())
            else:
                previous_round_measured_labels = self.measured.obs[self.ref_levels[i-1]]
                previous_round_reference_labels = self.reference.obs[self.ref_levels[i-1]]
                for cell_type in tqdm(np.unique(previous_round_measured_labels),desc='Performing Quantile Normalization '):
                    # self.update_user(f"{level} Performing Quantile Normalization on {cell_type}")
                    measured_m = previous_round_measured_labels==cell_type
                    reference_m = previous_round_reference_labels==cell_type
                    if measured_m.sum()>10 and reference_m.sum()>10:
                        self.measured.layers['harmonized'][measured_m,:] = basicu.quantile_matching(np.array(self.reference.layers['initial_harmonized'][reference_m,:]).copy(),np.array(self.measured.layers['initial_harmonized'][measured_m,:]).copy())

            self.update_user(f"{level} Calculating Likelihoods")
            idxes = self.likelihoods[level]['indexes']
            n = 1000
            measured_coordinates = self.measured.layers['harmonized'].copy()
            measured_coordinates
            for start_idx in tqdm(np.arange(0,idxes.shape[0],n),desc='Building Likelihoods'):
                if start_idx+n>idxes.shape[0]:
                    end_idx = idxes.shape[0]
                else:
                    end_idx = start_idx+n
                likelihoods = self.model.predict_proba(measured_coordinates[start_idx:end_idx,:])
                for idx,ct in enumerate(self.model.classes_):
                    jidx = np.where(self.likelihoods[self.ref_levels[-1]]['columns']==ct)[0][0]
                    self.likelihoods[self.ref_levels[-1]]['matrix'][start_idx:end_idx,jidx] = likelihoods[:,idx]
            """ zero out types that dont match higher levels """
            if i>0:
                current_round_reference_labels = self.reference.obs[level]
                previous_round_reference_labels = self.reference.obs[self.ref_levels[i-1]]
                converter = dict(zip(current_round_reference_labels,previous_round_reference_labels))
                previous_round_measured_labels = self.measured.obs[self.ref_levels[i-1]]
                for idx,cell_type in tqdm(enumerate(self.likelihoods[level]['columns']),desc='Updating Likelihood using previous level',total=self.likelihoods[level]['columns'].shape[0]):
                    higher_ct = converter[cell_type]
                    jdx = [j for j,ct in enumerate(previous_round_measured_labels) if higher_ct!=ct]
                    self.likelihoods[level]['matrix'][jdx,idx] = 0

            """ Propagate likelihoods to higher levels """
            for temp_level in self.ref_levels:
                if temp_level == self.ref_levels[-1]:
                    continue
                converter = dict(zip(np.array(self.reference.obs[self.ref_levels[-1]]),np.array(self.reference.obs[temp_level])))
                for ct in self.likelihoods[self.ref_levels[-1]]['columns']:
                    if not ct in converter.keys():
                        converter[ct] = np.nan
                for idx,ct in enumerate(self.likelihoods[temp_level]['columns']):
                    ct_idxes = [i for i,c in enumerate(self.likelihoods[self.ref_levels[-1]]['columns']) if converter[c]==ct]
                    self.likelihoods[temp_level]['matrix'][:,idx] = np.sum(self.likelihoods[self.ref_levels[-1]]['matrix'][:,ct_idxes],axis=1)
         
            if not isinstance(self.out_path,type(None)):
                metrics = {'likelihoods':self.likelihoods}
                for metric_label,metric in metrics.items():
                    self.measured.obs[level] = metric[level]['columns'][np.argmax(metric[level]['matrix'],axis=1)]
                    bit = f"Supervised : {level} {metric_label}"
                    n_columns = 4
                    n_rows = math.ceil(self.measured.obs[self.batch_name].unique().shape[0]/n_columns)
                    fig,axs = plt.subplots(n_columns,n_rows,figsize=[n_rows*5,n_columns*5])
                    fig.patch.set_facecolor('black')
                    fig.suptitle(f"{self.dataset.split('_')[0]} {bit}", color='white')
                    axs = axs.ravel()
                    for ax in axs:
                        ax.axis('off')
                    pallette = dict(zip(self.reference.obs[level], self.reference.obs[level+'_color']))
                    self.measured.obs[level+'_color'] = self.measured.obs[level].map(pallette)
                    for i,section in tqdm(enumerate(sorted(self.measured.obs[self.batch_name].unique())),desc=f"{level} KNN Classification"):
                        m = (self.measured.obs[self.batch_name]==section)
                        temp_data = self.measured[m,:].copy()
                        c = np.array(temp_data.obs[level+'_color'])
                        ax = axs[i]
                        ax.set_title(section, color='white')
                        ax.axis('off')
                        ax.axis('off')
                        im = ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_y'],c=c,s=0.1)

                        ax.set_xlim([-5,5])
                        ax.set_ylim([9,-1])
                    figure_path = os.path.join(self.out_path, 'Figure')
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    plt.savefig(os.path.join(figure_path,f"{bit}.png"))
                    plt.show()
            pallette = dict(zip(self.reference.obs[level], self.reference.obs[level+'_color']))
            self.measured.obs[level] = self.likelihoods[level]['columns'][np.argmax(self.likelihoods[level]['matrix'],axis=1)]
            self.measured.obs[level+'_color'] = self.measured.obs[level].map(pallette)
            # """ Backpropagate upstream classification to match this level"""
            # for temp_level in self.ref_levels:
            #     if temp_level==level:
            #         break
            #     converter = dict(zip(self.reference.obs[level] ,self.reference.obs[temp_level]))
            #     self.measured.obs[temp_level] = self.measured.obs[level].map(converter)
            #     pallette = dict(zip(self.reference.obs[temp_level], self.reference.obs[temp_level+'_color']))
            #     self.measured.obs[temp_level+'_color'] = self.measured.obs[temp_level].map(pallette)
            """ Add metrics for downstream QC """
        return self.measured

import pynndescent
import leidenalg
import igraph as ig

class graphLeiden(Classifier):
    def __init__(self,adata,verbose=True):
        self.adata = adata
        self.verbose = verbose
        self.train()
        
    def train(self):
        adata = self.adata
        X = np.array(adata.layers['classification_space'].copy()).copy()
        fileu.update_user(f"Building Feature Graph",verbose=self.verbose)
        G,knn = tmgu.build_knn_graph(X,metric='correlation')
        self.G = G

    def classify(self,resolution=5):
        adata = self.adata
        fileu.update_user(f"Unsupervised Clustering",verbose=self.verbose)
        TypeVec = self.G.community_leiden(resolution=resolution,objective_function='modularity').membership
        # Convert to PyTorch tensor
        adata.obs['leiden'] =np.array(TypeVec).astype(str)
        cts = np.array(adata.obs['leiden'].unique())
        colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
        pallette = dict(zip(cts, colors))
        adata.obs['leiden_colors'] = adata.obs['leiden'].map(pallette)
        if self.verbose:
                sections = np.unique(adata.obs['Slice'])
                x = adata.obs['ccf_z']
                y = adata.obs['ccf_y']
                c = np.array(adata.obs[f"leiden_colors"])
                cl = np.array(adata.obs[f"leiden"])
                pallette = dict(zip(cl,c))
                cts = np.unique(cl)
                n_columns = np.min([5,sections.shape[0]])
                n_rows = math.ceil((1+sections.shape[0])/n_columns)
                fig,axs  = plt.subplots(n_rows,n_columns,figsize=[5*n_columns,5*n_rows])
                axs = axs.ravel()
                for ax in axs:
                    ax.axis('off')
                plt.suptitle(f"Unsupervised Classification res:{resolution}")
                for idx,section in enumerate(sections):
                    m = np.isin(np.array(adata.obs['Slice']),[section])
                    ax = axs[idx]
                    ax.scatter(x[m],y[m],s=0.01,c=c[m],marker='x')
                    ax.set_title(section)
                    ax.set_aspect('equal')
                    ax.axis('off')
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pallette[key], markersize=10, label=key) for key in cts]
                axs[idx+1].legend(handles=handles, loc='center',ncol=3, fontsize=8)
                axs[idx+1].axis('off')
                plt.show()
        return adata


def filterUsingUnsupervised(adata,label='leiden',verbose=True):
    """ First Remove Sections that dont agree at all"""
    labels = np.array(adata.obs[label].copy()).astype(str)
    section_labels = np.array(adata.obs['Slice'].copy()).astype(str)

    # Convert labels and section_labels to pandas Series
    labels_series = pd.Series(labels, name='Type')
    section_labels_series = pd.Series(section_labels, name='Section')

    # Generate the occupancy matrix
    cm = pd.crosstab(labels_series, section_labels_series)

    correlation_matrix = cm.corr(method='pearson')
    avg_correlation = correlation_matrix.mean(1)
    avg_correlation = avg_correlation-np.median(avg_correlation)
    avg_correlation = avg_correlation/np.std(avg_correlation)
    good_sections = np.array(avg_correlation[np.abs(avg_correlation)<2].index)
    fileu.update_user(f" Removing Sections {np.array(section_labels_series.unique())[np.abs(avg_correlation)>2]}",verbose=verbose)
    adata = adata[adata.obs['Slice'].isin(good_sections)].copy()

    """ Next Remove Cells that are in outlier types """
    
    """ if 50% of the cells are in just 1 section toss it"""
    good_types = cm.index[(cm/np.sum(np.array(cm),axis=1,keepdims=True)).max(1)<0.5]
    bad_types = cm.index[(cm/np.sum(np.array(cm),axis=1,keepdims=True)).max(1)>0.5]
    temp = cm[cm.index.isin(bad_types)]
    temp.loc[:,temp.max(0)>0]
    fileu.update_user(f" Removing Types {bad_types}",verbose=verbose)
    adata = adata[adata.obs['leiden'].isin(good_types)].copy()

    if verbose:
        sections = np.unique(adata.obs['Slice'])
        x = adata.obs['ccf_z']
        y = adata.obs['ccf_y']
        c = np.array(adata.obs[f"leiden_colors"])
        cl = np.array(adata.obs[f"leiden"])
        pallette = dict(zip(cl,c))
        cts = np.unique(cl)
        n_columns = np.min([5,sections.shape[0]])
        n_rows = math.ceil((1+sections.shape[0])/n_columns)
        fig,axs  = plt.subplots(n_rows,n_columns,figsize=[5*n_columns,5*n_rows])
        axs = axs.ravel()
        for ax in axs:
            ax.axis('off')
        plt.suptitle(f"Filtered Unsupervised Classification")
        for idx,section in enumerate(sections):
            m = np.isin(np.array(adata.obs['Slice']),[section])
            ax = axs[idx]
            ax.scatter(x[m],y[m],s=0.01,c=c[m],marker='x')
            ax.set_title(section)
            ax.set_aspect('equal')
            ax.axis('off')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pallette[key], markersize=10, label=key) for key in cts]
        axs[idx+1].legend(handles=handles, loc='center',ncol=3, fontsize=8)
        axs[idx+1].axis('off')
        plt.show()
    return adata

from sklearn.linear_model import LogisticRegression
import joblib
class NeuronClassifier(Classifier):
    def __init__(self,verbose=True,bad_bits=[]):
        self.verbose = verbose
        self.bad_bits = bad_bits
        self.train()

    def train(self):
        fileu.update_user(f"Training Model",verbose=self.verbose)
        reference = anndata.read(pathu.get_path('allen_wmb_tree', check=True))
        reference = reference[:,np.isin(reference.var.index,self.bad_bits,invert=True)].copy()
        reference.layers['classification_space'] = basicu.robust_zscore(reference.X.copy())
        converter = {True:'Non_Neuron', False:'Neuron'}
        labels = np.array([converter[('NN' in i)] for i in reference.obs['subclass']])
        reference_coordinates = np.array(reference.layers['classification_space'])

        # Create a Logistic Regression model
        model = LogisticRegression()

        # Train the model using the reference coordinates and labels
        model.fit(reference_coordinates, labels)

        test = model.predict(reference_coordinates)
        fileu.update_user(f"{str(np.mean(test == labels))} Accuracy",verbose=self.verbose)
        self.model = model

    def classify(self,adata):

        # Convert to PyTorch tensor
        # Predict the labels for the measured coordinates
        measured_coordinates = np.array(adata.layers['classification_space'])
        predicted_labels = self.model.predict(measured_coordinates)
        fileu.update_user(f"Classifying Neurons",verbose=self.verbose)
        adata.obs['neuron'] = predicted_labels
        cts = np.array(adata.obs['neuron'].unique())
        colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
        pallette = dict(zip(cts, colors))
        pallette = {'Non_Neuron': 'r', 'Neuron': 'k'}
        adata.obs['neuron_colors'] = adata.obs['neuron'].map(pallette)
        if self.verbose:
            sections = np.unique(adata.obs['Slice'])
            x = adata.obs['ccf_z']
            y = adata.obs['ccf_y']
            c = np.array(adata.obs[f"neuron_colors"])
            cl = np.array(adata.obs[f"neuron"])
            pallette = dict(zip(cl,c))
            cts = np.unique(cl)
            n_columns = 5
            n_rows = math.ceil((1+sections.shape[0])/n_columns)
            fig,axs  = plt.subplots(n_rows,n_columns,figsize=[5*n_columns,5*n_rows])
            axs = axs.ravel()
            for ax in axs:
                ax.axis('off')
            # plt.suptitle(f"Unsupervised Classification res:{resolution}")
            for idx,section in enumerate(sections):
                m = np.isin(np.array(adata.obs['Slice']),[section])
                ax = axs[idx]
                ax.scatter(x[m],y[m],s=0.01,c=c[m],marker='x')
                ax.set_title(section)
                ax.set_aspect('equal')
                ax.axis('off')
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pallette[key], markersize=10, label=key) for key in cts]
            axs[idx+1].legend(handles=handles, loc='center',ncol=3, fontsize=8)
            axs[idx+1].axis('off')
            plt.show()

        return adata


def splitClassification(adata,ref_levels=['class', 'subclass','supertype','cluster'],weighted=False,verbose=False):
    for level in ref_levels:
        adata.obs[level] = 'Unassigned'
        adata.obs[level+'_color'] = 'Unassigned'
    neuron_adata = adata[adata.obs['neuron']=='Neuron'].copy()
    non_neuron_adata = adata[adata.obs['neuron']!='Neuron'].copy()

    def wrapperClassification(adata,ref_levels=['class', 'subclass','supertype','cluster'],weighted=False,neuron=None,verbose=False):
        gc.collect()
        salt = SpatialAssistedLabelTransfer(adata,ref_levels=ref_levels,batch_name='Slice',model='knn',weighted=weighted,neuron=neuron,verbose=verbose)
        salt.train()
        salt.classify()
        adata = salt.measured.copy()
        del salt
        gc.collect()
        return adata

    neuron_pfunc = partial(wrapperClassification,ref_levels=ref_levels,weighted=weighted,neuron=True,verbose=verbose)
    non_neuron_pfunc = partial(wrapperClassification,ref_levels=ref_levels,weighted=weighted,neuron=False,verbose=verbose)
    # for pfunc,temp_adata in dict(zip([neuron_pfunc,non_neuron_pfunc],[neuron_adata,non_neuron_adata])).items():
    for pfunc,temp_adata in dict(zip([non_neuron_pfunc,neuron_pfunc],[non_neuron_adata,neuron_adata])).items():
        temp_adata = pfunc(temp_adata)
        for level in ref_levels:
            adata.obs.loc[temp_adata.obs.index,level] = temp_adata.obs.loc[temp_adata.obs.index,level].copy()
            adata.obs.loc[temp_adata.obs.index,level+'_color'] = temp_adata.obs.loc[temp_adata.obs.index,level+'_color'].copy()
            adata.obs.loc[temp_adata.obs.index,level+'_score'] = temp_adata.obs.loc[temp_adata.obs.index,level+'_score'].copy()
        del pfunc
        del temp_adata
        gc.collect()
    return adata

