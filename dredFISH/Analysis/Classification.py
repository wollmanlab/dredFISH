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
from multiprocessing import Pool

from scipy.stats import entropy, mode, median_abs_deviation
from scipy.spatial.distance import squareform
from scipy.optimize import minimize_scalar

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pynndescent 
import anndata

from dredFISH.Analysis import TissueGraph

from dredFISH.Utils import pathu
from dredFISH.Utils import basicu
from dredFISH.Utils import celltypeu

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
    def __init__(self,k=1,approx_nn= 30,metric = 'correlation'):
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
        super().__init__(tax = None)
        self.k = k
        self.approx_nn = approx_nn
        self.metric = metric

    def train(self, ref_data, ref_label_id):
        """ Builds approx. KNN graph for queries
        """
        self._knn = pynndescent.NNDescent(ref_data,n_neighbors = self.approx_nn, metric = self.metric)
        self._knn.prepare()
        self._ref_label_id = np.array(ref_label_id)
        
        # update the taxonomy
        self.tax.add_types(ref_label_id,ref_data)

    def classify(self, data): 
        """Find class of rows in data based on approx. KNN
        """
        # get indices and remove self. 
        indices,distances = self._knn.query(data,k=self.k)
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
            logging.info("Running LDA in serial") 
            results = [self.lda_fit(n_topics) for n_topics in n_topics_list]
        else:
            n_procs = max(n_procs, len(n_topics_list))
            logging.info(f"Running LDA in parallel with {n_procs} cores")
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
        logging.info("Loading and preprocessing data")
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

        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("This Classifier requires a (cell level) TissueGraph object as input")

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

        #### preproc data
        logging.info("Loading and preprocessing data")

        base_level = self.ref_levels[-1]

        # get ref data features
        self.reference = anndata.read(self.ref_path)
        self.reference.layers['raw'] = self.reference.X.copy()
        self.spatial_reference = anndata.read(self.spatial_ref_path)
        self.measured = self._TG.adata.copy()

        """ Trim Spatial Reference """
        logging.info("Trimming Spatial Reference")
        window = 0.25
        section = np.mean(self.measured.obs['ccf_x'])
        m = (self.spatial_reference.obs['ccf_x']>(section-window))&(self.spatial_reference.obs['ccf_x']<(section+window))
        self.spatial_reference = self.spatial_reference[m].copy()

        """ Build Section Balanced """
        logging.info("Building Section Balanced ")
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
        self.section_balanced_reference = self.reference[idxes].copy()

        """ Initial Normalization""" # Add in Library Size Normalization
        logging.info("Performing Initial Normalization")
        self.measured.layers['harmonized'] = basicu.normalize_fishdata_regress(self.measured.layers['raw'].copy(),value='sum',leave_log=True,log=True,bitwise=True)
        self.reference.layers['harmonized']  = basicu.normalize_fishdata_regress(self.reference.layers['raw'].copy(),value='sum',leave_log=True,log=True,bitwise=False)
        self.section_balanced_reference.layers['harmonized'] = basicu.normalize_fishdata_regress(self.section_balanced_reference.layers['raw'].copy(),value='sum',leave_log=True,log=True,bitwise=False)
        mu_section = np.median(self.section_balanced_reference.layers['harmonized'],axis=0)
        std_section = np.std(self.section_balanced_reference.layers['harmonized'],axis=0)
        mu = np.median(self.reference.layers['harmonized'],axis=0)
        std = np.std(self.reference.layers['harmonized'],axis=0)
        X = self.reference.layers['harmonized'].copy()
        """ account for scaling due to different avg mu """
        X = X/mu.mean()
        X = X*mu_section.mean()
        """ use section mu and std for norm"""
        for i in range(X.shape[1]):
            X[:,i] = X[:,i] - mu_section[i]
            X[:,i] = X[:,i] / std_section[i]
        self.reference.layers['harmonized'] = X.copy()
        X = self.section_balanced_reference.layers['harmonized'].copy()
        """ use section mu and std for norm"""
        for i in range(X.shape[1]):
            X[:,i] = X[:,i] - mu_section[i]
            X[:,i] = X[:,i] / std_section[i]
        self.section_balanced_reference.layers['harmonized'] = X.copy()

        """ Build Class Balanced """
        logging.info("Building Class Balanced")
        idxes = []
        total_cells = 1000000
        cts = np.unique(self.reference.obs[base_level])
        for idx in cts:
            n_cells = int(total_cells/cts.shape[0])
            if n_cells>1:
                temp = np.where(self.reference.obs[base_level]==idx)[0]
                if temp.shape[0]>0:
                    idxes.extend(list(np.random.choice(temp,n_cells)))
        self.class_balanced_reference = self.reference.obs[base_level][idxes].copy()


        """ Calculate Spatial Priors """
        logging.info("Calculating Spatial Priors")
        max_distance = 0.1
        reference_coordinates = np.array(self.spatial_reference.obs[['ccf_z','ccf_y','ccf_z']])
        self.tree = NNDescent(reference_coordinates, metric='euclidean', n_neighbors=100, n_trees=10)
        measured_coordinates = np.array(self.measured.obs[['ccf_z','ccf_y','ccf_z']])
        c = np.array(self.spatial_reference.obs[base_level])
        cts = np.unique(self.reference.obs[base_level])
        priors = pd.DataFrame(np.zeros([self.measured.shape[0],cts.shape[0]]),index=self.measured.obs.index,columns=cts)
        for i,idx in tqdm(enumerate(priors.index),total=priors.shape[0]):
            neighbors,distances = self.tree.query(measured_coordinates[i:i+1,:],k=1000)
            neighbors = neighbors[distances<max_distance]
            for ct,cc in Counter(c[neighbors].ravel()).items():
                priors.loc[idx,ct] = cc
        priors = priors.fillna(0)
        priors = (priors/np.array(priors.sum(1))[:,None])

        self.priors = {}
        self.class_converters = {}
        for level in self.ref_levels:
            class_converter = self.class_balanced_reference.obs.loc[:,[base_level,level]]
            class_converter = class_converter.drop_duplicates()
            unique_labels = self.class_converter.obs[level].unique()
            prior = pd.DataFrame(np.zeros([priors.shape[0],unique_labels.shape[0]]),index=priors.index,columns=unique_labels)
            self.class_converters[level] = dict(zip(class_converter[base_level],class_converter[level]))
            for column in priors.columns:
                prior.loc[:,self.class_converters[level][column]] = prior.loc[:,self.class_converters[level][column]] + priors.loc[:,column]
            self.priors[level] = prior

        """ Calculate Centers """
        logging.info("Calculating Reference Centers")
        self.reference_centers = {}
        for level in self.ref_levels:
            unique_labels = self.class_balanced_reference.obs[level].unique()
            reference_centers = pd.DataFrame(index=unique_labels.shape[0],columns=self.class_balanced_reference.var.index)
            reference_centers[reference_centers.isna()] = 0
            for label in unique_labels:
                m = self.class_balanced_reference.obs[level]==label
                reference_centers.loc[label,:] = np.mean(self.class_balanced_reference.layers['harmonized'][m,:],axis=0)
            self.reference_centers[level] = reference_centers

    def train(self):
        logging.info("Training")
        self.likelihood_model = {}
        X = np.array(self.class_balanced_reference.layers['harmonized'])
        for level in self.ref_levels:
            Y = np.array(self.class_balanced_reference.obs[level].copy())
            self.likelihood_model[level] = self.model.fit(X, Y)
                
    def classify(self,iter=5):
        logging.info("Classifying")
        self.likelihoods = {}
        self.posteriors = {}
        self.measured_centers = {}
        self.measured.layers["initial_harmonized"]  = self.measured.layers['harmonized'].copy()
        for level in self.ref_levels:
            logging.info(level)
            for i in range(iter):
                logging.info(f"Iteration: {str(i)}")
                self.likelihoods[level] = pd.DataFrame(self.likelihood_model[level].predict_proba(self.measured.layers['harmonized']),
                                                            index = self.priors[level].index,columns=self.likelihood_model[level].classes_)
                self.likelihoods[level][self.likelihoods[level].isna()] = 0
                self.priors[level][self.priors[level].isna()] = 0
                self.posteriors[level] = self.likelihoods[level] * self.priors[level]
                self.posteriors[level][self.posteriors[level].isna()] = 0

                # nrm_vec = self.posteriors[level].sum(axis=1, keepdims=True)
                # nrm_vec[nrm_vec==0]=1
                # self.posteriors[level] = self.posteriors[level] / nrm_vec

                self.measured.obs[level] = self.likelihood_model[level].classes_[np.argmax(self.posteriors[level],axis=1)]

                measured_centers = pd.DataFrame(index=self.likelihood_model[level].classes_.shape[0],columns=self.class_balanced_reference.var.index)
                measured_centers[measured_centers.isna()] = 0
                """ Add Posterior weighting """
                for label in self.likelihood_model[level].classes_:
                    m = self.measured.obs[level]==label
                    measured_centers.loc[label,:] = np.mean(self.measured.layers['harmonized'][m,:],axis=0)
                self.measured_centers[level] = measured_centers
                center_shifts = self.measured_centers[level]-self.reference_centers[level]
                cell_shifts = np.dot(self.posteriors[level],center_shifts)
                self.measured.layers['harmonized'] = self.measured.layers['harmonized'] - cell_shifts
            self.measured.layers[f"{level}_harmonized"]  = self.measured.layers['harmonized'].copy()
        return self.measured
