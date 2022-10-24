"""
Data iterators for Allen Institute mouse brain cell atlas 
"""
import json
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product

import torch
import torch.nn as nn


class DataIter():
    """
    Creates data structure for iterating through cell expression data
    
    Attributes
    ----------
        labels: cell type label hierarchy names
        path_dict: dict with paths of all attributes
        n_iter: number of iterations
        cell_train_itr: number of cells in an iteration
    """
    def __init__(self, labels= ( 'Level_2_neighborhood_label',
                                 'Level_3_subclass_label',
                                 'Level_4_supertype_label',
                                 'Level_5_cluster_label' ),
                 path_dict= dict(root_path= '/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference',
                                 prbe_constraints_path= './10X/probe_constraints.npy',
                                 tenx_cells_path= './10X/cells.npy',
                                 tenx_genes_path= './10X/genes.npy',
                                 tenx_metadata_path= './10X/metadata.csv',
                                 tenx_counts_path= './10X/matrix.npy',
                                 smrt_lengths_path= './SmartSeq/gene_length.npy',
                                 smrt_genes_path='./SmartSeq/genes.npy',
                                 smrt_cells_path= './SmartSeq/cells.npy',
                                 smrt_metadata_path= './SmartSeq/metadata.csv',
                                 smrt_counts_path= './SmartSeq/matrix.npy'),
                 n_iter= int(3e3),
                 cell_train_itr= 50):
        self.iter= 0
        self.n_iter= n_iter
        self.finest= labels[-1]
        self.cell_train_itr= cell_train_itr
        self.get_train_objects(path_dict, labels)

    def get_train_objects(self, path_dict, labels):
        """
        Load data from data attribute paths and build label index
        
        Attributes
            path_dict: dict with paths of all attributes
            labels: cell type label hierarchy names
        """
        
        # load data 
        tenx_genes= np.load(os.path.join(path_dict['root_path'], path_dict['tenx_genes_path']))
        tenx_cells= np.load(os.path.join(path_dict['root_path'], path_dict['tenx_cells_path']))
        tenx_metadata= pd.read_csv(os.path.join(path_dict['root_path'], path_dict['tenx_metadata_path']), index_col=0).loc[tenx_cells]
        tenx_counts= np.load(os.path.join(path_dict['root_path'], path_dict['tenx_counts_path']))[tenx_metadata.Level_4_supertype_label != 'LQ']

        smrt_lengths= np.load(os.path.join(path_dict['root_path'], path_dict['smrt_lengths_path']))
        smrt_cells= np.load(os.path.join(path_dict['root_path'], path_dict['smrt_cells_path']))
        smrt_metadata= pd.read_csv(os.path.join(path_dict['root_path'], path_dict['smrt_metadata_path']), index_col=0).loc[smrt_cells]
        smrt_counts_= np.load(os.path.join(path_dict['root_path'], path_dict['smrt_counts_path']))[smrt_metadata.Level_4_supertype_label != 'LQ']
        # normalize counts by degree that gene lengths differ from average 
        smrt_counts= smrt_counts_ / (smrt_lengths/smrt_lengths.mean())

        prbe_constraints= np.load(os.path.join(path_dict['root_path'], path_dict['prbe_constraints_path']))
        
        tenx_metadata['row_ind']= range(tenx_metadata.shape[0])
        smrt_metadata['row_ind']= range(smrt_metadata.shape[0])

        self.tenx_ftrs= torch.FloatTensor(tenx_counts)
        self.smrt_ftrs= torch.FloatTensor(smrt_counts)

        self.cnstrnts= torch.FloatTensor(prbe_constraints)
        
        # factorize labels at coarse and fine levels 
        tenx_metadata['coarse'], coarse_values= tenx_metadata['Level_3_subclass_label'].factorize()
        tenx_metadata['fine'], fine_values= tenx_metadata['Level_5_cluster_label'].factorize()
        self.coarse_values= coarse_values.tolist()
        self.fine_values= fine_values.tolist()
        self.tenx_coarse= torch.LongTensor(tenx_metadata['coarse'].values)
        self.tenx_fine= torch.LongTensor(tenx_metadata['fine'].values)
        
        def remap(i):
            """
            Remaps two cell types from SM2
            """
            lm= {'254_L5 PT CTX' : '245_L5 PT CTX', 
                 '78_Sst*' : '104_Sst*'}
            return lm.get(i, i)

        smrt_metadata['coarse']= [-1 if pd.isna(i) else self.coarse_values.index(i) for i in smrt_metadata['Level_3_subclass_label']]
        smrt_metadata['fine']= [-1 if pd.isna(i) else self.fine_values.index(remap(i)) for i in smrt_metadata['Level_5_cluster_label']]
        self.smrt_coarse= torch.LongTensor(smrt_metadata['coarse'].values)
        self.smrt_fine= torch.LongTensor(smrt_metadata['fine'].values)

        indices= torch.LongTensor(sorted(set(map(tuple, tenx_metadata[tenx_metadata.fine>-1][['coarse','fine']].values.tolist())))).t()
        self.labl_map= torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1])).to_dense()
        
        # find cells that belong to each coarse/fine class 
        self.row_index= {'tenx': {}, 'smrt': {}}
        for coarse, vals in tenx_metadata.groupby('coarse'):
            self.row_index['tenx'][coarse]= {}
            for fine, rows in vals.groupby('fine'):
                self.row_index['tenx'][coarse][fine]= rows['row_ind'].values

        for coarse, vals in smrt_metadata.groupby('coarse'):
            self.row_index['smrt'][coarse]= {}
            for fine, rows in vals.groupby('fine'):
                self.row_index['smrt'][coarse][fine]= rows['row_ind'].values
        

    def cache_train_dat(self, output_path='/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference/cache/_2', drprt=0.):
        '''
        Write files to output directories for training and validation at the level of fine-grained
        labels with 50 observations per label with some level of dropout. Note that this will lead
        some classes to be overly redundant as there are fewer than 50 observations in that category. 
        
        These directories are where the data is cached in partitions. 
        
        Attributes
        ----------
            output_path: output directory path
            drprt: dropout rate
        '''
        drp= nn.Dropout(drprt)
        torch.save(self.labl_map, open(os.path.join(output_path, 'labl_map.pt'), 'wb'))

        for i,(tenx_ind, smrt_ind) in tqdm(enumerate(self)):
            data= {}
            data['tenx_ftrs']= drp(self.tenx_ftrs[tenx_ind])
            data['tenx_fine']= self.tenx_fine[tenx_ind]
            data['tenx_crse']= self.tenx_coarse[tenx_ind]
            
            data['smrt_ftrs']= drp(self.smrt_ftrs[smrt_ind])
            data['smrt_fine']= self.smrt_fine[smrt_ind]
            data['smrt_crse']= self.smrt_coarse[smrt_ind]
            
            data['cnstrnts']= self.cnstrnts
            data['fine_labels']= self.fine_values
            data['coarse_labels']= self.coarse_values
            torch.save(data, open(os.path.join(output_path, 'train_dat/%d.pt'%i), 'wb'))

            if not i%100:
                data= {'tenx':{}, 'smrt':{}}
                data['cnstrnts']= self.cnstrnts
                data['fine_labels']= self.fine_values
                data['coarse_labels']= self.coarse_values
                
                for coarse in self.row_index['tenx']:
                    for l in self.row_index['tenx'][coarse]:
                        inds= np.random.choice(self.row_index['tenx'][coarse][l], 50)
                        data['tenx'][l]= { 'tenx_ftrs': self.tenx_ftrs[inds],
                                           'tenx_fine': self.tenx_fine[inds],
                                           'tenx_crse': self.tenx_coarse[inds]}

                for coarse in self.row_index['smrt']:
                    for l in self.row_index['smrt'][coarse]:
                        inds= np.random.choice(self.row_index['smrt'][coarse][l], 50)
                        data['smrt'][l]= { 'smrt_ftrs': self.smrt_ftrs[inds],
                                           'smrt_fine': self.smrt_fine[inds],
                                           'smrt_crse': self.smrt_coarse[inds]}
                        
                torch.save(data, open(os.path.join(output_path, 'valid_dat/%d.pt'%i), 'wb'))


    def __next__(self):
        """
        Iterator bounded by number of iterations 
        """
        if self.iter<self.n_iter:
            tenx= []
            for j in self.row_index['tenx']:
                if j>=0:
                    n_fine= len(self.row_index['tenx'][j])
                    counts= np.random.multinomial(self.cell_train_itr, [1./n_fine]*n_fine)
                    for f,l in enumerate(self.row_index['tenx'][j].values()):
                        tenx.extend(np.random.choice(l, counts[f]))

            smrt= []
            for j in self.row_index['smrt']:
                if j>=0:
                    n_fine= len(self.row_index['smrt'][j])
                    counts= np.random.multinomial(self.cell_train_itr, [1./n_fine]*n_fine)
                    for f,l in enumerate(self.row_index['smrt'][j].values()):
                        smrt.extend(np.random.choice(l, counts[f]))

            self.iter+=1
            return tenx, smrt
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class DataIterCached():
    """
    Load cached data.
    
    Attributes
    ----------
        cached_path: path of 
        n_iter: number of iterations
    """
    def __init__(self, cached_path= '/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference/cache/_2',
                 n_iter= int(3e3)):
        self.iter= 0
        self.n_iter= n_iter
        self.labl_map= torch.load(os.path.join(cached_path, 'labl_map.pt'))
        self.train_files= [os.path.join(cached_path, 'train_dat', i) for i in os.listdir(os.path.join(cached_path, 'train_dat')) if i.endswith('pt')]
        self.valid_files= [os.path.join(cached_path, 'valid_dat', i) for i in os.listdir(os.path.join(cached_path, 'valid_dat')) if i.endswith('pt')]
        self.current= torch.load(np.random.choice(self.train_files))

    def validation(self):
        """
        Loads random validation data set 
        """
        return torch.load(np.random.choice(self.valid_files))

    def __next__(self):
        """
        Loads random training data set if iterations have not surpassed max 
        """
        if self.iter<=self.n_iter:
            f= np.random.choice(self.train_files)
            self.current= torch.load(f)
            self.iter+=1
            return self.current
        else:
            raise StopIteration()

    def __iter__(self):
        return self

