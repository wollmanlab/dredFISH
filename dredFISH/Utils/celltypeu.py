"""
"""
from typing import Union
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
logging.basicConfig(level=logging.INFO)

from . import basicu

class CellTypeClassify(object):
    def __init__(self,
                 X_refdata,
                 y_refdata,
                 X_data,
                 model=None,
                 verbose=True,
                ):
        """
        """
        self.verbose = verbose
        self.model = model
        
        # input
        self.X_refdata = np.array(X_refdata).copy()
        self.y_refdata = np.array(y_refdata).copy() 
        self.X_data = np.array(X_data).copy()
        
        # # got at the end of the run
        # self.y_data
        
    def backup(self):
        """save a copy
        """
        self.bck_X_refdata = self.X_refdata.copy()
        self.bck_X_data = self.X_data.copy()
        self.bck_y_refdata = self.y_refdata.copy()
    
    def reset(self):
        """reset
        """
        self.X_refdata = self.bck_X_refdata.copy()
        self.X_data = self.bck_X_data.copy()
        self.y_refdata = self.bck_y_refdata.copy()

    def buildModel(self):
        # default model
        if self.model == None:
            self.model = KNeighborsClassifier(n_neighbors=15, metric='correlation')
        # otherwise uses the user input
        
    def normalizeData(self, norm='per_bit'):
        # interprete (back compatible)
        if isinstance(norm, bool):
            if norm:
                norm = 'per_bit' # norm=True -> norm = 'per_bit'
            else:
                norm = 'per_cell'
        assert norm in ['per_cell', 'per_bit_equalscale', 'per_bit_refscale', 'per_bit_fscale']

        if norm == 'per_cell':
            # each row sums to 1
            self.X_data = self.X_data/np.sum(self.X_data, axis=1).reshape(-1,1)
            self.X_refdata = self.X_refdata/np.sum(self.X_refdata, axis=1).reshape(-1,1)
        elif norm == 'per_bit_equalscale':
            # each col sums to 0, std 1
            self.X_data = basicu.zscore(self.X_data, axis=0)
            self.X_refdata = basicu.zscore(self.X_refdata, axis=0) # across rows (0) for each column

        elif norm == 'per_bit_refscale':
            # align each dredFISH col to the mean and std of the corresponding scRNA-seq col
            mu1 = np.mean(self.X_data, axis=0)
            sigma1 = np.std(self.X_data, axis=0)
            mu2 = np.mean(self.X_refdata, axis=0)
            sigma2 = np.std(self.X_refdata, axis=0)

            if len(self.X_data) > 1:
                self.X_data = (self.X_data - mu1.reshape(1,-1))*(sigma2/sigma1).reshape(1,-1) + mu2.reshape(1,-1)
            else: # edge case: only 1 cell
                self.X_data = mu2.reshape(1,-1)

        elif norm == 'per_bit_fscale':
            # rescale each bit by F statistics 
            self.X_data = basicu.zscore(self.X_data, axis=0)
            self.X_refdata = basicu.zscore(self.X_refdata, axis=0) # across rows (0) for each column

            clsts = np.unique(self.y_refdata) # guaranteed more than 1 clusters
            X_refclsts = []
            for clst in clsts:
                X_refclst = self.X_refdata[self.y_refdata == clst,:]
                X_refclsts.append(X_refclst)

            f_stats = []
            for i in np.arange(self.X_refdata.shape[1]): # iterate over bits
                f, p = stats.f_oneway(*[X_refclst[:,i] for X_refclst in X_refclsts])
                f_stats.append(f)
            f_stats = np.array(f_stats)

            self.X_data = self.X_data*f_stats.reshape(1,-1)
            self.X_refdata = self.X_refdata*f_stats.reshape(1,-1)

            if self.verbose == 2: # extremely verbose
                f_stats_formatted = np.array2string(100*f_stats/np.sum(f_stats), precision=2, floatmode='fixed')
                expected = 100*1/len(f_stats)
                logging.info(f'F statistics (%): expected = {expected:.2g}\n{f_stats_formatted}')

            return f_stats
    
    def rankNorm(self):
        """rank normalize across bits
        """
        self.X_data = basicu.rank(self.X_data, axis=1) # across cols (1) for each row
        self.X_refdata = basicu.rank(self.X_refdata, axis=1)
            
    def classBalance(self, n_cells: Union[int, dict]=100, random_state=0):
        """ Class Balance  the reference data 
        select `n_cells` from each cluster without replacement
        if n_cells > cluster size, it selects **all** cells 
        """
        # sample
        if isinstance(n_cells, int):
            if n_cells == 0: # use all cells
                pass 
            elif n_cells > 0: # select equal number of cells
                _df = pd.Series(self.y_refdata).to_frame('label')
                idx = basicu.stratified_sample(_df, 'label', n_cells, random_state=random_state).sort_index().index.values

                self.X_refdata = self.X_refdata[idx,:]
                self.y_refdata = self.y_refdata[idx]
                
        elif isinstance(n_cells, dict): # non-equal number of cells
            _df = pd.Series(self.y_refdata).to_frame('label')
            idx = basicu.stratified_sample(_df, 'label', n_cells, random_state=random_state).sort_index().index.values

            self.X_refdata = self.X_refdata[idx,:]
            self.y_refdata = self.y_refdata[idx]
        
    def trainModel(self):
        self.model.fit(self.X_refdata, self.y_refdata)
        
    def predictLabels(self):
        self.y_data = self.model.predict(self.X_data)

    ############################
    def run(self, norm=True, ranknorm=False, n_cells: Union[int, dict]=100, 
        random_state=0,
        ):
        """ sample, normalize, classify
        """
        self.classBalance(n_cells=n_cells, random_state=random_state)
        self.normalizeData(norm=norm)
        if ranknorm: 
            self.rankNorm()

        self.buildModel()
        
        self.trainModel()
        self.predictLabels()
        return self.y_data

    ############################
    def run_bagging(self, n_estimators=10, norm=True, ranknorm=False, n_cells: Union[int, dict]=100, 
        random_state=0,
        ):
        """repeat the same algorithm for different subsets of samples several time, taking the average as the results
        """
        # backup
        self.backup()

        ensem_res = []
        for i_estimate in np.arange(n_estimators):

            self.classBalance(n_cells=n_cells, random_state=random_state)
            self.normalizeData(norm=norm)
            if ranknorm: 
                self.rankNorm()

            self.buildModel()
            
            self.trainModel()
            self.predictLabels()

            ensem_res.append(self.y_data)

            self.reset()

        ensem_res = np.array(ensem_res) # (#ensem, #cells)
        labels, idx = np.unique(ensem_res, return_inverse=True)
        idx = idx.reshape(ensem_res.shape)
        mode, count = stats.mode(idx, axis=0)
        if np.all(count == n_estimators):
            logging.info("Warning: all estimators are the same -- check random seed!!")
        # else:
        #     logging.info("looking good!")
        y_data = labels[mode].reshape(-1,)

        self.y_data = y_data
        return self.y_data
        
    def run_iterative_prior(self, 
                            overall_sample_fraction=0.5,
                            min_total_cells=100,
                            min_cells=10,
                            p0=0, # no momentum
                            subrun_func='run',
                            norm='per_bit_equalscale',
                            ranknorm=False,
                            random_state=0,
                            ):
        """Iteratively set up the number of cell to sample, control overall sample fraction
        """
        # backup
        self.backup()
        
        total_cells = int(overall_sample_fraction*len(self.y_refdata)) # sample this many
        if total_cells < min_total_cells:
            logging.debug(f"Small dataset, sampling {overall_sample_fraction:.2g} is too few, use {min_total_cells} cells instead")
            total_cells = min_total_cells
        
        # initialize
        types, _ = np.unique(self.y_refdata, return_counts=True)
        size_vec = np.repeat([1/len(types)], len(types))
        n_cells = pd.Series((total_cells*size_vec).astype(int), index=types).to_dict()

        diff = 1 
        i_iter = 0
        type_composition_th = 1e-2
        n_max_iter = 10

        while (diff > type_composition_th and i_iter < n_max_iter):
            i_iter += 1
            p = p0/i_iter
            logging.info(f"Re-sample iteration {i_iter}, diff {diff}")

            # run 
            self.y_data = getattr(self, subrun_func)(n_cells=n_cells, norm=norm, ranknorm=ranknorm, random_state=random_state) # note that n_cells are auto-generated
            _types, _counts = np.unique(self.y_refdata, return_counts=True)
            logging.info(f"Reference (scRNA-seq) sampled: {_types} \n {_counts}")
            _types, _counts = np.unique(self.y_data, return_counts=True)
            logging.debug(f"Prediction (dredFISH) sampled: {_types} \n {_counts}")

            # retrieve cluster sizses
            types_new, sizes = np.unique(self.y_data, return_counts=True)
            types_new_idx = basicu.get_index_from_array(types_new, types)
            sizes = np.hstack([sizes, 0])[types_new_idx] # this is to handle zero case
            size_vec_next = sizes/sizes.sum()
            size_vec_next = (1-p)*size_vec_next + p*size_vec # momentum term
            n_cells = pd.Series(np.clip((total_cells*size_vec_next).astype(int), min_cells, None), 
                                index=types).to_dict()

            # update and reset
            diff = np.abs(size_vec_next - size_vec).sum()
            size_vec = size_vec_next
            self.reset()

        # catch the last one
        logging.info(f"final diff {diff}")
        
        return self.y_data
    
    def run_iterative_prior_v2(self, 
                            min_cells_pertype=100,
                            max_ratio=10,
                            min_cells=20,
                            p0=0, # no momentum
                            subrun_func='run',
                            norm='per_bit_equalscale',
                            ranknorm=False,
                            random_state=0,
                            ):
        """Iteratively set up the number of cell to sample, control baseline number of cells
        """
        # backup
        self.backup()
        
        # initialize
        types, _ = np.unique(self.y_refdata, return_counts=True)
        size_vec = np.repeat([1/len(types)], len(types))
        size_ratio = np.clip(size_vec/size_vec.min(), 0, max_ratio)
        n_cells = pd.Series((min_cells_pertype*size_ratio).astype(int), index=types).to_dict()

        diff = 1 
        i_iter = 0
        type_composition_th = 1e-2
        n_max_iter = 10

        while (diff > type_composition_th and i_iter < n_max_iter):
            i_iter += 1
            p = p0/i_iter
            logging.info(f"Re-sample iteration {i_iter}, diff {diff}")

            # run 
            self.y_data = getattr(self, subrun_func)(n_cells=n_cells, norm=norm, ranknorm=ranknorm, random_state=random_state) # note that n_cells are auto-generated
            _types, _counts = np.unique(self.y_refdata, return_counts=True)
            logging.info(f"Reference (scRNA-seq) sampled: {_types} \n {_counts}")
            _types, _counts = np.unique(self.y_data, return_counts=True)
            logging.debug(f"Prediction (dredFISH) sampled: {_types} \n {_counts}")

            # retrieve cluster sizses
            types_new, sizes = np.unique(self.y_data, return_counts=True)
            types_new_idx = basicu.get_index_from_array(types_new, types)
            sizes = np.hstack([sizes, 0])[types_new_idx] # this is to handle zero case
            size_vec_next = sizes/sizes.sum()
            size_vec_next = (1-p)*size_vec_next + p*size_vec # momentum term
            size_ratio_next = np.clip(size_vec_next/size_vec_next.min(), 0, max_ratio)
            n_cells = pd.Series(np.clip((min_cells_pertype*size_ratio_next).astype(int), min_cells, None), 
                                index=types).to_dict()

            # update and reset
            diff = np.abs(size_vec_next - size_vec).sum()
            size_vec = size_vec_next
            self.reset()

        # catch the last one
        logging.info(f"final diff {diff}")
        
        return self.y_data

def iterative_classify(
                 X_refdata,
                 Y_refdata,
                 X_data,
                 levels,
                 run_func='run',
                 run_kwargs_perlevel=[],
                 model=None,
                 ignore_internal_failure=True,
                 verbose=True,
                 ):
    """allow kwargs per level
    """
    y_data = np.empty((len(X_data),len(levels)), dtype=object)
    if len(run_kwargs_perlevel) == 0:
        run_kwargs_perlevel = [dict()]*len(levels)
    if len(run_kwargs_perlevel) == 1:
        run_kwargs_perlevel = list(run_kwargs_perlevel)*len(levels)

    assert len(run_kwargs_perlevel) == len(levels)

    for _iter, (level, run_kwargs) in enumerate(zip(levels, run_kwargs_perlevel)):
        logging.info(f'iteration {_iter+1}/{len(levels)}')
        ### one iteration
        if _iter == 0:
            next_rounds  = [f'level_{_iter}']
            next_rounds_idx_refdata = [np.arange(len(X_refdata))]
            next_rounds_idx_data = [np.arange(len(X_data))]

        # update current to next
        curr_rounds = next_rounds
        curr_rounds_idx_refdata = next_rounds_idx_refdata
        curr_rounds_idx_data = next_rounds_idx_data

        # update next to empty
        next_rounds = []
        next_rounds_idx_refdata = []
        next_rounds_idx_data = []

        # go over each current round
        for _round, idx_refdata, idx_data in zip(curr_rounds, 
                                                 curr_rounds_idx_refdata,
                                                 curr_rounds_idx_data,
                                                ):
            logging.info(f'Current round: {_round}')
            # this round
            
            # initialize
            local_X_refdata = X_refdata[idx_refdata,:]
            local_y_refdata = Y_refdata[:,_iter][idx_refdata]
            local_X_data = X_data[idx_data,:]

            nclsts = len(np.unique(local_y_refdata))
            if nclsts == 1:
                local_y_data = np.repeat(local_y_refdata[0], len(local_X_data))
            else: # nclsts > 1
                rc = CellTypeClassify(
                                local_X_refdata,
                                local_y_refdata,
                                local_X_data,
                                model=model,
                                verbose=verbose,
                                )
                # run
                if not ignore_internal_failure:
                    # run
                    local_y_data = getattr(rc, run_func)(**run_kwargs)
                else:
                    try:
                        # run
                        local_y_data = getattr(rc, run_func)(**run_kwargs)
                    except:
                        logging.info(f'Failed splitting: {_round} \n {sys.exc_info()[0]}')
                        local_y_data = np.repeat('failed', len(local_X_data))
                        
            # catch results
            y_data[idx_data, _iter] = local_y_data

            # set up for the next round
            y_data_unique = np.unique(local_y_data)
            for cluster in y_data_unique:
                next_rounds.append(f'level_{_iter+1}_cluster_{cluster}')
                next_rounds_idx_refdata.append(idx_refdata[(local_y_refdata == cluster)])
                next_rounds_idx_data.append(idx_data[(local_y_data == cluster)])
                
    return y_data