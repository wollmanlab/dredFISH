"""
"""
import scanpy as sc
import logging
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# import scanpy.external as sce
# import matplotlib.pyplot as plt
# import seaborn as sns
logging.basicConfig(level=logging.INFO)

from dredFISH.Analysis import basicu
# from dredFISH.Analysis import TissueGraph as tgh

class CellTypeClassify(object):
    def __init__(self,
                 X_refdata,
                 y_refdata,
                 X_data,
                 model=None,
                 verbose=True):
        """
        """
        self.verbose = verbose
        self.model = model
        self.X_refdata = np.array(X_refdata).copy()
        self.y_refdata = np.array(y_refdata).copy() 
        self.X_data = np.array(X_data).copy()
        
        # # got at the end of the run
        # self.y_data

    def run(self):
        self.normalizeData()
        self.classBalance()
        self.buildModel()
        
        self.trainModel()
        self.predictLabels()
        return self.y_data
        
    def buildModel(self):
        # default model
        if self.model == None:
            self.model = KNeighborsClassifier(n_neighbors=15, metric='correlation')
        # other wise uses the user input
        
    def normalizeData(self):
        self.Xnorm_data = basicu.zscore(self.X_data, axis=0)
        self.Xnorm_refdata = basicu.zscore(self.X_refdata, axis=0) # across rows (0) for each column

    def classBalance(self, n_cells=100):
        """ Class Balance 
        select `n_cells` from each cluster without replacement
        if n_cells > cluster size, it selects **all** cells 
        """
        # sample
        _df = pd.Series(self.y_refdata).to_frame('label')
        idx = basicu.stratified_sample(_df, 'label', n_cells).sort_index().index.values
        
        self.Xbalanced_refdata = self.Xnorm_refdata[idx,:]
        self.ybalanced_refdata = self.y_refdata[idx]
        
    def trainModel(self):
        self.model.fit(self.Xbalanced_refdata, self.ybalanced_refdata)
        
    def predictLabels(self):
        self.y_data = self.model.predict(self.Xnorm_data)
        

def iterative_classify(
                 X_refdata,
                 Y_refdata,
                 X_data,
                 levels,
                 model=None,
                 verbose=True):
    """
    """
    y_data = np.empty((len(X_data),len(levels)), dtype=object)

    for _iter, level in enumerate(levels):
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
            local_X_refdata = X_refdata[idx_refdata,:]
            local_y_refdata = Y_refdata[:,_iter][idx_refdata]
            local_X_data = X_data[idx_data,:]
            rc = CellTypeClassify(
                             local_X_refdata,
                             local_y_refdata,
                             local_X_data,
                             model=model,
                             verbose=verbose)
            local_y_data = rc.run()

            # catch results
            y_data[idx_data, _iter] = local_y_data

            # set up for the next round
            y_data_unique = np.unique(local_y_data)
            for cluster in y_data_unique:
                next_rounds.append(f'level_{_iter+1}_cluster_{cluster}')
                next_rounds_idx_refdata.append(idx_refdata[(local_y_refdata == cluster)])
                next_rounds_idx_data.append(idx_data[(local_y_data == cluster)])
                
    return y_data