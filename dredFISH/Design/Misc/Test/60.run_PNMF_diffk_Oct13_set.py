#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import os
from sklearn.decomposition import PCA
import logging

from dredFISH.Design import PNMF
from dredFISH.Utils import basicu
from dredFISH.Utils.__init__plots import * 
import json

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

def calc_err(x, w, rho=1):
    """
    X: (p, n)
    w: (p, k)
    """
    return np.linalg.norm(x-rho*w.dot(w.T.dot(x)), ord='fro')**2

def calc_ss(x):
    """
    X: (p, n)
    """
    xmean = np.mean(x, axis=1)
    xcentered = x-xmean.reshape(-1,1)
    return (np.linalg.norm(xcentered, ord='fro')**2)

def calc_r2(x, w):
    """
    X: (p, n)
    w: (p, k)
    """
    xmean = np.mean(x, axis=1)
    xcentered = x-xmean.reshape(-1,1)
    return 1 - calc_err(x, w)/calc_ss(x)

# data (CPM with rep) new genes
scrna_genes_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100.npy'
cell_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100_cells.csv'
clst_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100_y_L5.npy'
output           = "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/var_ratio_oct13set_v4.json" 

# allen scrna matrix (CPM)
X = np.load(scrna_genes_path, allow_pickle=True).T # cell by gene
logX = np.log10(X+1)  
cells = pd.read_csv(cell_path)['0'].values

dict_res = {}
ks = np.array([2,5,10,16,24,48,100])
dict_res['k'] = ks.tolist()

### section 1: CPM
X_todo = X.copy() # use the original matrix (CPM) for PNMF; note that for PNMF the convention is (p,n) -- the transpose of this matrix
# use normal counts
r2s = []
for k in ks:
    logging.info(k)
    w, rec = PNMF.get_PNMF(X_todo.T, 
                           init='normal', k=k, verbose=True, report_stride=30)
    r2 = calc_r2(X_todo.T, w)
    r2s.append(r2)
r2s = np.array(r2s)
# PCA
res_pca = PCA(n_components=100).fit(X_todo) # no transpose here
cum_var_ratios = np.cumsum(res_pca.explained_variance_ratio_)

dict_res['r2_pnmf'] = r2s.tolist()
dict_res['r2_pca'] = cum_var_ratios.tolist()
### End of section 1

### section 2: log10(CPM+1)
X_todo = logX.copy() # use the original matrix (CPM) for PNMF; note that for PNMF the convention is (p,n) -- the transpose of this matrix
r2s = []
for k in ks:
    logging.info(k)
    w, rec = PNMF.get_PNMF(X_todo.T, 
                           init='normal', k=k, verbose=True, report_stride=30)
    r2 = calc_r2(X_todo.T, w)
    r2s.append(r2)
r2s = np.array(r2s)
# PCA logX
res_pca = PCA(n_components=100).fit(X_todo) # no transpose here
cum_var_ratios = np.cumsum(res_pca.explained_variance_ratio_)

dict_res['r2_pnmf_logx'] = r2s.tolist()
dict_res['r2_pca_logx'] = cum_var_ratios.tolist()
### End of section 2

# write
with open(output, "w") as fp:
    json.dump(dict_res, fp)