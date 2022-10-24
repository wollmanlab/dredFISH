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

prj_dir = '/bigstore/GeneralStorage/fangming/projects/dredfish/'
dat_dir = prj_dir + 'data/'
res_dir = prj_dir + 'res_dpnmf/v_python'
print(res_dir)
fig_dir = prj_dir + 'figures/'

# data (old)
f = '/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_genes_matrix.h5ad'
adata = anndata.read_h5ad(f, backed='r') # library size normed (no log)
# X = np.array(adata.X.todense()).copy() # already library size normalized
# adata

# data (CPM with rep)
scrna_genes_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100.npy'
cell_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100_cells.csv'
clst_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100_y_L5.npy'

# allen scrna matrix (CPM; only 10k genes)
X = np.load(scrna_genes_path, allow_pickle=True).T # cell by gene
cells = pd.read_csv(cell_path)['0'].values
# X.sum(axis=1)
y_l5 = np.load(clst_path, allow_pickle=True)

meta = adata.obs.copy()
meta = meta.loc[cells]
meta[['cluster_label', 'subclass_label', 'neighborhood_label', 'class_label']]

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

r2s = []
ks = np.array([2,5,10,16,24,48,100])
for k in ks:
    logging.info(k)
    # Xmat = X[:,:1000].T.copy()
    Xmat = X.T.copy()
    w_uni, rec_uni = PNMF.get_PNMF(Xmat, init='uniform', k=k, verbose=True, report_stride=30)
    r2 = calc_r2(Xmat, w_uni)
    r2s.append(r2)
r2s = np.array(r2s)

# compare with PCA
res_pca = PCA(n_components=100).fit(Xmat)
var_ratios = res_pca.explained_variance_ratio_
cum_var_ratios = np.cumsum(var_ratios)

dict_res = {}
dict_res['k'] = ks.tolist()
dict_res['pnmf_r2'] = r2s.tolist()
dict_res['pca_r2'] = cum_var_ratios.tolist()
with open("/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/var_ratio_10kgenes.json", "w") as fp:
    json.dump(dict_res, fp)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(ks, r2s, '-o', label="PNMF")
ax.plot(ks, cum_var_ratios[ks-1], '-o', label="PCA")
# ax.plot(2*ks, cum_var_ratios[ks-1], '--o', label="PCA (2x)", color='gray')
ax.set_xlabel("Number of components (k)")
ax.set_ylabel("Explained variance ratio")
ax.set_ylim([0,1])
ax.legend(bbox_to_anchor=(1,1))
ax.set_title(f"Test: {len(Xmat)} genes by 100 cells per L5 (~300) types")
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(ks, r2s, '-o', label="PNMF")
ax.plot(ks, cum_var_ratios[ks-1], '-o', label="PCA")
ax.plot(ks*2, cum_var_ratios[ks-1], '--o', label="PCA (2x)", color='gray')
ax.set_xlabel("Number of components (k)")
ax.set_ylabel("Explained variance ratio")
ax.set_ylim([0,1])
ax.legend(bbox_to_anchor=(1,1))
ax.set_title(f"Test: {len(Xmat)} genes by 100 cells per L5 (~300) types")
plt.show()