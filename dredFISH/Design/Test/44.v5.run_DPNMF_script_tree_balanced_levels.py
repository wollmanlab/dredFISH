import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import os
import time
import logging

from dredFISH.Design import PNMF
from dredFISH.Utils import basicu

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

res_dir = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python'
logging.info(res_dir)

# data (old)
f = '/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_genes_matrix.h5ad'
adata = anndata.read_h5ad(f, backed='r') # library size normed (no log)

# data (CPM with rep)
scrna_genes_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100.npy'
cell_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100_cells.csv'
clst_path        = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100_y_L5.npy'


# allen scrna matrix (CPM; only 10k genes)
X = np.load(scrna_genes_path, allow_pickle=True).T.copy() # cell by gene
cells = pd.read_csv(cell_path)['0'].values
meta = adata.obs.copy()
meta = meta.loc[cells]

### get the S matrix (weighted hierarchical)
# get a tree
# L5 - L3 - L1 - L0
levels = [
     'cluster_label', 
     'subclass_label', 
     'class_label',
    ]
tree = meta.groupby(levels).size() #['Lim1'].mean().dropna()
tree = tree[tree!=0]
tree = tree.reset_index()[levels]

# centroids
ctrds_lvl = []
types_lvl = []
# sub levels
for level in levels:
    logging.info(level)
    # get centroids
    ctrds_, types_ = basicu.group_mean(X, meta[level].values)
    logging.info(f"{ctrds_.shape}")
    logging.info(f"{types_.shape}")
    ctrds_lvl.append(ctrds_)
    types_lvl.append(types_)
# level 0 (global)
ctrds_, types_ = basicu.group_mean(X, ['']*len(meta))
ctrds_lvl.append(ctrds_)
types_lvl.append(types_)

# centroid diff and Sb
w_lvl = [1.0/382, 1.0/44 , 1.0/3] #
w_lvl = w_lvl/np.mean(w_lvl)
logging.info(f"{w_lvl}")

ngenes = ctrds_lvl[0].shape[1]
Sb = np.zeros((ngenes, ngenes))
l2maxmean = 0 # for ease of scaling
for i in range(len(levels)):
    a = ctrds_lvl[i+1]
    b = ctrds_lvl[i]
    
    _types = types_lvl[i]
    _lc = levels[i]
    if i+1 < len(levels):
        _lu = levels[i+1]
        types_map1up = meta.groupby(_lc)[_lu].first().loc[_types].values
        types_map1up_code, _types_u = pd.factorize(types_map1up, sort=True)
        assert np.all(_types_u == types_lvl[i+1])
        
        ctrds_diff = b - a[types_map1up_code]
    else:
        ctrds_diff = b - np.repeat(a, len(b), axis=0)
    
    # save ctrds_diff for each level
    logging.info(f"{i}, {ctrds_diff.shape}")
    
    # norm
    l2 = np.linalg.norm(ctrds_diff, axis=1) # no squared
    l2maxmean = max(np.mean(l2), l2maxmean) # max (across all levels) of the mean l2 per level
    # print(l2)
    
    ctrds_diff_norm = ctrds_diff/np.clip(l2.reshape(-1,1), 1e-5, None) # norm per cluster
    # l2n = np.linalg.norm(ctrds_diff_norm, axis=1) # l2n should be 1
    # print(l2n)
    
    Sb = Sb + w_lvl[i]*ctrds_diff_norm.T.dot(ctrds_diff_norm)
Sb = (l2maxmean**2)*Sb # restore unit/scale
### end -- calc Sb

### set up DPNMF
S = -Sb
k = 24
# mus = [0, 1e2, 1e4, 1e6]
mus = [     2e2, 5e2, 
       1e3, 2e3, 5e3, 
            2e4, 5e4,
       1e5, 2e5, 5e5,
       ]
output = os.path.join(res_dir, "smrt_X_DPNMF_testrun_tree_v2-p2.h5ad")
res_adata = anndata.AnnData(np.zeros((1000,24)))
for mu in mus:
    logging.info(f'DPNMF test {mu:.1e}')
    w, rec = PNMF.get_DPNMF(X[:,:1000].T, k, S[:1000,:1000], mu,
                                    init='normal', verbose=True, report_stride=30)
    res_adata.layers[f'w_mu{mu:.1e}'] = w
    res_adata.write(output)

output = os.path.join(res_dir, "smrt_X_DPNMF_tree_v2-p2.h5ad")
res_adata = anndata.AnnData(np.zeros((X.shape[1],24)))
for mu in mus:
    logging.info(f'DPNMF all {mu:.1e}')
    w, rec = PNMF.get_DPNMF(X.T, k, S, mu,
                                    init='normal', verbose=True, report_stride=30)
    res_adata.layers[f'w_mu{mu:.1e}'] = w
    res_adata.write(output)