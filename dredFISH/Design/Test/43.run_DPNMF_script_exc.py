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

# get the S matrix
ctrds_l3, types_l3 = basicu.group_mean(X, meta['subclass_label'].values)
logging.info(ctrds_l3.shape)
logging.info(types_l3)

ctrds_l2, types_l2 = basicu.group_mean(X, meta['neighborhood_label'].values)
logging.info(ctrds_l2.shape)
logging.info(types_l2)

ctrds_l1, types_l1 = basicu.group_mean(X, meta['class_label'].values)
logging.info(ctrds_l1.shape)
logging.info(types_l1)

types_l3tol1 = meta.groupby('subclass_label')['class_label'].first().loc[types_l3].values
types_l3tol1_code, types_l1_ = pd.factorize(types_l3tol1, sort=True)
assert np.all(types_l1_ == types_l1)
ctrds_diff = ctrds_l3 - ctrds_l1[types_l3tol1_code]
logging.info(f"cluster diff mat: {ctrds_diff.shape}")

# select to change exc cells only
ctrds_diff = ctrds_diff[types_l3tol1 == 'Glutamatergic']#, # _code
logging.info(f"cluster diff mat (exc only): {ctrds_diff.shape}")


Sb = ctrds_diff.T.dot(ctrds_diff)
S = -Sb
k = 24
mus = [0, 1, 100, 10000, 1e6]

output = os.path.join(res_dir, "smrt_X_DPNMF_test_v1_exc.h5ad")
res_adata = anndata.AnnData(np.zeros((1000,24)))
for mu in mus:
    logging.info(f'DPNMF test {mu:.1e}')
    w_uni, rec_uni = PNMF.get_DPNMF(X[:,:1000].T, k, S[:1000,:1000], mu,
                                    init='uniform', verbose=True, report_stride=30)
    res_adata.layers[f'w_mu{mu:.1e}'] = w_uni
    res_adata.write(output)

output = os.path.join(res_dir, "smrt_X_DPNMF_v1_exc.h5ad")
res_adata = anndata.AnnData(np.zeros((X.shape[1],24)))
for mu in mus:
    logging.info(f'DPNMF all {mu:.1e}')
    w_uni, rec_uni = PNMF.get_DPNMF(X.T, k, S, mu,
                                    init='uniform', verbose=True, report_stride=30)
    res_adata.layers[f'w_mu{mu:.1e}'] = w_uni
    res_adata.write(output)