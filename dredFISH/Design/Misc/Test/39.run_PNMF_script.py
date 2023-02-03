import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import os
import time
import logging

from dredFISH.Design import PNMF

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

res_dir = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python'
output = os.path.join(res_dir, "smrt_X_v1.h5ad")
logging.info(res_dir)
# data (CPM with rep)
scrna_genes_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/smrt_withrep_X_subL5n100.npy'

# allen scrna matrix (CPM; only 10k genes)
X = np.load(scrna_genes_path, allow_pickle=True).T.copy() # cell by gene

# # # # run PNMF orig -- what we do and have to do
ti = time.time()
logging.info('PCA')
w_pca, rec_pca = PNMF.get_PNMF(X.T, init='pca',     k=24, verbose=True, report_stride=30)
print(time.time()-ti)
res_adata = anndata.AnnData(w_pca)
res_adata.layers['w_pca'] = w_pca
res_adata.write(output)

logging.info('PCA 2X')
w_p2x, rec_p2x = PNMF.get_PNMF(X.T, init='pca_2x',  k=24, verbose=True, report_stride=30)
print(time.time()-ti)
res_adata.layers['w_p2x'] = w_p2x
res_adata.write(output)

logging.info('Normal')
w_nrm, rec_nrm = PNMF.get_PNMF(X.T, init='normal',  k=24, verbose=True, report_stride=30)
print(time.time()-ti)
res_adata.layers['w_nrm'] = w_nrm
res_adata.write(output)

logging.info('Uniform')
w_uni, rec_uni = PNMF.get_PNMF(X.T, init='uniform', k=24, verbose=True, report_stride=30)
print(time.time()-ti)
res_adata.layers['w_uni'] = w_uni
res_adata.write(output)