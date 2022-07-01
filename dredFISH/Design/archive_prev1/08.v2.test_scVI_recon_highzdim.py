#!/usr/bin/env python
# coding: utf-8

import scanpy as sc
import anndata
import logging
import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import scvi
from dredFISH.Analysis import basicu

logging.basicConfig(level=logging.INFO)
logging.info('hi')

def get_mse(y_true, y_pred):
    """
    """
    mse = np.power(y_true-y_pred, 2).mean()
    return mse

def get_r2(y_true, y_pred):
    """
    """
    # r2 = 1-(np.power(y_true-y_pred, 2).mean()/np.power(y_true-np.mean(y_true, axis=0), 2).mean())
    r2 = 1 - np.linalg.norm(y_true-y_pred)**2/np.linalg.norm(y_true-np.mean(y_true, axis=0))**2
    return r2

# # file paths and load data
prj_dir = '/bigstore/GeneralStorage/fangming/projects/dredfish/'
dat_dir = prj_dir + 'data/'
res_dir = prj_dir + 'data_dump/'
# allen data
scrna_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_genes_matrix.h5ad'
# analysis metadata
meta_path = '/bigstore/GeneralStorage/fangming/projects/dredfish/data_dump/analysis_meta_Mar31.json'

# allen scrna matrix
ref_data = anndata.read_h5ad(scrna_path)
# analysis
with open(meta_path, 'r') as fh:
    meta = json.load(fh)
ref_data.obs = ref_data.obs.rename({
                                    'class_label': 'Level_1_class_label',
                                    'neighborhood_label': 'Level_2_neighborhood_label',
                                    'subclass_label': 'Level_3_subclass_label',
                                    'cluster_label': 'Level_5_cluster_label',
                                    }, axis=1)
adata = ref_data.copy()
X = adata.X.todense().astype(int) # within the support of Poisson
adata.X = X 

# split and test
np.random.seed(0)
n_latents = [256, 512, 1024]
for n_hl in n_latents: 
    logging.info(f'Testing n_latent = {n_hl}')
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        # model
        scvi.model.SCVI.setup_anndata(adata_train, layer=None, batch_key=None)
        vae = scvi.model.SCVI(adata_train, 
                              n_hidden=n_hl,
                              n_latent=n_hl, 
                              n_layers=2, 
                              gene_likelihood="poisson")

        # train
        # vae.train(max_epochs=3) # test
        vae.train()

        # test
        # z = vae.get_latent_representation()
        # rho_scvi = vae.get_normalized_expression() # fraction of gene expression for each cell (sums to 1)
        # Xhat = vae.posterior_predictive_sample()
        Xhat_test = vae.posterior_predictive_sample(adata=adata_test)

        # eval
        X_test = adata_test.X
        Xn_test = np.log10(X_test+1)
        Xhatn_test = np.log10(Xhat_test+1)
        mse = get_mse(Xn_test, Xhatn_test) 
        r2 = get_r2(Xn_test, Xhatn_test)

        logging.info(f"n_latent={n_hl}, MSE={mse}, r2={r2}")
        # save
        output_model_dir = f'/bigstore/GeneralStorage/fangming/projects/dredfish/data_dump/scvi_model_nz{n_hl}_May24'
        prefix = ''
        vae.save(output_model_dir, prefix, overwrite=True, save_anndata=True)
        break
