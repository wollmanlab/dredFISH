#!/usr/bin/env python

import logging
import numpy as np
import os

from models import PNMF

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

# specify input (CPM with rep) and output (factorized weight matrix) 
input_X_path  = '/greendata/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100.npy'
output_w_path = '/greendata/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/output_demo/PNMF_smrt.npy'
logging.info(f"{input_X_path} -> \n" + f"{output_w_path}")

# parameters
select_first_few_genes = 1000 # or 0 for not including a restriction
k = 24

# load data - allen scrna matrix (CPM; only 10k genes)
logging.info('load data...')
X = np.load(input_X_path, allow_pickle=True) # this is gene by cell; # .T.copy() # cell by gene
logging.info(f'data shape: {X.shape}')
# select first few genes if used 
if select_first_few_genes > 0:
    X = X[:select_first_few_genes,:]
logging.info(f'data shape (updated): {X.shape}')

# run and save results
logging.info('run PNMF...')
w, rec = PNMF.get_PNMF(X, k, init='normal', verbose=True, report_stride=30)
logging.info('save results...')
outdir = os.path.dirname(output_w_path)
if not os.path.isdir(outdir):
    os.mkdir(outdir)
np.save(output_w_path, w)
logging.info(f'Done! Saved results to: {output_w_path}')
