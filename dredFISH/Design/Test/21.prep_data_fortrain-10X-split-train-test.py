"""
"""
from glob import escape
import anndata
import pandas as pd
import numpy as np
import scanpy as sp
import h5py
import zarr
import os
import tqdm

from dredFISH.Analysis import basicu
import logging


def split_train_test(zarr_file, keys_copy, keys_split, frac=0.9, chunksize=10, random_seed=None):
    """randomly select frac vs 1-frac samples into training and test (validation) set.
    Save them as separate zarr files
    """
    assert frac <= 1 and frac >= 0
    # the original zarr file
    z = zarr.open(zarr_file, 'r')
    size = len(z['counts'])
    
    path_train = zarr_file.replace('.zarr', '_train.zarr')
    path_test = zarr_file.replace('.zarr', '_test.zarr')
    logging.info(f"{zarr_file} -> \n{path_train} and \n{path_test}\n")
    if random_seed: np.random.seed(random_seed)
    cond_train = np.random.rand(size) < frac
    ntrain = cond_train.sum()
    ntest = (~cond_train).sum()
    logging.info(f"{size}, {ntrain} ({ntrain/size:.3f}), {ntest} ({ntest/size:.3f})")
    
    z_train = zarr.open(path_train, mode='w')
    z_test = zarr.open(path_test, mode='w')
    for key in keys_copy:
        logging.info(key)
        z_train[key] = z[key]
        z_test[key] = z[key]

    for key in keys_split:
        logging.info(key)
        trn_idx = np.arange(size)[cond_train]
        tst_idx = np.arange(size)[~cond_train]

        if z[key].ndim == 1:
            chunks = (chunksize,)
            shape_trn = (len(trn_idx), )
            shape_tst = (len(tst_idx), )
            logging.info(shape_trn)
            logging.info(shape_tst)
            # create datasets
            z_train.create_dataset(key, shape=shape_trn, chunks=chunks)
            z_test.create_dataset(key, shape=shape_tst, chunks=chunks)
            # load train
            dat = z[key].oindex[trn_idx]
            z_train[key][:] = dat
            # load test
            dat = z[key].oindex[tst_idx]
            z_test[key][:] = dat

        elif z[key].ndim == 2:
            chunks = (chunksize,None)
            _sample = z[key].oindex[0,:]
            n_ftrs = len(_sample)
            shape_trn = (len(trn_idx), n_ftrs)
            shape_tst = (len(tst_idx), n_ftrs)
            logging.info(shape_trn)
            logging.info(shape_tst)
            # create datasets
            z_train.create_dataset(key, shape=shape_trn, chunks=chunks)
            z_test.create_dataset(key, shape=shape_tst, chunks=chunks)


            #### load in chunks
            readchunksize = 100000
            # load train
            if len(trn_idx) < readchunksize: 
                dat = z[key].oindex[trn_idx,:]
                z_train[key][:] = dat
            else: # too large
                curr_head = 0
                ncells = len(trn_idx)
                while tqdm.tqdm(curr_head < ncells):
                    l, r = curr_head, curr_head+readchunksize
                    r = min(ncells, r)
                    curr_head += readchunksize
                    subidx = trn_idx[l:r]
                    logging.info(f"{l},{r}, {subidx.shape}, {subidx[0]}, {subidx[-1]}")
                    z_train[key].oindex[l:r,:] = z[key].oindex[subidx,:]
            
            # load test
            if len(tst_idx) < readchunksize: 
                dat = z[key].oindex[tst_idx,:]
                z_test[key][:] = dat
            else: # too large
                curr_head = 0
                ncells = len(tst_idx)
                while tqdm.tqdm(curr_head < ncells):
                    l, r = curr_head, curr_head+readchunksize
                    r = min(ncells, r)
                    curr_head += readchunksize
                    subidx = tst_idx[l:r]
                    logging.info(f"{l},{r}, {subidx.shape}, {subidx[0]}, {subidx[-1]}")
                    z_test[key].oindex[l:r,:] = z[key].oindex[subidx,:]
            #### end load in chunks
        else:
            raise ValueError('unimplemented')
    return 

logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )
dat_dir = '/bigstore/GeneralStorage/fangming/projects/dredfish/data/'
output = os.path.join(dat_dir, 'rna', 'scrna_10x_ctxhippo_a_count_matrix_v3.zarr')
logging.info(output)

z = zarr.open(output, mode='r')
#
keys_copy = ['num_probe_limit', 'l1_cat', 'l2_cat', 'l3_cat', 'l5_cat']
keys_split = [key for key in z.keys() if key not in keys_copy]
# keys_split = [key for key in keys_split if key != 'counts'] # debugging only
logging.info(f"Copy: {keys_copy}")
logging.info(f"Split: {keys_split}")

# actually split the data
logging.info('begin splitting')
split_train_test(output, keys_copy, keys_split, frac=0.9)