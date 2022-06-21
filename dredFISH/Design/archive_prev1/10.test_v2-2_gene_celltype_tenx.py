import numpy as np
import torch
import os
from dredFISH.Design.model_v1_gene_celltype import train_model
from Design import data_loader_scrna_tinydata
import sys
import glob

disable_tqdm = False # True # False
print(f"disable tqdm (clean log): {disable_tqdm}")
print(f"GPU: {torch.cuda.is_available()}")
    
n_epochs = 3 
n_iter = None #3 

lmd0 = 1 
n_bit = 70
n_rcn = 1
lr = 1e-1

res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/10-v8-2-tenx-gene140_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}'
print(res_path)

# load data
trn_dataloader = data_loader_scrna_tinydata.load_Allen_data(
    datasetkey='tenx_trn_v3', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=64,
)
tst_dataloader = data_loader_scrna_tinydata.load_Allen_data(
    datasetkey='tenx_tst_v3', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=500,
)

# gsubidx = torch.tensor(np.arange(n_gns)).to(device) # selected genes for recon
f = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/', 'rna', 'gidx_sub140_tenx_v1.pt')
gsubidx = torch.load(f)

# train
train_model(trn_dataloader, tst_dataloader, gsubidx, res_path, lmd0, 
            n_bit=n_bit, n_rcn_layers=n_rcn, n_epochs=n_epochs, n_iter=n_iter, lr=lr, 
            path_trained_model='',
            disable_tqdm=disable_tqdm,
            )
