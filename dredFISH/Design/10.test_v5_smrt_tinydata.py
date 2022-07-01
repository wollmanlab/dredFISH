import numpy as np
import torch
import os
from dredFISH.Design.model_v1p3_gene_celltype_tuneinit import train_model
from Design import data_loader_scrna
import sys
import glob

disable_tqdm = False # True # False
print(f"disable tqdm (clean log): {disable_tqdm}")
print(f"GPU: {torch.cuda.is_available()}")
    
n_epochs = 1000 
n_iter = None #3 

lmd0 = 1 
n_bit = 70
n_rcn = 1
lr = 1e-2
scale=1.5e4
noise=None # (1e4, 1e3)

tinydata = 100 # don't do this!!!! 

res_path =      f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/10-v10-5-smrt-tinydata-gene140_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}'
# path_trained_model = glob.glob(os.path.join(res_path_prev, 'model=*'))[0]
# print(path_trained_model)
print(res_path)

# load data
trn_dataloader = data_loader_scrna.load_Allen_data(
    datasetkey='smrt_trn', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=30,
    tinydata=tinydata,
)
tst_dataloader = data_loader_scrna.load_Allen_data(
    datasetkey='smrt_tst', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=50,
)

# gsubidx = torch.tensor(np.arange(n_gns)).to(device) # selected genes for recon
f = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/', 'rna', 'gidx_sub140_smrt_v1.pt')
gsubidx = torch.load(f)

# train
train_model(trn_dataloader, tst_dataloader, gsubidx, res_path, lmd0, 
            n_bit=n_bit, n_rcn_layers=n_rcn, 
            scale=scale, noise=noise,
            n_epochs=n_epochs, n_iter=n_iter, lr=lr, 
            # path_trained_model=path_trained_model,
            disable_tqdm=disable_tqdm,
            )
