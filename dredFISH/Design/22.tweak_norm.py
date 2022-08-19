import numpy as np
import torch
import os
from dredFISH.Design.model_v2p1_norm import train_model
from dredFISH.Design.data_loader_scrna import load_Allen_data
import sys
import glob

disable_tqdm = False # True # False
print(f"disable tqdm (clean log): {disable_tqdm}")
print(f"GPU: {torch.cuda.is_available()}")

studybatch = "types_v22_Aug19"
    
n_epochs = 3
n_iter = None #3 

lmd0 = 1 
n_bits = [12, 24] # [6,12,24]
n_rcn = 0 # 1, 2, 3, 0 
lr = 1e-2
scale = 1e2
noise = None #(1e1, 1)
# scale = 1.5e5
# noise = (1e5, 1e4)

# load data
trn_dataloader = load_Allen_data(
    datasetkey='smrt_trn', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=64,
)
tst_dataloader = load_Allen_data(
    datasetkey='smrt_tst', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=500,
)

# gsubidx = torch.tensor(np.arange(n_gns)).to(device) # selected genes for recon
f = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/', 'rna', 'gidx_sub140_smrt_v1.pt')
gsubidx = torch.load(f)

# train
for n_bit in n_bits:
    n_bit = n_bit
    res_path = f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/{studybatch}_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}'
    print(res_path)
    train_model(trn_dataloader, tst_dataloader, gsubidx, res_path, lmd0, 
                n_bit=n_bit, n_rcn_layers=n_rcn, 
                scale=scale, noise=noise,
                n_epochs=n_epochs, n_iter=n_iter, lr=lr, 
                # path_trained_model=path_trained_model,
                disable_tqdm=disable_tqdm,
                )
