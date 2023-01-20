import numpy as np
import torch
import os
from dredFISH.Design.model_v2p6_bit_cnst import train_model
from dredFISH.Design.data_loader_scrna import load_Allen_data
import sys
import glob

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU if exists
# device = 'cpu'
device = None
disable_tqdm = False # True # False
print(f"disable tqdm (clean log): {disable_tqdm}")
print(f"GPU: {torch.cuda.is_available()}")
print(f"Use: {device}")

n_epochs = 3
n_iter = None #3 

libsize_norm = True
lmd0 = 0 # recon loss  
lmd2 = 1 # gene constraints
lmd3 = 0 #0.1 #.01 # sparsity constraint
drprt = 0 #1.0/24 #.01 #0.05

n_rcn = 0 # 1, 2, 3, 0 
lr = 1e-2
noise = None #(1e1, 1)
# noise = (1e5, 1e4)

studybatch = f"types_v28_Sep15_v6"

# n_bit = 24
# scale = 1e4
# n_bits = [24] 
# scales = [100, 1e3, 1e4, 1e5]

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

f = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/', 'rna', 'gidx_cnstrnts_pshop_mm10_isoflat.pt')
cnstrnts_idx = torch.load(f)
f = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/', 'rna',      'cnstrnts_pshop_mm10_isoflat.pt')
cnstrnts = torch.load(f)

# for n_bit in n_bits:
#     for scale in scales:
n_bit = 24
min_sgnl, scale, max_sgnl = (1.0/2)*1e5, 1e5, 10*1e5
lmd1 = 2 # bit constraint

res_path = f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/{studybatch}_scale{scale:.1e}_min{min_sgnl:.1e}_max{max_sgnl:.1e}_nbit{n_bit}_drprt{drprt:.1e}_lmds_{lmd0:.1e}_{lmd1:.1e}_{lmd2:.1e}_{lmd3:.1e}'
print(res_path)
train_model(trn_dataloader, tst_dataloader, 
            res_path, 
            gsubidx, 
            cnstrnts_idx,
            cnstrnts,
            lmd0, lmd1, lmd2, lmd3,
            n_bit=n_bit, n_rcn_layers=n_rcn, 
            drprt=drprt,
            scale=scale, 
            min_sgnl=min_sgnl,
            max_sgnl=max_sgnl,
            noise=noise,
            lr=lr, n_epochs=n_epochs, n_iter=n_iter, 
            # path_trained_model=path_trained_model,
            disable_tqdm=disable_tqdm,
            libsize_norm=libsize_norm,
            device=device,
            )
