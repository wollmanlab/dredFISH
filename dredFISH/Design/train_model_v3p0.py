
import os
import torch
from data_loaders.data_files import DATABASE as db
from data_loaders.data_loader_scrna import load_Allen_data
from models.model_v3p0_basic import train_model

# set up
studybatch = f"v3p0_basic_2023Jan20"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU if exists
# device = 'cpu'
# device = None

disable_tqdm = False # True # False
print(f"disable tqdm (clean log): {disable_tqdm}")
print(f"GPU: {torch.cuda.is_available()}")
print(f"Use: {device}")

n_epochs = 3
n_iter = None #3 
libsize_norm = True
lmd0 = 0 # recon loss  
lmd1 = 2 # bit constraint
lmd2 = 1 # gene constraints
lmd3 = 0 #0.1 #.01 # sparsity constraint
drprt = 0 #1.0/24 #.01 #0.05 # dropout applied to the projected matrix (cell by basis), i.e. randomly remove an entire bit for a cell
n_rcn = 0 # 1, 2, 3, 0 
lr = 1e-2
noise = None #(1e1, 1) (1e5, 1e4)
n_bit = 24
min_sgnl, scale, max_sgnl = (1.0/2)*1e5, 1e5, 10*1e5

res_path = ('/greendata/GeneralStorage/fangming/projects/dredfish/res_nn/'
            f'{studybatch}_scale{scale:.1e}_min{min_sgnl:.1e}_max{max_sgnl:.1e}_nbit{n_bit}_drprt{drprt:.1e}'
            f'_lmds_{lmd0:.1e}_{lmd1:.1e}_{lmd2:.1e}_{lmd3:.1e}'
            )
print(res_path)

# load data: train, test, and constraints
trn_dataloader = load_Allen_data(
    datasetkey='smrt_trn', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=64,
    database=db,
)
tst_dataloader = load_Allen_data(
    datasetkey='smrt_tst', 
    keyX='counts', keyY='l3_code', keyYcat='l3_cat', 
    batch_size=500,
    database=db,
)
gsubidx      = torch.load(db['smrt_sub140_geneidx'])
cnstrnts_idx = torch.load(db['smrt_pshopcnst_geneidx'])
cnstrnts     = torch.load(db['smrt_pshopcnst'])

# train the model
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
            libsize_norm=libsize_norm,
            disable_tqdm=disable_tqdm,
            device=device,
            # path_trained_model=path_trained_model,
            )
