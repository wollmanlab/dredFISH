import numpy as np
import torch
from dredFISH.Design import model_gene_recon_partial_simple_smrt 
from dredFISH.Design import model_gene_recon_partial_simple_tenx

print(f"GPU: {torch.cuda.is_available()}")
    
# torchgpu env
# lmd1_range= 5e-9, 1e-12
# min_pos_range= 1.25e5, 2.5e5
n_iter = 5000 
lmd1 = 1e-10
min_pos = 2e5

# # lmd0 = 1e-10
# lmd0s = np.logspace(-5, 5, 3)
# print(lmd0s)
# for lmd0 in lmd0s:
lmd0 = 1e5
# n_bits = [50, 70, 100]
n_bit = 70
# n_rcns = [1, 2, 3]
n_rcn = 1
lr_s = [1e-1, 1e-2, 1e-3]
lr = 1e-1

res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/06-v8-smrt-simple-gene140_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}'
print(res_path)
model_gene_recon_partial_simple_smrt.train_model(res_path, lmd0, lmd1, min_pos, 
            n_bit=n_bit, n_rcn_layers=n_rcn, n_iter=n_iter, lr=lr)

res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/06-v8-tenx-simple-gene140_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}'
print(res_path)
model_gene_recon_partial_simple_tenx.train_model(res_path, lmd0, lmd1, min_pos, 
            n_bit=n_bit, n_rcn_layers=n_rcn, n_iter=n_iter, lr=lr)