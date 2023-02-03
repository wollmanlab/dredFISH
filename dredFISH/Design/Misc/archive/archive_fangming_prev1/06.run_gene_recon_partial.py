import numpy as np
import torch
import dredFISH.Design.model_gene_recon_partial import train_model

print(f"GPU: {torch.cuda.is_available()}")
    
# torchgpu env
# lmd1_range= 5e-9, 1e-12
# min_pos_range= 1.25e5, 2.5e5
n_iter = 3000 
lmd1 = 1e-10
min_pos = 2e5

# lmd0 = 1e-10
lmd0s = np.logspace(-5, 5, 6)
print(lmd0s)
for lmd0 in lmd0s:
    print(lmd0)
    res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/06-gene140_lmd0{lmd0:.2e}'
    print(res_path)
    train_model(res_path, lmd0, lmd1, min_pos, n_iter=n_iter)