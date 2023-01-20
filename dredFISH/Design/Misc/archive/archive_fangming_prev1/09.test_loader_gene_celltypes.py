import numpy as np
import torch
from dredFISH.Design.model_v1_gene_celltype import train_model

print(f"GPU: {torch.cuda.is_available()}")
    
# torchgpu env
# lmd1_range= 5e-9, 1e-12
# min_pos_range= 1.25e5, 2.5e5
lmd1 = 1e-10
min_pos = 2e5
n_epochs = 3 
n_iter = None 

# # lmd0 = 1e-10
# lmd0s = np.logspace(-5, 5, 3)
# print(lmd0s)
# for lmd0 in lmd0s:
lmd0 = 1 # 1e5
# n_bits = [50, 70, 100]
n_bit = 70
# n_rcns = [1, 2, 3]
n_rcn = 1
lr = 1e-1
path_trained_model = ('/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/'
    '09-v3-gene-celltype_lmd01.00e+00_nbit70_nrcn1_lr0.1/model=xxx-xxx-90000.0-2.00E+05-70-0-1.00E-10-0.01-1.0.pt'
)
res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/09-v3-gene-celltype_lmd0{lmd0:.2e}_nbit{n_bit}_nrcn{n_rcn}_lr{lr:.1g}-lag2'
print(res_path)
train_model(res_path, lmd0, lmd1, min_pos, 
            n_bit=n_bit, n_rcn_layers=n_rcn, n_epochs=n_epochs, n_iter=n_iter, lr=lr, 
            path_trained_model=path_trained_model,
            )
