import numpy as np
import torch
import model_celltype
    
# torchgpu env
n_iter = 3000 
lmd1_range= 5e-9, 1e-12
min_pos_range= 1.25e5, 2.5e5
lmd0 = 1e-10
lmd1 = 1e-10
min_pos = 2e5
drprts = np.linspace(0, 1, 11)[:-1]
print(drprts)

for drprt in drprts:
    res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/05-1_drprt{drprt}'
    print(f"GPU: {torch.cuda.is_available()}")
    model_celltype.train_model(res_path, lmd0, lmd1, min_pos, n_iter=n_iter, drprt=drprt)