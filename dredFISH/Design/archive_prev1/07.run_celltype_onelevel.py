import numpy as np
import torch
import model_celltype_onelevel

print(f"GPU: {torch.cuda.is_available()}")
    
# torchgpu env
# lmd1_range= 5e-9, 1e-12
# min_pos_range= 1.25e5, 2.5e5
n_iter = 3000 
lmd1 = 1e-10
min_pos = 2e5
lmd0 = 1e-10

dropouts = np.linspace(0, 1, 11)[:-1]
print(dropouts)

levels = ['fine', 'crse']
for level in levels:
    for drprt in dropouts: 
        res_path= f'/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/07-v2-{level}-drprt{drprt:.1f}'
        print(res_path)
        tenx_label= f'tenx_{level}'
        smrt_label= f'smrt_{level}'
        model_celltype_onelevel.train_model(res_path, lmd1, min_pos, tenx_label, smrt_label, n_iter=n_iter, drprt=drprt)