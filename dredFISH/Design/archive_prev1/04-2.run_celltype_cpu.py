import torch
import model_celltype
    
# routine env
res_path= '/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/04-2_cpu'
n_iter = 3000 
lmd1_range= 5e-9, 1e-12
min_pos_range= 1.25e5, 2.5e5
lmd0 = 1e-10
lmd1 = 1e-10
min_pos = 2e5
drprt = 0
print(f"GPU: {torch.cuda.is_available()}")
model_celltype.train_model(res_path, lmd0, lmd1, min_pos, n_iter=n_iter, drprt=drprt)