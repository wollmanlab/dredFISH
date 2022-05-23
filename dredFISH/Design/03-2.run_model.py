import torch
import model_gene_recon
    
res_path= '/bigstore/GeneralStorage/fangming/projects/dredfish/res_nn/03-2_cpu'
n_iter = 10
lmd1_range= 5e-9, 1e-12
min_pos_range= 1.25e5, 2.5e5
lmd0 = 1e-10
lmd1 = 1e-10
min_pos = 2e5
print(f"GPU: {torch.cuda.is_available()}")
model_gene_recon.train_model(res_path, lmd0, lmd1, min_pos, n_iter=n_iter)