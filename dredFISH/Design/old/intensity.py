import numpy as np
import json
import pandas as pd
import os 
import torch
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

reference_path = '/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference/'
model_path = '/home/jperrie/Documents/neural_network_probe_set/'

# first we find the bit intensity for DPNMF if we haven't done it already
# we may need to restart if memory is filled after saving to file 
if not os.path.isfile('per_bit_dpnmf.npy'):
    mat = torch.tensor(np.load(os.path.join(reference_path, '10X_dpnmf/matrix.npy')).astype(np.float64))
    dpnmf_loading = torch.tensor(pd.read_csv(os.path.join(reference_path, '10X_dpnmf/weights.csv'), index_col = 0).values.astype(np.float64))
    dpnmf_projection = torch.mm(mat, dpnmf_loading)
    per_bit_dpnmf = dpnmf_projection.mean(0)
    np.save('per_bit_dpnmf', per_bit_dpnmf)

mat = torch.tensor(np.load(os.path.join(reference_path, '10X/matrix.npy')).astype(np.float64))

def find_init(directory):
    paths = glob(os.path.join(model_path, directory,'*result*'))
    
    max_min_acc = [
         np.min((np.mean(list(json.load(open(x, ))['2000']['smrt_fine_acc'].values())) * 
         np.mean(list(json.load(open(x, ))['2000']['smrt_crse_acc'].values())), 
         np.mean(list(json.load(open(x, ))['2000']['tenx_fine_acc'].values())) * 
         np.mean(list(json.load(open(x, ))['2000']['tenx_crse_acc'].values()))))       
         for x in paths]
    
    index = np.argmax(max_min_acc)
    return paths[index]


base_path="results/embmat" + find_init("results").split("results/result")[1]
base_nn_loading = torch.tensor(np.array(json.load(open(base_path))))

x5_path="results_5/embmat" + find_init("results_5").split("results_5/result")[1]
x5_nn_loading = torch.tensor(np.array(json.load(open(x5_path))))

x10_path="results_10/embmat" + find_init("results_10").split("results_10/result")[1]
x10_nn_loading = torch.tensor(np.array(json.load(open(x10_path))))

base_projection = torch.mm(mat, base_nn_loading)
x5_projection = torch.mm(mat, x5_nn_loading)
x10_projection = torch.mm(mat, x10_nn_loading)

per_bit_base = base_projection.mean(0)
per_bit_x5 = x5_projection.mean(0)
per_bit_x10 = x10_projection.mean(0)

np.save('per_bit_base', per_bit_base)
np.save('per_bit_x5', per_bit_x5)
np.save('per_bit_x10', per_bit_x10)

# reload everything 

per_bit_base = np.load('per_bit_base.npy')
per_bit_x5 = np.load('per_bit_x5.npy')
per_bit_x10 = np.load('per_bit_x10.npy')
per_bit_dpnmf = np.load('per_bit_dpnmf.npy')

# plotting

cmap = matplotlib.cm.get_cmap('Spectral')
plt.clf()
fig,ax=plt.subplots(figsize=(10,8))
plt.plot(per_bit_base, color=cmap(np.linspace(0.05,0.95,4)[0]), marker='o', markersize=12, label='x1', alpha=0.5, linestyle='')
plt.plot(per_bit_x5, color=cmap(np.linspace(0.05,0.95,4)[1]), marker='o', markersize=12, label='x5', alpha=0.5, linestyle='')
plt.plot(per_bit_x10, color=cmap(np.linspace(0.05,0.95,4)[2]), marker='o', markersize=12, label='x10', alpha=0.5, linestyle='')
plt.plot(3*per_bit_dpnmf, color=cmap(np.linspace(0.05,0.95,4)[3]), marker='o', markersize=12, label='DPNMF', alpha=0.5, linestyle='')
plt.legend(loc="upper right")
plt.xlabel("Bits")
plt.ylabel("Average # readout probes")
plt.tight_layout()
plt.savefig("nn.png")

# violin plots fine-grained accuracy 
def get_df(path):
    df_dict_tenx = json.load(open(path, ))['2000']['tenx_fine_acc']
    df_dict_tenx = pd.DataFrame.from_dict(df_dict_tenx, orient="index")
    df_dict_tenx.rename(columns={0: path.split("/")[0] + "_tenx"}, inplace=True)
    
    
    df_dict_sm2 = json.load(open(path, ))['2000']['smrt_fine_acc']
    df_dict_sm2 = pd.DataFrame.from_dict(df_dict_sm2, orient="index")
    df_dict_sm2.rename(columns={0: path.split("/")[0] + "_sm2"}, inplace=True)
    
    df = pd.concat([df_dict_tenx,df_dict_sm2],axis=1)
    return df

base_path = 'results/result' + base_path[(base_path.index("embmat")+len("embmat")):]
x5_path = 'results_5/result' + x5_path[(x5_path.index("embmat")+len("embmat")):]
x10_path = 'results_10/result' + x10_path[(x10_path.index("embmat")+len("embmat")):]

df = pd.concat([get_df(base_path),get_df(x5_path),get_df(x10_path)],axis=1)
avg_acc = df[1:].mean().to_dict()


plt.clf()
fig,ax=plt.subplots(figsize=(10,8))
sns.heatmap(df, cmap="YlGnBu",cbar_kws={'label': 'Accuracy'})
ax.set_yticks([])
ax.set_xticklabels(["x1|10X|"+str(np.around(avg_acc["results_tenx"],4)),
                    "x1|SM2|"+str(np.around(avg_acc["results_sm2"],4)),
                    "x5|10X|"+str(np.around(avg_acc["results_5_tenx"],4)),
                    "x5|SM2|"+str(np.around(avg_acc["results_5_sm2"],4)),
                    "x10|10X|"+str(np.around(avg_acc["results_10_tenx"],4)),
                    "x10|SM2|"+str(np.around(avg_acc["results_10_sm2"],4))])
plt.ylabel("Cell type")
plt.xlabel("NN model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("nn_acc.png")
