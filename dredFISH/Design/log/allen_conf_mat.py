import numpy as np
import pandas as pd 
import os

path_dict= dict(root_path= '/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference',
                                prbe_constraints_path= './10X/probe_constraints.npy',
                                tenx_cells_path= './10X/cells.npy',
                                tenx_genes_path= './10X/genes.npy',
                                tenx_metadata_path= './10X/metadata.csv',
                                tenx_counts_path= './10X/matrix.npy',
                                smrt_lengths_path= './SmartSeq/gene_length.npy',
                                smrt_genes_path='./SmartSeq/genes.npy',
                                smrt_cells_path= './SmartSeq/cells.npy',
                                smrt_metadata_path= './SmartSeq/metadata.csv',
                                smrt_counts_path= './SmartSeq/matrix.npy')

tenx_cells= np.load(os.path.join(path_dict['root_path'], path_dict['tenx_cells_path']))
tenx_metadata= pd.read_csv(os.path.join(path_dict['root_path'], path_dict['tenx_metadata_path']), index_col=0, low_memory=False).loc[tenx_cells]

smrt_cells= np.load(os.path.join(path_dict['root_path'], path_dict['smrt_cells_path']))
smrt_metadata= pd.read_csv(os.path.join(path_dict['root_path'], path_dict['smrt_metadata_path']), index_col=0, low_memory=False).loc[smrt_cells]

tenx_12= tenx_metadata[['Level_1_class_label','Level_2_neighborhood_label']].values
tenx_23= tenx_metadata[['Level_2_neighborhood_label','Level_3_subclass_label']].values
tenx_34= tenx_metadata[['Level_3_subclass_label','Level_4_supertype_label']].values
tenx_45= tenx_metadata[['Level_4_supertype_label','Level_5_cluster_label']].values

lv1_dict= {i:idx for idx,i in enumerate(set(tenx_metadata['Level_1_class_label'].values))}
lv2_dict= {i:idx for idx,i in enumerate(set(tenx_metadata['Level_2_neighborhood_label'].values))}
lv3_dict= {i:idx for idx,i in enumerate(set(tenx_metadata['Level_3_subclass_label'].values))}
lv4_dict= {i:idx for idx,i in enumerate(set(tenx_metadata['Level_4_supertype_label'].values))}
lv5_dict= {i:idx for idx,i in enumerate(set(tenx_metadata['Level_5_cluster_label'].values))}

from sklearn.metrics import confusion_matrix

cm_12= confusion_matrix([lv1_dict[x] for x in tenx_12[:,0]], [lv2_dict[x] for x in tenx_12[:,1]])[:len(lv1_dict)]
cm_23= confusion_matrix([lv2_dict[x] for x in tenx_23[:,0]], [lv3_dict[x] for x in tenx_23[:,1]])[:len(lv2_dict)]
cm_34= confusion_matrix([lv3_dict[x] for x in tenx_34[:,0]], [lv4_dict[x] for x in tenx_34[:,1]])[:len(lv3_dict)]
cm_45= confusion_matrix([lv4_dict[x] for x in tenx_45[:,0]], [lv5_dict[x] for x in tenx_45[:,1]])[:len(lv4_dict)]

import seaborn as sns
import matplotlib.pyplot as plt 
import math

def fmt(x, pos):
    return r'$2^{{{}}}$'.format(int(x))

def plot_confusion_matrix(conf_mat,xdict,ydict,filename,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(16,9))
    hm=sns.heatmap(np.log2(conf_mat+math.pow(2,-5)), cmap='Reds')
    ax.set_xticklabels([x for idx,x in enumerate(ydict) if idx in (ax.get_xticks()-0.5).astype(np.int32)], rotation=45)
    ax.set_yticklabels([x for idx,x in enumerate(xdict) if idx in (ax.get_yticks()-0.5).astype(np.int32)], rotation=0)
    ax.set_xlabel(ylabel,fontsize=16)
    ax.set_ylabel(xlabel,fontsize=16)
    cbar = hm.collections[0].colorbar
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(list(map(lambda x: fmt(x,_), cbar.get_ticks())))
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

plot_confusion_matrix(cm_12,lv1_dict,lv2_dict,
    'tmp1.png','Level_1_class_label','Level_2_neighborhood_label')

plot_confusion_matrix(cm_23,lv2_dict,lv3_dict,
    'tmp2.png','Level_2_neighborhood_label','Level_3_subclass_label')

plot_confusion_matrix(cm_34,lv3_dict,lv4_dict,
    'tmp3.png','Level_3_subclass_label','Level_4_supertype_label')

plot_confusion_matrix(cm_45,lv4_dict,lv5_dict,
    'tmp4.png','Level_4_supertype_label','Level_5_cluster_label')
