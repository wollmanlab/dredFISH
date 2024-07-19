from dredFISH.Analysis.TissueGraph import *
from dredFISH.Analysis.Classification import *


def feature_harmonization(adata,ref_adata,max_iterations=4,plot=False,N=2,sample_frequency=1/10,resolution=2):
    if not 'classification_space' in ref_adata.layers.keys():
        ref_adata.layers['classification_space'] = basicu.robust_zscore(basicu.normalize_fishdata_robust_regression(np.array(ref_adata.X)))
    
    ref_adata.obs['reference'] = True
    adata.obs['reference'] = False
    merge_adata = anndata.concat([adata, ref_adata])

    merge_adata = recursive_unsupervised_classification(merge_adata,max_iterations=max_iterations,plot=False,N=N,sample_frequency=sample_frequency,resolution=resolution)
    merge_adata.X = merge_adata.layers['classification_space']

    adata = merge_adata[merge_adata.obs['reference'] == False]
    ref_adata = merge_adata[merge_adata.obs['reference']]

    """ Harmonize the features """
    cell_type_label = f"unsupervised_L{max_iterations-1}"
    unique_labels = np.unique(merge_adata.obs[cell_type_label])

    X = torch.tensor(adata.X,dtype=torch.float32)
    ref_X = torch.tensor(ref_adata.X,dtype=torch.float32)
    y = torch.tensor(adata.obs[cell_type_label].values,dtype=torch.float32)
    ref_y = torch.tensor(ref_adata.obs[cell_type_label].values,dtype=torch.float32)
    z = torch.tensor(adata.obs['Slice_id'].values,dtype=torch.float32)
    harmonized_X = torch.zeros_like(X)

    for label in tqdm(unique_labels,desc='Harmonizing features'):
        m = ref_y == label
        if torch.sum(m)==0:
            continue
        idx = torch.where(ref_y == label)[0]
        label_ref_X = ref_X[idx]
        # ref_parameters = torch.quantile(label_ref_X, torch.tensor([0.05, 0.95]), dim=0)
        # ref_parameters[1,ref_parameters[0]==ref_parameters[1]] = ref_parameters[0,ref_parameters[0]==ref_parameters[1]]+1
        # label_ref_X_median = torch.median(ref_X,axis=0).values
        # label_ref_X_norm = label_ref_X - label_ref_X_median
        # label_ref_X_std = torch.median(torch.abs(label_ref_X_norm), axis=0).values
        # if torch.sum(label_ref_X_std==0)>0:
        #     label_ref_X_std[label_ref_X_std==0] = torch.std(label_ref_X_norm[label_ref_X_std==0], axis=0)
        # label_ref_X_std[label_ref_X_std==0] = 1
        for section in torch.unique(z):
            m = (y == label)&(z==section)
            if torch.sum(m)==0:
                continue
            idx = torch.where(m)[0]
            label_X = X[idx]
            # parameters = torch.quantile(label_X, torch.tensor([0.05, 0.95]), dim=0)
            # parameters[1,parameters[0]==ref_parameters[1]] = parameters[0,parameters[0]==parameters[1]]+1
            # label_X_norm = label_X-parameters[0]
            # label_X_norm = label_X_norm/(parameters[1]-parameters[0])
            # label_X_norm = label_X_norm*(ref_parameters[1]-ref_parameters[0])
            # label_X_norm = label_X_norm+ref_parameters[0]

            # # label_X_norm = label_X - torch.median(label_X,axis=0).values
            # # label_X_std = torch.median(torch.abs(label_X_norm),axis=0).values
            # # if torch.sum(label_X_std==0)>0:
            # #     label_X_std[label_X_std==0] = torch.std(label_X_norm[label_X_std==0],axis=0)
            # # label_X_std[label_X_std==0] = 1
            # # label_X_norm = label_X_norm / label_X_std

            # # label_X_norm = label_X_norm * label_ref_X_std
            # # label_X_norm = label_X_norm + label_ref_X_median
            # harmonized_X[idx] = label_X_norm

            harmonized_X[idx] = torch.tensor(basicu.quantile_matching(label_ref_X.numpy(),label_X.numpy()),dtype=torch.float32)
            
    return harmonized_X.numpy(),adata,ref_adata






