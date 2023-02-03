import numpy as np
import pandas as pd
import anndata
import os
import logging

from dredFISH.Utils import basicu
from models import PNMF

### get the S matrix (weighted hierarchical)
def get_lowdim_covmat_tree_distance(levels, meta, Xt):
    """Xt is a cell by gene matrix here
    """
    # get a tree
    # L5 - L3 - L1 - L0
    # levels = [
    #     'cluster_label', 
    #     'subclass_label', 
    #     'class_label',
    #     ]
    tree = meta.groupby(levels).size() 
    tree = tree[tree!=0]
    tree = tree.reset_index()[levels]

    # centroids - per level
    ctrds_lvl = []
    types_lvl = []
    # sub levels
    for level in levels:
        logging.info(level)
        # get centroids
        ctrds_, types_ = basicu.group_mean(Xt, meta[level].values)
        logging.info(f"{ctrds_.shape}")
        logging.info(f"{types_.shape}")
        ctrds_lvl.append(ctrds_)
        types_lvl.append(types_)
    # level 0 (global)
    ctrds_, types_ = basicu.group_mean(Xt, ['']*len(meta))
    ctrds_lvl.append(ctrds_)
    types_lvl.append(types_)

    # centroid diff and Sb
    ngenes = ctrds_lvl[0].shape[1]
    Sb = np.zeros((ngenes, ngenes))
    l2maxmean = 0 # for ease of scaling
    for i in range(len(levels)):
        a = ctrds_lvl[i+1]
        b = ctrds_lvl[i]
        
        _types = types_lvl[i]
        _lc = levels[i]
        if i+1 < len(levels):
            _lu = levels[i+1]
            types_map1up = meta.groupby(_lc)[_lu].first().loc[_types].values
            types_map1up_code, _types_u = pd.factorize(types_map1up, sort=True)
            assert np.all(_types_u == types_lvl[i+1])
            
            ctrds_diff = b - a[types_map1up_code]
        else:
            ctrds_diff = b - np.repeat(a, len(b), axis=0)
        
        # save ctrds_diff for each level
        logging.info(f'{i}, {ctrds_diff.shape}')
        
        # norm
        l2 = np.linalg.norm(ctrds_diff, axis=1) # no squared
        l2maxmean = max(np.mean(l2), l2maxmean) # max (across all levels) of the mean l2 per level
        # print(l2)
        
        ctrds_diff_norm = ctrds_diff/np.clip(l2.reshape(-1,1), 1e-5, None) # norm per cluster
        # l2n = np.linalg.norm(ctrds_diff_norm, axis=1) # l2n should be 1
        # print(l2n)
        
        Sb = Sb + w_lvl[i]*ctrds_diff_norm.T.dot(ctrds_diff_norm)
    Sb = (l2maxmean**2)*Sb # restore unit/scale
    ### end -- calc Sb
    return Sb

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

# specify input (CPM with rep) and output (factorized weight matrix) 
input_X_path  = '/greendata/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100.npy'
output_w_path_format = '/greendata/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/output_demo/TreeDPNMF_smrt_mu{}.npy' # left out the parameter mu
logging.info(f"{input_X_path} -> \n" + f"{output_w_path_format}")
cell_path     = '/greendata/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v_python/inputs/smrt_withrep_X_subL5n100_cells.csv'
meta_path     = '/greendata/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_genes_matrix.h5ad'

# DPNMF parameters
select_first_few_genes = 1000 # or 0 for not including a restriction
k = 24
mus = [0, 1e2, 1e4, 1e6] # strength of the Dterm
levels = [
    'cluster_label', 
    'subclass_label', 
    'class_label',
    ]
w_lvl = [1, 1, 1] #
w_lvl = w_lvl/np.mean(w_lvl)
logging.info(f"{w_lvl}")

# load data - allen scrna matrix (CPM; only 10k genes)
logging.info('load data...')
X = np.load(input_X_path, allow_pickle=True) # this is gene by cell; # .T.copy() # cell by gene
logging.info(f'data shape: {X.shape}')
# select first few genes if used 
if select_first_few_genes > 0:
    X = X[:select_first_few_genes,:]
logging.info(f'data shape (updated): {X.shape}')

cells = pd.read_csv(cell_path, index_col=0, header=0)['0'] #.values 
meta = anndata.read_h5ad(meta_path, backed='r').obs
meta = meta.loc[cells]

# get the S matrix
Sb = get_lowdim_covmat_tree_distance(levels, meta, X.T) # this assumes cell by gene

# run and save results
S = -Sb # S = Sw-Sb
outdir = os.path.dirname(output_w_path_format)
if not os.path.isdir(outdir):
    os.mkdir(outdir)

for mu in mus:
    logging.info(f'run Tree-DPNMF {mu:.1e}')

    w, rec = PNMF.get_DPNMF(X, k, S, mu, init='normal', verbose=True, report_stride=30)
    output_w_path = output_w_path_format.format(f"{mu:.1e}")
    np.save(output_w_path, w)
    logging.info(f'Done! Saved results to: {output_w_path}')
