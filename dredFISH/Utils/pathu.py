import os

datasets = {
    'allen_smrt_dpnmf': '/greendata/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_matrix.h5ad',
    'allen_tenx_dpnmf': '/greendata/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_10x_ctxhippo_a_DPNMF_matrix.h5ad',
    'allen_anat_tree': '/greendata/GeneralStorage/fangming/reference/allen_ccf/structures.json',
}

def get_path(key, datasets=datasets, check=False):
    """
    Retrieve path from a key; or the key itself 
    """
    if key in datasets.keys():
        path = datasets[key]
    else:
        path = key
    
    if check:
        assert os.path.isfile(path)
    return path