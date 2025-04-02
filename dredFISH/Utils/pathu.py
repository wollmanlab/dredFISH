import os

datasets = {
    'allen_smrt_dpnmf': '/greendata/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_DPNMF_matrix.h5ad',
    'allen_merfish': '/scratchdata1/ExternalData/Allen_WMB_2023Sep05/merfish.h5ad',
    'allen_tenx_dpnmf': '/greendata/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_10x_ctxhippo_a_DPNMF_matrix.h5ad',
    'allen_anat_tree': '/greendata/GeneralStorage/fangming/reference/allen_ccf/structures.json',
    'allen_wmb_tree': '/scratchdata1/ExternalData/Allen_WMB_2024Mar06/projected_Tree_reorder_2024Mar06/10X_combined.h5ad',
    'allen_wmb_tree_supercell': '/scratchdata1/ExternalData/Allen_WMB_2024Mar06/projected_Tree_reorder_2024Mar06/10X_combined_supercells.h5ad',
    'allen_smrt_tree':'/greendata/binfo/mouse/Brain/Sequencing/Allen_SmartSeq_2020/projected_tree_2023Sep08.h5ad',
    'spatial_reference':'/scratchdata1/ExternalData/Allen_WMB_2024Mar06/spatial_data.h5ad'
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