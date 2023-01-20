import os

alln_ddir = "/greendata/GeneralStorage/fangming/projects/dredfish/data/rna"
DATABASE = {
    # smart-seq data
    'smrt':        os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v4.zarr'),
    'smrt_trn':    os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v4_train.zarr'),
    'smrt_tst':    os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v4_test.zarr'),
    # smart-seq data: a subset of genes that are predictable (by PCAs; kNNs)
    'smrt_sub140_geneidx'   : os.path.join(alln_ddir, 'gidx_sub140_smrt_v1.pt'), 
    # smart-seq data gene constraints (pshop)
    'smrt_pshopcnst_geneidx': os.path.join(alln_ddir, 'gidx_cnstrnts_pshop_mm10_isoflat.pt'),
    'smrt_pshopcnst'        : os.path.join(alln_ddir,      'cnstrnts_pshop_mm10_isoflat.pt'),

    # 10x data
    'tenx':        os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2.zarr'),
    'tenx_trn':    os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2_train.zarr'),
    'tenx_tst':    os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2_test.zarr'),

    # 10x data - a different version
    'tenx_v3':     os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3.zarr'),
    'tenx_trn_v3': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3_train.zarr'),
    'tenx_tst_v3': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3_test.zarr'),
}