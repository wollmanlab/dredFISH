import os

alln_ddir = "/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna"
ALLEN_DATADICT = {
    'smrt': os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v3.zarr'),
    'smrt_trn': os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v3_train.zarr'),
    'smrt_tst': os.path.join(alln_ddir, 'scrna_ss_ctxhippo_a_exon_count_matrix_v3_test.zarr'),

    'tenx': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2.zarr'),
    'tenx_trn': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2_train.zarr'),
    'tenx_tst': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_exon_count_matrix_v2_test.zarr'),

    'tenx_v3': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3.zarr'),
    'tenx_trn_v3': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3_train.zarr'),
    'tenx_tst_v3': os.path.join(alln_ddir, 'scrna_10x_ctxhippo_a_count_matrix_v3_test.zarr'),
}