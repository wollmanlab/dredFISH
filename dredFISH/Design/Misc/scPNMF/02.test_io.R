suppressPackageStartupMessages(library(scPNMF))
suppressPackageStartupMessages(library(anndata))
suppressPackageStartupMessages(library(rhdf5))
library(reticulate)

# inpth <- '/bigstore/binfo/mouse/Brain/Sequencing/Allen_SmartSeq_CtxHippo/source/expression_matrix.hdf5'

# x = load(file = "/bigstore/binfo/mouse/Brain/Sequencing/Allen_SmartSeq_CtxHippo/source/Seurat.ss.rda")


# cells <- h5read("/bigstore/binfo/mouse/Brain/Sequencing/Allen_SmartSeq_CtxHippo/source/expression_matrix.hdf5","sample_names")
# read hdf5
# read xxx 
# read xxx

np <- import("numpy")
mat <- np$load(".npy")

# random example
M1 <- matrix(rnorm(36),nrow=6)
m <- dim(M1)[1]
n <- dim(M1)[2]
rownames(M1) <- 1:m
colnames(M1) <- 1:n

# data example
# data(zheng4)
# X <- SummarizedExperiment::assay(zheng4, "logcounts")

# mu_val <- 0
# inpth <- "/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna/scrna_ss_ctxhippo_a_exon_count_matrix_v2.h5ad"
# ad <- read_h5ad(inpth)
# labels <- ad$obs$subclass_label

# outpth <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
# outpth_w <- paste(outpth, "mu_", toString(mu_val), "_dpnmf_test_w.csv", sep = "")
# outpth_s <- paste(outpth, "mu_", toString(mu_val), "_dpnmf_test_s.csv", sep = "")

res <- PNMFfun(X = M1, 
            K = 2, 
            method = "EucDist", # "DPNMF"
            verboseN = TRUE,
            maxIter=1000,
            tol=1e-4,
            seed = 0,
            )
W <- res$Weight
S <- res$Score

# write.csv(W, file=outpth_w)
# write.csv(S, file=outpth_s)