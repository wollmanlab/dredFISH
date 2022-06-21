# library(reticulate)
# library(QuickPNMFs)
# library(data.table)
# library(RANN)
# library(mvtnorm)
# library(pryr)
# library(rjson)
# library(ggplot2)
# library(progress)

datapth <- "/bigstore/binfo/mouse/Brain/Sequencing/Allen_10X_2020/dredfish_filtered/DPNMF_L3_1000_balanced_Glutamatergic/" # nolint
outpth <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"

np <- import("numpy")
counts <- np$load(paste(datapth, "normalized_matrix.npy", sep = ""))
annotations <- fread(paste(datapth, "metadata.csv", sep = ""))
cell_labels <- annotations$Level_3_subclass_label

# genes <- np$load(paste(datapth, "genes.npy", sep = ""))
# cells <- np$load(paste(datapth, "cells.npy", sep = ""))
gc()

labels <- cell_labels
counts_balanced <- counts

mu_val <- 50
dpnmf <- PNMFfun(t(counts_balanced), rank=24,method="DPNMF", label=labels, mu=mu_val)
loadings <- t(dpnmf$basis)
np$save(paste(outpth,"mu_",toString(mu_val),"_dpnmf_test.npy",sep = ""),loadings)