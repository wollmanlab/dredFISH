library(scPNMF)

mu_val <- 0
outpth <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
outpth_w <- paste(outpth, "mu_", toString(mu_val), 
                "_dpnmf_test_w.csv", sep = "") 
outpth_s <- paste(outpth, "mu_", toString(mu_val), 
                "_dpnmf_test_s.csv", sep = "")

data(zheng4)
X <- SummarizedExperiment::assay(zheng4, "logcounts")
res <- PNMFfun(X = X, 
            K = 3, 
            method = "EucDist", # "DPNMF"
            seed = 0,
            )
w <- res$Weight
s <- res$Score

write.csv(w, file=outpth_w)
write.csv(s, file=outpth_s)