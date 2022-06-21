suppressPackageStartupMessages(library(scPNMF))
suppressPackageStartupMessages(library(anndata))
library(reticulate)
np <- import("numpy")

# example
ddir <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
name <- "test"
mu <- 0
k <- 2

in_X <- paste(ddir, name, "_X.npy", sep = "") 
in_y <- paste(ddir, name, "_y.npy", sep = "") 
ot_w <- paste(ddir, name, "_W", "_mu", toString(mu), "_k", toString(k), ".csv", sep = "")
ot_s <- paste(ddir, name, "_S", "_mu", toString(mu), "_k", toString(k), ".csv", sep = "")

X <- np$load(in_X)
labels <- np$load(in_y, allow_pickle = TRUE)

m <- dim(X)[1]
n <- dim(X)[2]
rownames(X) <- 1:m
colnames(X) <- 1:n

res <- PNMFfun(X = X, 
            K = k, 
            method = "EucDist", # "DPNMF"
            verboseN = TRUE,
            maxIter=1000,
            tol=1e-4,
            seed = 0,
            )
W <- res$Weight
S <- res$Score

write.csv(W, file=ot_w)
write.csv(S, file=ot_s)