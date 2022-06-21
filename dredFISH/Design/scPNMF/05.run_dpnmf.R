suppressPackageStartupMessages(library(scPNMF))
suppressPackageStartupMessages(library(anndata))
library(reticulate)
np <- import("numpy")

# example
ddir <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
name0 <- "smrt"
namex <- "subL3n10"
namey <- "L3"
mu <- 1
k <- 24

in_X <- paste(ddir, name0, "_X_", namex, ".npy", sep = "") 
in_y <- paste(ddir, name0, "_X_", namex, "_y_", namey, ".npy", sep = "") 
ot_w <- paste(ddir, name0, "_X_", namex, "_y_", namey, 
                "_dpnmfW_", "mu", toString(mu), "_k", toString(k), ".csv", sep = "")
ot_s <- paste(ddir, name0, "_X_", namex, "_y_", namey, 
                "_dpnmfS_", "mu", toString(mu), "_k", toString(k), ".csv", sep = "")

X <- np$load(in_X)
labels <- np$load(in_y, allow_pickle = TRUE)

# X <- X[1:100,] # first 100 genes -- speed up
print(length(labels))
print(dim(X))

m <- dim(X)[1]
n <- dim(X)[2]
rownames(X) <- 1:m
colnames(X) <- 1:n

print("Running DPNMF...")
res <- PNMFfun(X = X, 
            K = k, 
            method = "DPNMF", # "EucDist", # "DPNMF"
            label = labels,
            mu = mu,
            lambda = 0.01,
            verboseN = TRUE,
            maxIter=1000,
            tol=1e-4,
            seed = 0,
            )
W <- res$Weight
S <- res$Score

write.csv(W, file=ot_w)
write.csv(S, file=ot_s)