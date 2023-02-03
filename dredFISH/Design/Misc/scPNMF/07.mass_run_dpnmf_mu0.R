suppressPackageStartupMessages(library(scPNMF))
suppressPackageStartupMessages(library(anndata))
library(reticulate)
np <- import("numpy")

# example
ddir <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
name0 <- "smrt" 
k <- 24 
mu <- 0

namex_range <- c("subL3n10", "subL3n100", "subL3n1000", 
                 "subL5n10", "subL5n100", "subL5n1000",
                 "all", "all"
                )
namey_range <- c("L3", "L3", "L3",
                 "L5", "L5", "L5",
                 "L3", "L5"
                )
for (i in 1:length(namex_range)){ 
    namex <- namex_range[i]
    namey <- namey_range[i]
    print(namex)
    print(namey)

    in_X <- paste(ddir, name0, "_X_", namex, ".npy", sep = "") 
    in_y <- paste(ddir, name0, "_X_", namex, "_y_", namey, ".npy", sep = "") 
    ot_w <- paste(ddir, name0, "_X_", namex, "_y_", namey, 
                    "_dpnmfW_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")
    ot_s <- paste(ddir, name0, "_X_", namex, "_y_", namey, 
                    "_dpnmfS_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")

    X <- np$load(in_X, allow_pickle = TRUE)
    labels <- np$load(in_y, allow_pickle = TRUE)

    m <- dim(X)[1]
    n <- dim(X)[2]
    rownames(X) <- 1:m
    colnames(X) <- 1:n

    print(length(labels))
    print(dim(X))
    print("Running DPNMF...")
    res <- PNMFfun(X = X, 
                K = k, 
                method = "DPNMF", # "EucDist", # "DPNMF"
                label = labels,
                mu = mu,
                lambda = 0.01,
                verboseN = TRUE,
                maxIter=2000,
                tol=1e-4,
                seed = 0,
                )
    W <- res$Weight
    S <- res$Score

    write.csv(W, file=ot_w)
    write.csv(S, file=ot_s)
}
