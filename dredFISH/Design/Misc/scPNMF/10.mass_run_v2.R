suppressPackageStartupMessages(library(scPNMF))
suppressPackageStartupMessages(library(anndata))
library(reticulate)
np <- import("numpy")

# example
ddir <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/"
ddirout <- "/bigstore/GeneralStorage/fangming/projects/dredfish/res_dpnmf/v2/"
name0 <- "smrt" 
k <- 24 
mu_range <- c(1e4, 1e2, 1, 1e-2, 1e-4)
namex_range <- c("all")
namey_range <- c("L5") # same length as namex_range

# namex_range <- c(
#     "subL3n10", "subL3n100", "subL3n1000", 
#     "subL5n10", "subL5n100", "subL5n1000",
#     "all", "all"
#     )
# namey_range <- c(
#     "L3", "L3", "L3",
#     "L5", "L5", "L5",
#     "L3", "L5"
#     )

for (i in 1:length(namex_range)){ 
    namex <- namex_range[i]
    namey <- namey_range[i]
    print(namex)
    print(namey)

    for (j in 1:length(mu_range)){
        mu <- mu_range[j]
        print(mu)

        in_X <- paste(ddir, name0, "_X_", namex, ".npy", sep = "") 
        in_y <- paste(ddir, name0, "_X_", namex, "_y_", namey, ".npy", sep = "") 
        ot_w <- paste(ddirout, name0, "_X_", namex, "_y_", namey, 
                        "_dpnmfW_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")
        ot_s <- paste(ddirout, name0, "_X_", namex, "_y_", namey, 
                        "_dpnmfS_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")
        ot_wsel <- paste(ddirout, name0, "_X_", namex, "_y_", namey, 
                        "_dpnmfWsel_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")
        ot_ssel <- paste(ddirout, name0, "_X_", namex, "_y_", namey, 
                        "_dpnmfSsel_", "k", toString(k), "_mu", toString(mu), ".csv", sep = "")
        print(ot_w)
        print(ot_s)

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
                    method =  "DPNMF", # "EucDist", # "DPNMF"
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

        # select good basis (test of corr vs library size; multimodal test)
        W_sel <- basisSelect(W = W, S = S,
                                X = X, toTest = TRUE, toAnnotate = FALSE, mc.cores = 1)
        S_sel <- S[,colnames(W_sel)]

        write.csv(W, file=ot_w)
        write.csv(S, file=ot_s)
        write.csv(W_sel, file=ot_wsel)
        write.csv(S_sel, file=ot_ssel)
    }
}
