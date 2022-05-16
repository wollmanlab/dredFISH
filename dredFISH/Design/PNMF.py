# Projective Non-negative Matrix Factorization; 
# Implemented by Fangming Xie following the scPNMF (Song et al. 2021), and orignally Yang and Oja, 2010.
# the original scPNMF is an R package wrapper of an underlying cpp code.

from distutils.ccompiler import new_compiler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_PNMF(X, k, random_seed=0, tol=1e-5, max_iter=1000, zero_tol=1e-10, verbose=False, report_stride=1):
    """
    Args:
        - X: a p by n non-negative matrix (2d numpy array)
             Note that it is the transpose of (n,p)
        - k: number of dimensions
    Output:
        - w: the weight matrix (p, k) with ||w||_2 = 1
        - record: recorded the error function every xxx time (m,2)

    ===
    optimize ||X-WW^tX||_F^2
    update with 
        w = w*ratio
        w = w/||w||_2
        where ratio = (2 XXt W)/(WWt XXt W + XXt WWt W)

    """
    np.random.seed(random_seed)
    assert np.any(X >= 0)
    k = int(k)
    assert k <= np.min(X.shape) # rank limit

    # use the abs(PCA) -- suppose to be U (or the V of X.T) to initialize
    pca = PCA(n_components=k)
    pca.fit(X.T)
    vt = pca.components_

    # initialize
    w = np.abs(vt.T) # [:,:k] # redundant
    xxt = X.dot(X.T)
    record = []
    error = 1
    i = 0
    # iterate
    while error > tol and i < max_iter:
        # record last w
        wlast = w

        # update w (multiplication rule)
        a = xxt.dot(w)
        wwt = w.dot(w.T)
        wtw = w.T.dot(w)
        denom = wwt.dot(a)+a.dot(wtw)
        denom = np.clip(denom, zero_tol, None)
        ratio = 2*a/denom
        w = w*ratio

        # norm w (very useful in practice)
        w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value)
        
        # compute error
        error = np.linalg.norm(w-wlast, 'fro')**2
        i += 1

        # record and report
        if i % report_stride == 0:
            record.append((i, error))
            if verbose:
                print(i, error)
 
    return w, np.array(record)