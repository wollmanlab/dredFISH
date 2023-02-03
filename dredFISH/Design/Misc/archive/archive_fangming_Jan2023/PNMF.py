# Projective Non-negative Matrix Factorization; 
# Implemented by Fangming Xie following the scPNMF (Song et al. 2021), and orignally Yang and Oja, 2010.
# the original scPNMF is an R package wrapper of an underlying cpp code.

# from distutils.ccompiler import new_compiler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging

# set up 
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

def initialize(X, k, init='uniform'):
    """
    Args:
        - X: a p by n non-negative matrix (2d numpy array)
             Note that it is the transpose of (n,p)
        - k: number of dimensions
    Output:
        - w: the weight matrix (p, k) 
    """
    m, n = X.shape
    # initialize
    if init == 'pca':
        # use the abs(PCA) -- suppose to be U (or the V of X.T) to initialize
        pca = PCA(n_components=k)
        pca.fit(X.T)
        vt = pca.components_
        w = np.abs(vt.T) # [:,:k] # redundant
    elif init == 'pca_2x':
        # use the positive and negative parts of the PCAs
        kh_p = int((k+1)/2) # half (+1)

        pca = PCA(n_components=kh_p)
        pca.fit(X.T)
        vt = pca.components_
        wp = np.clip( vt.T, 0, None)
        wn = np.clip(-vt.T, 0, None)
        w = np.hstack([wp, wn])
        w = w[:,
              np.vstack([np.arange(kh_p), kh_p+np.arange(kh_p)]).T.reshape(-1,)] # p and n one by one
        w = w[:,:k]
        # print(wp.shape, wn.shape, w.shape)

    elif init == 'normal':
        w = np.abs(np.random.randn(m, k)) 
    elif init == 'uniform':
        w = np.random.rand(m, k) 
    else:
        raise ValueError("not implemented")

    w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value) (very useful in practice)
    return w

def get_PNMF(X, k, 
            init='pca', 
            random_seed=0, tol=1e-5, max_iter=1000, 
            zero_tol=1e-10, verbose=False, 
            report_stride=1,
            report_target_error=False, # slow
            ):
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
    m, n = X.shape
    k = int(k)
    assert k <= min(m, n) # rank limit

    # initialize
    w = initialize(X, k, init=init)

    # norm w (very useful in practice)
    w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value)
    xxt = X.dot(X.T)
    record = []
    error = 1
    i = 0
    # iterate
    while error > tol and i < max_iter:
        # record last w
        wlast = w

        # prep
        a = xxt.dot(w)
        wwt = w.dot(w.T)
        wtw = w.T.dot(w)
        denom = wwt.dot(a)+a.dot(wtw)
        denom = np.clip(denom, zero_tol, None)
        ratio = 2*a/denom

        # update w (multiplication rule)
        w = w*ratio
        w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value) (very useful in practice)

        # compute error
        error = np.linalg.norm(w-wlast, 'fro')**2
        # record and report
        if i % report_stride == 0:
            if verbose:
                logging.info(f"{i}, {error:.2e}")
            if report_target_error:
                target_error = np.linalg.norm(X-w.dot(w.T.dot(X)), 'fro')**2
                record.append((i, error, target_error))
            else:
                record.append((i, error,))
                
        # count up
        i += 1

    return w, np.array(record)

def get_DPNMF(X, k, s, mu,
            init='pca', 
            random_seed=0, tol=1e-5, max_iter=1000, 
            zero_tol=1e-10, verbose=False, 
            report_stride=1,
            report_target_error=False, # slow
            ):
    """
    Args:
        - X: a p by n non-negative matrix (2d numpy array)
             Note that it is the transpose of (n,p)
        - k: number of dimensions
        - s: a p by p matrix
        - mu: strength of the D term (S) 
    Output:
        - w: the weight matrix (p, k) with ||w||_2 = 1
        - record: recorded the error function every xxx time (m,2)

    ===
    optimize 1/2||X-WW^tX||_F^2 + 1/2
    update with 
        w = w*ratio
        w = w/||w||_2
        where ratio = (2 XXt W)/(WWt XXt W + XXt WWt W)

    """
    np.random.seed(random_seed)
    assert np.any(X >= 0)
    m, n = X.shape
    k = int(k)
    assert k <= min(m, n) # rank limit
    assert (m, m) == s.shape

    # initialize
    w = initialize(X, k, init=init)

    sp = np.clip( s, 0, None)
    sn = np.clip(-s, 0, None)

    # norm w (very useful in practice)
    w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value)
    xxt = X.dot(X.T)
    record = []
    error = 1
    i = 0
    # iterate
    while error > tol and i < max_iter:
        # record last w
        wlast = w

        # prep
        a = xxt.dot(w)
        num = 2*a + mu*sn.dot(w)

        wwt = w.dot(w.T)
        wtw = w.T.dot(w)
        denom = wwt.dot(a) + a.dot(wtw) + mu*sp.dot(w)
        denom = np.clip(denom, zero_tol, None)

        ratio = num/denom

        # update w (multiplication rule)
        w = w*ratio
        w = w/np.linalg.norm(w, ord=2) # 2-norm (largest singular value) (very useful in practice)

        # compute error
        error = np.linalg.norm(w-wlast, 'fro')**2
        # record and report
        if i % report_stride == 0:
            if verbose:
                logging.info(f"{i}, {error:.2e}")
            if report_target_error:
                target_error = np.linalg.norm(X-w.dot(w.T.dot(X)), 'fro')**2
                record.append((i, error, target_error))
            else:
                record.append((i, error,))
                
        # count up
        i += 1

    return w, np.array(record)