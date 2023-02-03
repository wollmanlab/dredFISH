"""
"""
from tokenize import group
from typing import Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
import logging

def reset_logging(**kwargs):
    """reset logging.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # logging.basicConfig(level=logging.WARNING)
    logging.basicConfig(**kwargs)
    return

def rank_array(array):
    """Return ranking of each element of an array
    """
    array = np.array(array)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def rank_rows(matrix):
    """Return rankings of each rwo in a 2d array
    """
    matrix = np.array(matrix)
    return np.apply_along_axis(rank_array, 1, matrix) # row = 1

def spearman_corrcoef(X, Y):
    """return spearman correlation matrix for each pair of rows of X and Y
    """
    return np.corrcoef(rank_rows(X), rank_rows(Y))

def spearmanr_paired_rows(X, Y):
    """with p-value, slow
    """
    X = np.array(X)
    Y = np.array(Y)
    corrs = []
    ps = []
    for x, y in zip(X, Y):
        r, p = stats.spearmanr(x, y)
        corrs.append(r)
    return np.array(corrs), np.array(ps)

def corr_paired_rows_fast(X, Y, offset=0, mode='pearsonr'):
    """low memory compared to fast2
    """
    if mode == 'pearsonr':
        X = np.array(X)
        Y = np.array(Y)
    elif mode == 'spearmanr':
        # rank
        X = rank_rows(X)
        Y = rank_rows(Y)
    # zscore
    X = (X - X.mean(axis=1).reshape(-1,1))/(X.std(axis=1).reshape(-1,1)+offset)
    Y = (Y - Y.mean(axis=1).reshape(-1,1))/(Y.std(axis=1).reshape(-1,1)+offset)
    # corrs
    corrs = (X*Y).mean(axis=1)
    return corrs

def corr_paired_rows_fast2(X, Y, offset=0, nan_to_num=False, mode='pearsonr'):
    """
    """
    if mode == 'pearsonr':
        X = np.array(X)
        Y = np.array(Y)
    elif mode == 'spearmanr':
        # rank
        X = rank_rows(X)
        Y = rank_rows(Y)
    # zscore
    xz = stats.zscore(X, axis=1, nan_policy='propagate', ddof=0)
    yz = stats.zscore(Y, axis=1, nan_policy='propagate', ddof=0)
    xy_cc = np.nanmean(xz*yz, axis=1)

    if nan_to_num:
        xy_cc = np.nan_to_num(xy_cc) # turn np.nan into zero
    return xy_cc

def get_index_from_array(arr, inqs, na_rep=-1):
    """Get index of array
    """
    arr = np.array(arr)
    arr = pd.Series(arr).reset_index().set_index(0)
    idxs = arr.reindex(inqs)['index'].fillna(na_rep).astype(int).values
    return idxs

def diag_matrix(X, rows=np.array([]), cols=np.array([]), threshold=None):
    """Diagonalize a matrix as much as possible
    threshold controls the level of diagnalization
    a smaller threshold encourges more number of strict diagnal values,
    while discourages less number of free columns (quasi-diagnal)
    """
    di, dj = X.shape
    transposed = 0
    
    # enforce nrows <= ncols
    if di > dj:
        di, dj = dj, di
        X = X.T.copy()
        rows, cols = cols.copy(), rows.copy()
        transposed = 1
        
    # start (di <= dj)
    new_X = np.nan_to_num(X.copy(), 0)
    new_rows = rows.copy() 
    new_cols = cols.copy() 
    if new_rows.size == 0:
        new_rows = np.arange(di)
    if new_cols.size == 0:
        new_cols = np.arange(dj)
        
    # bring the greatest values in the lower right matrix to diagnal position 
    for idx in range(min(di, dj)):

        T = new_X[idx: , idx: ]
        i, j = np.unravel_index(T.argmax(), T.shape) # get the coords of the max element of T
        
        if threshold and T[i, j] < threshold:
            dm = idx # new_X[:dm, :dm] is done (0, 1, ..., dm-1) excluding dm
            break
        else:
            dm = idx+1 # new_X[:dm, :dm] will be done

        # swap row idx, idx+i
        tmp = new_X[idx, :].copy()
        new_X[idx, :] = new_X[idx+i, :].copy() 
        new_X[idx+i, :] = tmp 
        
        tmp = new_rows[idx]
        new_rows[idx] = new_rows[idx+i]
        new_rows[idx+i] = tmp

        # swap col idx, idx+j
        tmp = new_X[:, idx].copy()
        new_X[:, idx] = new_X[:, idx+j].copy() 
        new_X[:, idx+j] = tmp 
        
        tmp = new_cols[idx]
        new_cols[idx] = new_cols[idx+j]
        new_cols[idx+j] = tmp
        
    # 
    if dm == dj:
        pass
    elif dm < dj: # free columns

        col_dict = {}
        sorted_col_idx = np.arange(dm)
        free_col_idx = np.arange(dm, dj)
        linked_rowcol_idx = new_X[:, dm:].argmax(axis=0)
        
        for col in sorted_col_idx:
            col_dict[col] = [col]
        for col, key in zip(free_col_idx, linked_rowcol_idx): 
            if key in col_dict.keys():
                col_dict[key] = col_dict[key] + [col]
            else:
                col_dict[key] = [col]
                
            
        new_col_order = np.hstack([col_dict[key] for key in sorted(col_dict.keys())])
        
        # update new_X new_cols
        new_X = new_X[:, new_col_order].copy()
        new_cols = new_cols[new_col_order]
    else:
        raise ValueError("Unexpected situation: dm > dj")
    
    if transposed:
        new_X = new_X.T
        new_rows, new_cols = new_cols, new_rows
    return new_X, new_rows, new_cols 

def diag_matrix_rows(X):
    """Diagonalize a matrix as much as possible by only rearrange rows
    """
    di, dj = X.shape
    
    new_X = np.nan_to_num(np.array(X.copy()), 0)
    new_rows = np.arange(di) 
    new_cols = np.arange(dj) 
    
    # free to move rows
    row_dict = {}
    free_row_idx = np.arange(di)
    linked_rowcol_idx = new_X.argmax(axis=1) # the column with max value for each row
    
    for row, key in zip(free_row_idx, linked_rowcol_idx): 
        if key in row_dict.keys():
            row_dict[key] = row_dict[key] + [row]
        else:
            row_dict[key] = [row]
            
    new_row_order = np.hstack([row_dict[key] for key in sorted(row_dict.keys())])
    # update new_X new_cols
    new_X = new_X[new_row_order, :].copy()
    new_rows = new_rows[new_row_order]
    
    return new_X, new_rows, new_cols 

def diag_matrix_cols(X):
    """
    """
    new_X, new_rows, new_cols = diag_matrix_rows(X.T)
    # flip back
    new_X = new_X.T
    new_cols, new_rows = new_rows, new_cols
    return new_X, new_rows, new_cols 

def encode_mat(a, d, entrysize=1):
    """ Given a matrix and a dictionary; map elements of a according to d
    a - numpy array
    d - dictionary
    """
    u,inv = np.unique(a,return_inverse = True)
    if entrysize == 1:
        b = np.array([d[x] for x in u])[inv].reshape(a.shape)
        return b
    elif entrysize > 1:
        theshape = np.hstack([a.shape, entrysize])
        b = np.array([d[x] for x in u])[inv].reshape(theshape)
        return b

def group_sum(mat, groups, group_order=[]):
    """
    mat is a matrix (cell-by-feature) ; group are the labels (for each cell).
    
    this can be speed up!!! take advantage of the cluster label structure... check my metacell analysis script as well
    """
    m, n = mat.shape
    assert m == len(groups)
    if len(group_order) == 0:
        group_order = np.unique(groups)
    
    group_idx = get_index_from_array(group_order, groups)
    groupmat = sparse.csc_matrix(([1]*m, (group_idx, np.arange(m)))) # group by cell
    
    return groupmat.dot(mat), group_order

def group_mean(mat, groups, group_order=[], expand=False, clip_groupsize=False):
    """
    mat is a matrix (cell-by-feature) ; group are the labels (for each cell).

    len(group_order) determines the number of clusters; will infer from `mat` if empty.
    """
    n, p = mat.shape
    assert n == len(groups)
    if len(group_order) == 0:
        group_order = np.unique(groups)
    k = len(group_order)
    
    group_idx = get_index_from_array(group_order, groups) # get index from `group_order` for each entry in `group` 
    groupmat = sparse.csc_matrix(([1]*n, (group_idx, np.arange(n))), shape=(k,n)) # group by cell
    groupsize = np.sum(groupmat, axis=1)
    if clip_groupsize:
        groupsize = np.clip(groupsize, 1, None) # avoid 0 
    groupmat_norm = groupmat/groupsize  # row
    
    if not expand:
        return np.asarray(groupmat_norm.dot(mat)), group_order # (k,p)
    else:
        return np.asarray(groupmat.T.dot(groupmat_norm.dot(mat))) # (n,p) recover the cells by coping clusters

def libsize_norm(mat, scale=None):
    """cell by gene matrix, norm to median library size
    assume the matrix is in sparse format, the output will keep sparse
    """
    lib_size = mat.sum(axis=1)
    if scale is None:
        factor = np.median(lib_size)
    else:
        factor = scale # most often 1e6

    matnorm = (mat/lib_size.reshape(-1,1))*factor
    return matnorm

def sparse_libsize_norm(mat):
    """cell by gene matrix, norm to median library size
    assume the matrix is in sparse format, the output will keep sparse
    """
    lib_size = np.ravel(mat.sum(axis=1))
    lib_size_median = np.median(lib_size)

    lib_size_inv = sparse.diags(lib_size_median/lib_size)
    matnorm = lib_size_inv.dot(mat)
    return matnorm

def zscore(v, allow_nan=False, ignore_zero=False, zero_threshold=1e-10, **kwargs):
    """
    v is an numpy array (any dimensional)

    **kwargs are arguments of np.mean and np.std, such as


    axis=0 # zscore across rows for each column (if v is 2-dimensional)
    axis=1 # zscore across cols for each row  (if v is 2-dimensional)
    """
    if allow_nan:
        vcopy = v.copy()
        if ignore_zero:
            vcopy[vcopy<zero_threshold] = np.nan # turn a number to nan (usually 0)
        return (vcopy-np.nanmean(vcopy, **kwargs))/(np.nanstd(vcopy, **kwargs))
    else:
        return (v-np.mean(v, **kwargs))/(np.std(v, **kwargs))

def stratified_sample(df, col, n: Union[int, dict], return_idx=False, group_keys=False, sort=False, random_state=0, **kwargs):
    """
    n (int) represents the number for each group
    n (dict) can be used to sample different numbers for each group
    does not allow oversampling
    """
    if isinstance(n, int):
        dfsub = df.groupby(col, group_keys=group_keys, sort=sort, **kwargs).apply(
            lambda x: x.sample(n=min(len(x), n), 
                random_state=random_state)
            )
    elif isinstance(n, dict):
        dfsub = df.groupby(col, group_keys=group_keys, sort=sort, **kwargs).apply(
            lambda x: x.sample(n=min(len(x), n[x[col].iloc[0]]), 
                random_state=random_state)
            )

    if not return_idx:
        return dfsub
    else:
        idx = get_index_from_array(df.index.values, dfsub.index.values)
        return dfsub, idx

def stratified_sample_withrep(df, col, n: Union[int, dict], return_idx=False, group_keys=False, sort=False, random_state=0, **kwargs):
    """
    n (int) represents the number for each group
    n (dict) can be used to sample different numbers for each group
    replace=True: allow oversampling
    """
    if isinstance(n, int):
        dfsub = df.groupby(col, group_keys=group_keys, sort=sort, **kwargs).apply(
            lambda x: x.sample(n=n, 
                replace=True, random_state=random_state)
            )
    elif isinstance(n, dict):
        dfsub = df.groupby(col, group_keys=group_keys, sort=sort, **kwargs).apply(
            lambda x: x.sample(n=n[x[col].iloc[0]], 
                replace=True, random_state=random_state)
            )

    if not return_idx:
        return dfsub
    else:
        idx = get_index_from_array(df.index.values, dfsub.index.values)
        return dfsub, idx

def clip_by_percentile(vector, low_p=5, high_p=95):
    """
    """
    low_val = np.percentile(vector, low_p)
    high_val = np.percentile(vector, high_p)
    vector_clip = np.clip(vector, low_val, high_val)
    return vector_clip

def rank(arr, **kwargs):
    """rank is equivalent to argsort twice
    """
    arr = np.array(arr)
    return np.argsort(np.argsort(arr, **kwargs), **kwargs)

def normalize_fishdata(X, norm_cell=True, norm_basis=True, allow_nan=False):
    """
    X -- cell by basis raw count matrix
    
    0 clipping; cell normalization; bit normalization
    """
    # clip at 0
    X = np.clip(X, 0, None)
    # total counts per cell
    bitssum = X.sum(axis=1)
    logging.info(f"{bitssum.shape[0]} cells, minimum counts = {bitssum.min()}")

    # normalize by cell 
    if norm_cell:
        X = X/np.clip(bitssum.reshape(-1,1), 1e-10, None)

    # further normalize by bit
    if norm_basis:
        X = zscore(X, axis=0, allow_nan=allow_nan) # 0 - across rows (cells) for each col (bit) 

    return X 

def normalize_fishdata_logrowmedian(X, norm_basis=True, allow_nan=False):
    """
    X -- cell by basis raw count matrix
    
    0 clipping; cell normalization; bit normalization
    """
    # clip at 0
    X = np.clip(X, 0, None)

    # feature as the log counts removed by median per cell
    X = np.log10(X+1)
    X = X - np.median(X, axis=1).reshape(-1,1)

    # further normalize by bit
    if norm_basis:
        X = zscore(X, axis=0, allow_nan=allow_nan) # 0 - across rows (cells) for each col (bit) 

    return X

def swap_mask(mat, lookup_o2n):
    """create from the old mask matrix a new matrix with the swapped labels according to the lookup table (pd.Series) 
    lookup_o2n = pd.Series(lbl, index=unq)
    newmat = swap_mask(mat, lookup_o2n)
    """
    i, j = np.nonzero(mat)
    vec = mat[i,j]
    unq, inv = np.unique(vec, return_inverse=True)
    # assert np.all(unq[inv] == vec) # unq[inv] should recreates vec
    
    newmat = mat.copy()
    newmat[i,j] = lookup_o2n.loc[unq].values[inv]
    return newmat
