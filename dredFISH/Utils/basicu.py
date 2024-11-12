"""
"""
from tokenize import group
from typing import Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression,RANSACRegressor
import matplotlib.pyplot as plt
import logging
import torch

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
    if 'axis' in kwargs.items():
        if kwargs['axis'] ==1:
            v = v.T
    if allow_nan:
        vcopy = v.copy()
        if ignore_zero:
            vcopy[vcopy<zero_threshold] = np.nan # turn a number to nan (usually 0)
            for i in range(vcopy.shape[1]):
                c = vcopy[:,i]
                vmin,vmid,vmax = np.percentile(c[np.isnan(c)==False],[25,50,75])
                if vmax!=vmin:
                    vcopy[:,i] = (c-vmid)/(vmax-vmin)
                else:
                    # likely not enough observations or all are 1 value
                    vcopy[:,i] = 0
    else:
        vcopy = v.copy()
        for i in range(vcopy.shape[1]):
            c = vcopy[:,i]
            vmin,vmid,vmax = np.percentile(c[np.isnan(c)==False],[25,50,75])
            if vmax!=vmin:
                vcopy[:,i] = (c-vmid)/(vmax-vmin)
            else:
                # likely not enough observations or all are 1 value
                vcopy[:,i] = 0
    if 'axis' in kwargs.items():
        if kwargs['axis'] ==1:
            vcopy = vcopy.T
    return vcopy
    

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

def normalize_fishdata_to_mean(X, norm_cell=True, norm_basis=True, allow_nan=False):
    """
    X -- cell by basis raw count matrix
    
    0 clipping; cell normalization; bit normalization
    """
    # clip at 0
    X = np.clip(X, 0, None)
    # total counts per cell
    bitssum = X.sum(axis=1)
    bitssum_mean = bitssum.mean()
    logging.info(f"{bitssum.shape[0]} cells, minimum counts = {bitssum.min()}")

    # normalize by cell 
    if norm_cell:
        X = X/np.clip(bitssum.reshape(-1,1), 1e-10, None)

    # further normalize by bit
    if norm_basis:
        X = zscore(X, axis=0, allow_nan=allow_nan) # 0 - across rows (cells) for each col (bit) 

    return X*bitssum_mean

def normalize_fishdata_log_regress(X):
    """
    X -- cell by basis raw count matrix
    
    Advanced normalization using sum. Key idea is instead of divide by the sum
    regress in log space and use the residual. 
    """
    # clip at 1
    X = np.clip(X, 1, None).astype(float)
    log_X = np.log10(X)
    log_sm = np.log10(X.sum(axis=1))

    # init empty residual
    Res = np.zeros_like(X)

    # normalize by cell 
    for i in range(X.shape[1]):
        model = LinearRegression().fit(log_sm.reshape(-1, 1), log_X[:, i])
        Res[:,i] =  log_X[:,i] - model.predict(log_sm.reshape(-1, 1))

    # go back from log to linear scale
    X = 10**Res

    return X 

def normalize_fishdata_log_regress_to_mean(X):
    """
    X -- cell by basis raw count matrix
    
    Advanced normalization using sum. Key idea is instead of divide by the sum
    regress in log space and use the residual. 
    """
    # clip at 1
    X = np.clip(X, 1, None).astype(float)
    log_X = np.log10(X)
    log_sm = np.log10(X.sum(axis=1))

    # init empty residual
    Res = np.zeros_like(X)

    # normalize by cell 
    for i in range(X.shape[1]):
        model = LinearRegression().fit(log_sm.reshape(-1, 1), log_X[:, i])
        Res[:,i] =  (log_X[:,i] - model.predict(log_sm.reshape(-1, 1)))+model.predict(log_sm.reshape(-1, 1)).mean()

    # go back from log to linear scale
    X = 10**Res

    return X 

def normalize_fishdata_regress_to_mean(X):
    """
    X -- cell by basis raw count matrix
    
    Advanced normalization using sum. Key idea is instead of divide by the sum
    regress in log space and use the residual. 
    """
    # normalize by cell 
    newX = np.zeros_like(X)
    for i in range(X.shape[1]):
        model = LinearRegression().fit(X.sum(1).reshape(-1, 1), X[:, i])
        newX [:,i] =  (X[:,i] / model.predict(X.sum(1).reshape(-1, 1)))*model.predict(X.sum(1).reshape(-1, 1)).mean()

    return newX

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

def normalize_fishdata_log(X, norm_cell=True, norm_basis=True, allow_nan=False):
    """
    X -- cell by basis raw count matrix
    
    0 clipping; cell normalization; bit normalization
    """
    # clip at 0
    X = np.clip(X, 0, None)

    # feature as the log counts removed by median per cell
    X = np.log10(X+1)

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

def normalize_fishdata_robust_regression(X):
    """
    Regression of the "sum" out of each basis using a robust estimate of the sum 
    so that "high" cells in a few bits won't skew that sum estimate too mucn

    Approach works in following steps: 
    1. Use n-1 other basis to predict each basis
    2. Do a RANSAC regression for each bit vs the other estimate to identify "outlier" cells, i.e. cells whos response
        is more extreme than expected based on the other basis sum. 
    3. Replace the values for the outlier cells/basis with the predictions from other basis to calcualte the sum. 
    4. Divide by sum and adjust for scale. 
    """

    # step 1: cross prediction matrix P
    P = np.zeros_like(X)
    
    for target_col in range(X.shape[1]):
        # Prepare the features (X) and target (y) for the current target column
        F = np.delete(X, target_col, axis=1)
        y = X[:, target_col]
        linear_model = LinearRegression().fit(F,y)
        
        # Predict the target column using the trained model
        P[:, target_col] = linear_model.predict(F)

    # Step 2: fit RANSAC regression to find outliers
    inliners = np.zeros_like(X)
    common = P.mean(axis=1).reshape(-1,1)
    for i in range(X.shape[1]):
        f = X[:,i]
        # Step 1: Initial Linear Regression to estimate residuals
        init_reg = LinearRegression().fit(common,f)
        std_residuals = np.std(f - init_reg.predict(common))
        ransac = RANSACRegressor(LinearRegression(), 
                                residual_threshold = std_residuals, 
                                random_state=42)
        ransac.fit(common, f)
        inliners[:,i] = ransac.inlier_mask_

    # Step 3: replace outliers with cross-predictions
    Xrobust = X*inliners + P*(1-inliners)

    # Step 4: Normalize by dividing by sum and rescaling
    Nrm = X/Xrobust.mean(axis=1).reshape(-1,1)*Xrobust.mean()

    return Nrm

# def quantile_matching(M1, M2):
#     """
#     function gets matrices M1 and M2 and for each pairs of cols, calculate the qq-plot and uses that to interpolate values 
#     of M2 into M1 space. 
#     """ 
#     if not M1.shape[1]==M2.shape[1]: 
#         raise ValueError("Matrices used to match quantile must have the same number of cols")

#     interpolators = []
#     quantiles = np.linspace(0, 1, 1000)  # 0.01 resolution from 0 to 1
#     for i in range(M1.shape[1]):
#         quantiles_M1 = np.quantile(M1[~np.isnan(M1[:,i]), i], quantiles)
#         quantiles_M2 = np.quantile(M2[~np.isnan(M2[:,i]), i], quantiles)
#         interpolators.append(interp1d(quantiles_M2, quantiles_M1, fill_value="extrapolate"))

#     # Initialize an empty array for I_mms with the same number of rows as N_mms and 13 columns
#     I = np.zeros(M2.shape)

#     # Apply interpolators to each row in N_mms for the selected columns
#     for i, interp in enumerate(interpolators):
#         I[:, i] = interp(M2[:, i])

#     # Replace any nans with the median of the column
#     I = np.nan_to_num(I, nan=np.nanmedian(I, axis=0))

#     return I

# def quantile_matching(M1, M2):
#     """
#     Matches quantiles between matrices M1 and M2, minimizing NaNs.
#     """

#     if M1.shape[1] != M2.shape[1]:
#         raise ValueError("Matrices must have the same number of columns.")

#     # Prepare data: Replace NaNs with column-specific values (median or mean)
#     M1_filled = np.nan_to_num(M1, nan=np.nanmedian(M1, axis=0))
#     M2_filled = np.nan_to_num(M2, nan=np.nanmedian(M2, axis=0))
    
#     # Optionally consider scaling the data using StandardScaler before interpolating:
#     # from sklearn.preprocessing import StandardScaler
#     # scaler = StandardScaler()
#     # M1_scaled = scaler.fit_transform(M1_filled)
#     # M2_scaled = scaler.transform(M2_filled)
#     # M1_filled = M1_scaled
#     # M2_filled = M2_scaled

#     interpolators = []
#     quantiles = np.linspace(0, 1, np.min([M1.shape[1], M2.shape[1],1000]))
#     for i in range(M1.shape[1]):
#         quantiles_M1 = np.quantile(M1_filled[:, i], quantiles)
#         quantiles_M2 = np.quantile(M2_filled[:, i], quantiles)
        
#         quantiles_M1 += np.random.normal(0, 1e-8, quantiles_M1.shape) 
#         quantiles_M2 += np.random.normal(0, 1e-8, quantiles_M2.shape)

#         # Use linear interpolation for better NaN handling
#         interpolators.append(interp1d(quantiles_M2, quantiles_M1, kind='linear', 
#                                       bounds_error=False, fill_value="extrapolate"))
        
#         # Optionally clip extrapolated values:
#         # upper_bound = np.quantile(M1_filled[:, i], 0.99)
#         # lower_bound = np.quantile(M1_filled[:, i], 0.01)
#         # interp = lambda x: np.clip(interp(x), lower_bound, upper_bound)

#     I = np.zeros(M2.shape)
#     for i, interp in enumerate(interpolators):
#         interpolated = interp(M2_filled[:, i])
#         I[:, i] = np.nan_to_num(interpolated, nan=np.nanmedian(interpolated))

#     return I

def quantile_matching(M1, M2,n=100):
    """
    Matches quantiles between matrices M1 and M2, minimizing NaNs.
    """

    if M1.shape[1] != M2.shape[1]:
        raise ValueError("Matrices must have the same number of columns.")

    # Prepare data: Replace NaNs with column-specific values (median or mean)
    M1_filled = np.nan_to_num(M1, nan=np.nanmedian(M1, axis=0))
    M2_filled = np.nan_to_num(M2, nan=np.nanmedian(M2, axis=0))
    
    # Optionally consider scaling the data using StandardScaler before interpolating:
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # M1_scaled = scaler.fit_transform(M1_filled)
    # M2_scaled = scaler.transform(M2_filled)
    # M1_filled = M1_scaled
    # M2_filled = M2_scaled

    interpolators = []
    quantiles = np.linspace(0, 1, np.min([M1.shape[0], M2.shape[0],n]))
    for i in range(M1.shape[1]):
        quantiles_M1 = np.quantile(M1_filled[:, i], quantiles)
        quantiles_M2 = np.quantile(M2_filled[:, i], quantiles)
        
        # quantiles_M1 += np.random.normal(0, 1e-8, quantiles_M1.shape) 
        # quantiles_M2 += np.random.normal(0, 1e-8, quantiles_M2.shape)

        # Use linear interpolation for better NaN handling
        interpolators.append(interp1d(quantiles_M2, quantiles_M1, kind='linear', 
                                      bounds_error=False, fill_value="extrapolate"))
        
        # Optionally clip extrapolated values:
        # upper_bound = np.quantile(M1_filled[:, i], 0.99)
        # lower_bound = np.quantile(M1_filled[:, i], 0.01)
        # interp = lambda x: np.clip(interp(x), lower_bound, upper_bound)

    I = np.zeros(M2.shape)
    for i, interp in enumerate(interpolators):
        interpolated = interp(M2_filled[:, i])
        I[:, i] = np.nan_to_num(interpolated, nan=np.nanmedian(interpolated))

    return I

def swap_mask(mat, lookup_o2n):
    """create from the old mask matrix a new matrix with the swapped labels according to the lookup table (pd.Series) 
    lookup_o2n = pd.Series(lbl, index=unq)
    newmat = swap_mask(mat, lookup_o2n)
    """
    indices = np.nonzero(mat)
    vec = mat[indices]
    unq, inv = np.unique(vec, return_inverse=True)
    # assert np.all(unq[inv] == vec) # unq[inv] should recreates vec
    
    newmat = mat.copy()
    newmat[indices] = lookup_o2n.loc[unq].values[inv]
    return newmat

from sklearn.linear_model import LinearRegression
from tqdm import trange

def normalize_fishdata_regress(X,value='sum',leave_log=False,log=True,bitwise=False,labels=False,n_cells=100):
    """
    X -- cell by basis raw count matrix
    
    Advanced normalization using sum. Key idea is instead of divide by the sum
    regress in log space and use the residual. 
    """
    X = X.copy()
    if isinstance(labels,bool):
        BalancedX = X.copy()
    else:
        _df = pd.Series(labels).to_frame('label')
        idx = stratified_sample(_df, 'label', n_cells, random_state=None).sort_index().index.values
        BalancedX = X[idx,:].copy()
    if isinstance(value,str):
        if value == 'sum':
            Y = X.sum(axis=1)
        elif value =='mean':
            Y = X.mean(axis=1)
        elif value =='min':
            Y = X.min(axis=1)
        elif value =='max':
            Y = X.max(axis=1)
        elif value =='median':
            Y = np.median(X,axis=1)
        elif value == 'none':
            Y = ''
        else:
            raise ValueError('Incorrect value type '+str(value))
    else:
        Y = value

    if log:
        X = np.log10(np.clip(X,1,None))
        BalancedX = np.log10(np.clip(BalancedX,1,None))
        if not isinstance(Y,str):
            Y = np.log10(np.clip(Y,1,None))
    if not isinstance(Y,str):
        if isinstance(labels,bool):
            BalancedY = Y.copy()
        else:
            BalancedY = Y[idx].copy()

  

    # init empty residual
    Res = np.zeros_like(X)

    # normalize by cell 
    for i in range(X.shape[1]):
        if isinstance(Y,str):
            Res[:,i] =  X[:,i]
        else:
            model = LinearRegression().fit(BalancedY.reshape(-1, 1), BalancedX[:, i])
            if log:
                Res[:,i] =  (X[:,i] - model.predict(Y.reshape(-1, 1)))+model.predict(Y.reshape(-1, 1)).mean()
            else:
                Res[:,i] =  (X[:,i] / model.predict(X.sum(1).reshape(-1, 1)))*model.predict(X.sum(1).reshape(-1, 1)).mean()
    if log:
        if leave_log:
            X = Res
        else:
            X  = 10**Res
            BalancedX = 10**BalancedX
    else:
        if leave_log:
            X = np.log10(np.clip(X,1,None))
        else:
            X = Res

    if bitwise:
        if isinstance(labels,bool):
            X = zscore(X, axis=0, allow_nan=False,balancedX=False) # 0 - across rows (cells) for each col (bit)
        else:
            X = zscore(X, axis=0, allow_nan=False,balancedX=BalancedX) # 0 - across rows (cells) for each col (bit)
    return X

def process_batch(args):
    X_batch, Y_batch, BalancedX, BalancedY, log, leave_log = args
    Res_batch = np.zeros_like(X_batch)
    for i in range(X_batch.shape[1]):
        if isinstance(Y_batch, str):
            Res_batch[:, i] = X_batch[:, i]
        else:
            model = LinearRegression().fit(BalancedY.reshape(-1, 1), BalancedX[:, i])
            if log:
                Res_batch[:, i] = (X_batch[:, i] - model.predict(Y_batch.reshape(-1, 1))) + model.predict(Y_batch.reshape(-1, 1)).mean()
            else:
                Res_batch[:, i] = (X_batch[:, i] / model.predict(X_batch.sum(1).reshape(-1, 1))) * model.predict(X_batch.sum(1).reshape(-1, 1)).mean()
    if log and not leave_log:
        Res_batch = 10**Res_batch
    return Res_batch

from multiprocessing import Pool

def multiprocess_normalize_fishdata_regress(X, value='sum', leave_log=False, log=True, bitwise=False, labels=False, n_cells=100):
    """
    X -- cell by basis raw count matrix
    
    Advanced normalization using sum. Key idea is instead of divide by the sum
    regress in log space and use the residual. 
    """
    X = X.copy()
    if isinstance(labels, bool):
        BalancedX = X.copy()
    else:
        _df = pd.Series(labels).to_frame('label')
        idx = stratified_sample(_df, 'label', n_cells, random_state=None).sort_index().index.values
        BalancedX = X[idx, :].copy()
    if isinstance(value, str):
        if value == 'sum':
            Y = X.sum(axis=1)
        elif value == 'mean':
            Y = X.mean(axis=1)
        elif value == 'min':
            Y = X.min(axis=1)
        elif value == 'max':
            Y = X.max(axis=1)
        elif value == 'median':
            Y = np.median(X, axis=1)
        elif value == 'none':
            Y = ''
        else:
            raise ValueError('Incorrect value type ' + str(value))
    else:
        Y = value

    if log:
        X = np.log10(np.clip(X, 1, None))
        BalancedX = np.log10(np.clip(BalancedX, 1, None))
        if not isinstance(Y, str):
            Y = np.log10(np.clip(Y, 1, None))
    if not isinstance(Y, str):
        if isinstance(labels, bool):
            BalancedY = Y.copy()
        else:
            BalancedY = Y[idx].copy()

    # Split data into 10 batches
    n_batches = 10
    batch_size = X.shape[0] // n_batches
    batches = [(X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size], BalancedX, BalancedY, log, leave_log) for i in range(n_batches)]

    # Handle the last batch if the number of rows is not perfectly divisible
    if X.shape[0] % n_batches != 0:
        batches.append((X[n_batches*batch_size:], Y[n_batches*batch_size:], BalancedX, BalancedY, log, leave_log))

    # Use multiprocessing to process batches
    with Pool(n_batches) as pool:
        results = pool.map(process_batch, batches)

    # Combine results
    Res = np.vstack(results)

    if bitwise:
        if isinstance(labels, bool):
            X = zscore(Res, axis=0, allow_nan=False, balancedX=False)  # 0 - across rows (cells) for each col (bit)
        else:
            X = zscore(Res, axis=0, allow_nan=False, balancedX=BalancedX)  # 0 - across rows (cells) for each col (bit)
    else:
        X = Res

    return X
    

def zscore_matching(M1, M2):
    if M1.shape[1] != M2.shape[1]:
        raise ValueError("Matrices must have the same number of columns.")

    # print(np.median(M1, axis=0))
    # print(np.std(M1, axis=0))
    # print(np.median(M2, axis=0))
    # print(np.std(M2, axis=0))

    # Prepare data: Replace NaNs with column-specific values (median or mean)
    M1_filled = np.nan_to_num(M1, nan=np.nanmedian(M1, axis=0))
    M2_filled = np.nan_to_num(M2, nan=np.nanmedian(M2, axis=0))


    I = np.zeros(M2.shape)
    for i in range(M1.shape[1]):
        vmin,vmax = np.percentile(M1_filled[:,i][~np.isnan(M1_filled[:,i])],[5,95])
        ref_mask = (M1_filled[:,i]>vmin)&(M1_filled[:,i]<vmax)
        vmin,vmax = np.percentile(M2_filled[:,i][~np.isnan(M2_filled[:,i])],[5,95])
        measured_mask = (M2_filled[:,i]>vmin)&(M2_filled[:,i]<vmax)

        if (np.sum(ref_mask)>25) & (np.sum(measured_mask)>25):

            ref_mu = np.nanmedian(M1_filled[ref_mask, i])
            ref_sigma = np.nanstd(M1_filled[ref_mask, i])
            measured_mu = np.nanmedian(M2_filled[measured_mask, i])
            measured_sigma = np.nanstd(M2_filled[measured_mask, i])
            if ref_sigma == 0:
                ref_sigma = 1
            if measured_sigma == 0:
                measured_sigma = 1
            
            I[:, i] = (((M2_filled[:, i] - np.median(M2_filled[:, i])) / np.std(M2_filled[:, i])) * ref_sigma )+ ref_mu
        else:
            I[:, i] = np.nanmedian(M1_filled[:, i]) + (M2_filled[:, i] - np.nanmedian(M2_filled[:, i]))
    
    # Replace any nans with the median of the column
    I = np.nan_to_num(I, nan=np.nanmedian(I, axis=0))

    return I


def zscore(X,allow_nan=False, ignore_zero=False, zero_threshold=1e-10,balancedX=False, **kwargs):
    """
    X is an numpy array (any dimensional)

    **kwargs are arguments of np.mean and np.std, such as


    axis=0 # zscore across rows for each column (if v is 2-dimensional)
    axis=1 # zscore across cols for each row  (if v is 2-dimensional)
    """
    if isinstance(balancedX,bool):
        balancedX = X.copy()
    if 'axis' in kwargs.items():
        if kwargs['axis'] ==1:
            X = X.T
    mu = np.zeros(X.shape[1])
    std = np.zeros(X.shape[1])
    outX = X.copy()
    for i in range(X.shape[1]):
        c = balancedX[:,i].copy()
        if ignore_zero:
            c[c<zero_threshold] = np.nan 
        if not allow_nan:
            c=c[np.isnan(c)==False]
        mu[i] = np.median(c)
        std[i] = np.std(c)
        outX[:,i] = (X[:,i]-mu[i])/std[i]
    if 'axis' in kwargs.items():
        if kwargs['axis'] ==1:
            outX = outX.T
    return outX

def filter_cells_nuc(adata, thresh=10, vis=False, inplace=False):
    """
    This function masks cells that have too big of a change in nuclear stain across rounds.

    Parameters:
    adata: The AnnData object containing the data.
    thresh: The threshold for masking cells. Default is 10.
    vis: A boolean indicating whether to visualize the data. Default is True.
    inplace: A boolean indicating whether to modify the original AnnData object. Default is False.

    Returns:
    If inplace is True, returns the modified AnnData object.
    If inplace is False, returns the mask array.
    """
    
    # Copy the AnnData object
    # adata = adata.copy()
    
    # Get the 'nuc_raw' layer data
    nuc = np.array(adata.layers['nuc_raw']).copy()

    # Get the minimum value of nuc across the second axis
    nuc_min = np.min(nuc, axis=1)

    # scale each bit to same brightness
    nuc = np.median(np.median(nuc, axis=0, keepdims=True).ravel()) * nuc / np.median(nuc, axis=0, keepdims=True)

    # Keep Cells whose dapi is within 50% of median for all but at most 2 rounds
    mask = (nuc/np.clip(np.median(nuc,axis=1,keepdims=True),1,None))>0.5
    
    # # Scale the nuc data so that the median of each column is equal to the global median
    # nuc_scaled = nuc * np.median(np.median(nuc, axis=0, keepdims=True)) / np.median(nuc, axis=0, keepdims=True)
    
    # # Copy the scaled nuc data
    # zscored = nuc_scaled.copy()
    
    # # Subtract the median of each row from the corresponding row
    # zscored = zscored - np.median(zscored, axis=1, keepdims=True)
    
    # # Get the global standard deviation
    # std = np.median(np.median(np.abs(zscored), axis=1, keepdims=True))
    
    # # Normalize the zscored data
    # zscored = zscored / std
    
    # # Create a mask where the absolute value of zscored is greater than the threshold
    # mask = np.abs(zscored) > thresh
    
    # Set the mask to False where the minimum value of nuc is less than the threshold
    mask[nuc_min < thresh, :] = False
    
    # # If vis is True, visualize the data
    # if vis:
    #     vmin = -20
    #     vmax = 20
    #     plt.figure(figsize=(10, 10))
    #     for i in range(zscored.shape[1]):
    #         plt.hist(np.clip(zscored[:, i], vmin, vmax), bins=np.linspace(vmin, vmax, 100), label=i, alpha=0.1)
    #     plt.show()
    
    # If inplace is True, modify the original AnnData object
    if inplace:
        adata.layers['nuc_mask'] = mask
        return adata
    
    # If inplace is False, return the mask array
    else:
        return mask

def correct_linear_staining_patterns(X, XY, Section=None):
    """
    This function corrects for linear staining patterns in the data.

    Parameters:
    X: A 2D numpy array where each row is a cell and each column is a feature.
    XY: A 2D numpy array where each row is a cell and the columns are the x and y coordinates of the cell.
    Section: A 1D numpy array indicating the section each cell belongs to. If None, all cells are assumed to belong to the same section.

    Returns:
    out_X: A 2D numpy array with the same shape as X, but with corrected values.
    """
    
    # Initialize the output array with zeros
    out_X = np.zeros(X.shape, dtype=X.dtype)
    
    # If Section is None, assume all cells belong to the same section
    if isinstance(Section, type(None)):
        Section = np.ones(X.shape[0])
    
    # For each unique section
    for section in np.unique(Section):
        # Get a boolean mask where each cell belongs to the current section
        m = Section == section
        
        # Get the x and y coordinates of the cells in the current section
        xy = XY[m, :]
        
        # For each feature
        for i in range(X.shape[1]):
            # Get the values of the current feature for the cells in the current section
            x = np.array(X[m, i]).copy()
            vmin,vmax = np.percentile(x,[1,99])
            mask = (x>=vmin) & (x<=vmax)
            
            # Initialize a linear regression model
            model = LinearRegression()
            
            # Fit the model to the data
            model.fit(xy[mask,:], x[mask])
            
            # Get the predicted values of the model
            predicted_values = model.predict(xy)
            
            # Correct the values of the current feature for the cells in the current section
            out_X[m, i] = (x - predicted_values) + predicted_values.mean()
    
    # Return the corrected data
    return out_X

def image_coordinate_correction(X,XY):
    # Initialize a linear regression model
    model = LinearRegression()
    out_X = X.copy()

    for i in range(X.shape[1]):
        x = X[:,i]
        vmin,vmax = np.percentile(x,[1,99])
        mask = (x>=vmin) & (x<=vmax)
        # Fit the model to the data
        model.fit(XY[mask,:], x[mask])
        predicted_x = model.predict(XY)
        out_X[:,i] = (x-predicted_x) + predicted_x.mean()
    return out_X

def batch_bit_scaling(X, Section=None):
    """
    This function scales each bit (feature) in each batch (section) so that the median value is the same across batches.

    Parameters:
    X: A 2D numpy array where each row is a cell and each column is a feature.
    Section: A 1D numpy array indicating the section each cell belongs to. If None, all cells are assumed to belong to the same section.

    Returns:
    out_X: A 2D numpy array with the same shape as X, but with scaled values.
    """
    
    # Initialize the output array with zeros
    out_X = np.zeros(X.shape, dtype=X.dtype)
    
    # If Section is None, assume all cells belong to the same section
    if isinstance(Section, type(None)):
        Section = np.ones(X.shape[0])
    
    # For each feature
    for i in range(X.shape[1]):
        # Get the values of the current feature
        x = np.array(X[:, i]).copy()
        
        # Calculate the median of the clipped values
        median = np.median(np.clip(x, 1, None))
        
        # For each unique section
        for section in np.unique(Section):
            # Get a boolean mask where each cell belongs to the current section
            m = Section == section
            
            # Get the values of the current feature for the cells in the current section
            x = np.array(X[m, i]).copy()
            
            # Scale the values of the current feature for the cells in the current section
            out_X[m, i] = x * median / np.median(np.clip(x, 1, None))
    
    # Return the scaled data
    return out_X

def robust_zscore(X, axis=0):
    """
    This function applies a robust z-score normalization to the data along the specified axis.

    Parameters:
    X: A numpy array to be normalized.
    axis: The axis along which the normalization is applied. Default is 0.

    Returns:
    X: A numpy array with the same shape as the input, but with normalized values.
    """
    
    # Copy the input array
    revert = False
    if not isinstance(X,torch.Tensor):
        revert = True
        X = torch.tensor(X)
    # X = np.array(X).copy()
    
    # Subtract the median of the specified axis from the corresponding axis
    # X = X - np.median(X, axis=axis, keepdims=True)
    X = X - torch.median(X, axis=axis).values
    
    # Calculate the median absolute deviation along the specified axis
    # std = np.median(np.abs(X), axis=axis, keepdims=True)
    std = torch.median(torch.abs(X), axis=axis).values
    if torch.sum(std==0)>0:
        std[std == 0] = torch.std(X[:,std==0], axis=axis)
    std[std == 0] = 1
    # Divide the specified axis by its corresponding median absolute deviation
    X = X / std
    
    # Return the normalized data
    if revert:
        return X.numpy()
    else:
        return X
    
def quantile_scale(X, axis=0,vmin=0.05,vmax=0.95):
    """
    This function applies a robust z-score normalization to the data along the specified axis.

    Parameters:
    X: A numpy array to be normalized.
    axis: The axis along which the normalization is applied. Default is 0.

    Returns:
    X: A numpy array with the same shape as the input, but with normalized values.
    """
    
    # Copy the input array
    revert = False
    if not isinstance(X,torch.Tensor):
        revert = True
        X = torch.tensor(X)
    parameters = torch.quantile(X, torch.tensor([vmin, vmax]), dim=axis)
    parameters[1,parameters[0]==parameters[1]] = parameters[0,parameters[0]==parameters[1]]+1
    X = X - parameters[0]
    X = X/(parameters[1]-parameters[0])

    if revert:
        return X.numpy()
    else:
        return X

def linear_transform(ref_X,X, axis=0,vmin=0.05,vmax=0.95):
    if ref_X.shape[1] != X.shape[1]:
        raise ValueError("Matrices must have the same number of columns.")
    
    # Copy the input array
    revert = False
    if not isinstance(X,torch.Tensor):
        revert = True
        X = torch.tensor(X,dtype=torch.float32)
        ref_X = torch.tensor(ref_X,dtype=torch.float32)
    parameters = torch.quantile(X, torch.tensor([vmin, vmax],dtype=torch.float32), dim=axis)
    parameters[1,parameters[0]==parameters[1]] = parameters[0,parameters[0]==parameters[1]]+1
    X = X - parameters[0]
    X = X/(parameters[1]-parameters[0])
    ref_parameters = torch.quantile(ref_X, torch.tensor([vmin, vmax],dtype=torch.float32), dim=axis)
    ref_parameters[1,ref_parameters[0]==ref_parameters[1]] = ref_parameters[0,ref_parameters[0]==ref_parameters[1]]+1
    X = X*(ref_parameters[1]-ref_parameters[0])+ref_parameters[0]

    if revert:
        return X.numpy()
    else:
        return X
    

