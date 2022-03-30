# scripts for image processing

import numpy as np
from scipy import ndimage
from scipy import sparse
import pandas as pd
from sklearn.decomposition import PCA
import datashader as ds

from . import powerplots


def max_norm(ar, maxp=99):
    """ To [0-1], assume the array contains non-negative elements; maxp specify the percentile to clip
    """
    maxclip = np.percentile(ar.flatten(), maxp)
    normar = np.clip(ar, None, maxclip)
    normar = normar/maxclip
    return normar

def block_mean(ar, fact):
    """Downsample an 2d numpy array by a factor; taking the block mean

    """
    assert isinstance(fact, int)
    sx, sy = ar.shape
    dx, dy = int(sx/fact), int(sy/fact)
    tx, ty = dx*fact, dy*fact
    
    X, Y = np.ogrid[0:tx, 0:ty]
    
    regions = dy * (X/fact).astype(int) + (Y/fact).astype(int)
    res = ndimage.mean(ar[:tx,:ty], labels=regions, index=np.arange(regions.max()+1))
    res.shape = (dx, dy)
    return res

def block_downsamp_3d(ar, fact):
    """Downsample an 2d numpy array by a factor; taking the block mean

    """
    assert isinstance(fact, int)
    sx, sy, sz = ar.shape
    dx, dy = int(sx/fact), int(sy/fact)
    tx, ty = dx*fact, dy*fact
    
    ar = ar[:tx,:ty,:]
    res = ar[::fact,::fact,:]
    return res 

# rotation transformation
def pca_rotate(pmat, allow_reflection=True, vec_chiral=np.array([]), random_state=0):
    """(n,2) matrix as input (point set)
    learn a rotation/reflection along PC1 and PC2.
    """
    # U, s, Vt = fbpca.pca(mat, k=2)
    # pcmat = U.dot(np.diag(s))
    _res = PCA(n_components=2, random_state=random_state).fit(pmat)
    
    pcmat = _res.transform(pmat)
    Vt = _res.components_

    # det = -1 reflection; det = 1 rotation
    if not allow_reflection and np.linalg.det(Vt) < 0: # ~ -1
        pcmat[:,0] = -pcmat[:,0]
        Vt[0,:] = -Vt[0,:]
        
    # define chirality (dorsal-ventral)
    if isinstance(vec_chiral, np.ndarray) and len(vec_chiral) > 0:
        delta = vec_chiral[pcmat[:,1]>0].sum() - vec_chiral[pcmat[:,1]<0].sum()
        if delta < 0: # rotate 180 
            pcmat = -pcmat
            Vt = -Vt
    return pcmat, Vt

# 
def image_to_pointset(imgmat, scale=1, preserve_intensity=False):
    """Assume imgmat is non-negative. Turn non-zero pixels into points.
    """
    pmat = sparse.coo_matrix(imgmat)
    pmat = scale*np.vstack([pmat.row, pmat.col]).T
    return pmat
    
def pointset_to_image(pmat, resolution=1, return_coords=False):
    """Generate heatmap based on pointset
    
    resolution is approximate, it was adjusted according to rounding error (number of pixels)
    ds.Canvas rasterize the pointset by (nx, ny) equal-sized bins.
    coordinates can be recovered by
    
    xcoords = xmin + xr*(1/2 + col(x)_index) # coords represents the min-points of each bin 
    ycoords = ymin + yr*(1/2 + row(y)_index) 
    """
    xmin = pmat[:,0].min()
    xmax = pmat[:,0].max()
    ymin = pmat[:,1].min()
    ymax = pmat[:,1].max()
    rangex = xmax - xmin
    rangey = ymax - ymin

    ps = powerplots.PlotScale(rangex, rangey, pxl_scale=resolution)
    xr = ps.pxl_scale_ux
    yr = ps.pxl_scale_uy

    data = pd.DataFrame(pmat, columns=['x', 'y'])
    aggdata = ds.Canvas(plot_width=ps.npxlx, plot_height=ps.npxly).points(data, 'x', 'y', agg=ds.count())
    imgmat = aggdata.values

    if return_coords:
        coordmat = np.array([
            [xmin, ymin], # origin
            [xr, yr], # resolution
        ])
        return imgmat, coordmat 
    else:
        return imgmat

def imgidx_to_coords(rowidx, colidx, coordmat):
    """
    coordmat - 2d array [[xmin, ymin], [xr, yr]]
    """
    [[xmin, ymin], [xr, yr]] = coordmat

    xcoords = xmin + xr*(1/2.0 + colidx)
    ycoords = ymin + yr*(1/2.0 + rowidx)
    return xcoords, ycoords

def coords_to_imgidx(xcoords, ycoords, coordmat):
    """
    coordmat - 2d array [[xmin, ymin], [xr, yr]]
    """
    [[xmin, ymin], [xr, yr]] = coordmat

    colidx = ((xcoords-xmin)/xr).astype(int)
    rowidx = ((ycoords-ymin)/yr).astype(int)

    return rowidx, colidx

def remove_zero_paddings(imgmat, threshold=0.01):
    """
    """
    x = imgmat.sum(axis=1)
    th_x = threshold*np.max(x)
    y = imgmat.sum(axis=0)
    th_y = threshold*np.max(y)
    lcut, rcut = np.arange(len(x))[x>th_x][[1,-1]]
    bcut, tcut = np.arange(len(y))[y>th_y][[1,-1]]

    return imgmat[lcut:rcut,bcut:tcut]

def broad_padding(moving, fixed_shape):
    """match CCF padding
    assume moving image is a subset of CCF
    """
    m_moving, n_moving = moving.shape
    m_fixed, n_fixed = fixed_shape
    
    assert m_moving <= m_fixed
    assert n_moving <= n_fixed
    
    # pad the moving
    m_diff1, m_diff2 = 0, 0
    if m_moving < m_fixed:
        m_diff = m_fixed - m_moving
        m_diff1 = int(m_diff/2)
        m_diff2 = m_diff - m_diff1
        
    n_diff1, n_diff2 = 0, 0
    if n_moving < n_fixed:
        n_diff = n_fixed - n_moving
        n_diff1 = int(n_diff/2)
        n_diff2 = n_diff - n_diff1
        
    pad_transform = [(m_diff1, m_diff2), (n_diff1, n_diff2)]
    
    return np.pad(moving, pad_transform), pad_transform

def get_img_centroid(mat, scale=1):
    m, n = mat.shape
    normmat = mat/mat.sum()
    
    center_m = np.arange(m).dot(normmat).sum()
    center_n = normmat.dot(np.arange(n)).sum()
    return center_m*scale, center_n*scale

def flip_points(points):
    """assume a n*2 matrix
    """
    trans_mat = np.array([[1,0],[0,-1]])
    return np.dot(points, trans_mat)
