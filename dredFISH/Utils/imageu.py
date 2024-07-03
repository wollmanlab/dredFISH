# scripts for image processing

import numpy as np
from scipy import ndimage
from scipy import sparse
import pandas as pd
from sklearn.decomposition import PCA
import datashader as ds

import torch

from skimage import io
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import block_reduce

from . import powerplots


def max_norm(ar, maxp=99):
    """ To [0-1], assume the array contains non-negative elements; maxp specify the percentile to clip
    """
    maxclip = np.percentile(ar.flatten(), maxp)
    normar = np.clip(ar, None, maxclip)
    normar = normar/maxclip
    return normar

def block_mean(ar, fact):
    """Downsample an 2d numpy array by a bin; taking the block mean

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
    """Downsample an 2d numpy array by a bin; taking the block mean

    """
    assert isinstance(fact, int)
    sx, sy, sz = ar.shape
    dx, dy = int(sx/fact), int(sy/fact)
    tx, ty = dx*fact, dy*fact
    
    ar = ar[:tx,:ty,:]
    res = ar[::fact,::fact,:]
    return res 

# rotation transformation
def pca_rotate(pmat, allow_reflection=False, vec_chiral=np.array([]), random_state=0):
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
    assert pmat.shape[1] == 2

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

def valued_pointset_to_image(pmat, resolution=1, return_coords=False):
    """Generate heatmap based on pointset
    pmat has 3 cols
    
    resolution is approximate, it was adjusted according to rounding error (number of pixels)
    ds.Canvas rasterize the pointset by (nx, ny) equal-sized bins.
    coordinates can be recovered by
    
    xcoords = xmin + xr*(1/2 + col(x)_index) # coords represents the min-points of each bin 
    ycoords = ymin + yr*(1/2 + row(y)_index) 

    """
    assert pmat.shape[1] == 3

    xmin = pmat[:,0].min()
    xmax = pmat[:,0].max()
    ymin = pmat[:,1].min()
    ymax = pmat[:,1].max()
    rangex = xmax - xmin
    rangey = ymax - ymin

    ps = powerplots.PlotScale(rangex, rangey, pxl_scale=resolution)
    xr = ps.pxl_scale_ux
    yr = ps.pxl_scale_uy

    data = pd.DataFrame(pmat, columns=['x', 'y', 'z'])
    aggdata = ds.Canvas(plot_width=ps.npxlx, plot_height=ps.npxly).points(data, 'x', 'y', agg=ds.mean('z'))
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
    trans_mat = np.array([[-1,0],[0,-1]])
    return np.dot(points, trans_mat)

def stkread(full_file_list):
    """
    read a stack of images based on list of filenames
    """
    img = io.imread(full_file_list[0])
    sz = ((img.shape[0],img.shape[1],len(full_file_list)))
    stk = np.zeros(sz,dtype = img.dtype)
    stk[:,:,0] = img
    for i in range(1,len(full_file_list)):
        stk[:,:,i] = io.imread(full_file_list[i])
    return stk


def iterative_gaussian_filter(img,sigma=1,iter=1,smooth_bound = True):
    img_iter = img.copy()
    for i in range(iter):
        img_iter = gaussian_filter(img_iter,sigma=sigma,mode='nearest')
        img_iter[:sigma,:] = img[:sigma,:]
        img_iter[-sigma:,:] = img[-sigma:,:]
        img_iter[:,:sigma] = img[:,:sigma]
        img_iter[:,-sigma:] = img[:,-sigma:] 
    if smooth_bound: 
        img_iter = gaussian_filter(img_iter,sigma=sigma,mode='nearest')

    return img_iter

def median_bin(img_or_stk,bin = 2): 
    # if only a single image: 
    if len(img_or_stk.shape)==2: 
        img_flt = block_reduce(median_filter(img_or_stk[:,:,i],2), tuple([2,2]), np.mean)
        return img_flt
    else: 
        stk = img_or_stk

    # init an empty downsized stack
    stk_ds = np.zeros((stk.shape[0]//bin,stk.shape[1]//bin,stk.shape[2]),dtype=stk.dtype)
    for i in range(stk.shape[2]): 
        stk_ds[:,:,i] = block_reduce(median_filter(stk[:,:,i],2), tuple([2,2]), np.mean)

    return stk_ds

def fast_median_bin(stk,bin=2):
    if stk.shape[0] % bin != 0 or stk.shape[1] % bin != 0:
            raise ValueError(f"Image dimensions must be divisible by {bin} for downsampling.")
    if len(stk.shape)==2: 
        reshaped = stk.reshape(stk.shape[0] // bin, bin, stk.shape[1] // bin, bin)
        output = np.median(reshaped, axis=(1, 3)) 
    else:
        output = np.zeros([stk.shape[0]//bin,stk.shape[1]//bin,stk.shape[2]],dtype=stk.dtype)
        for i in range(stk.shape[2]): 
            img = stk[:,:,i].copy()
            reshaped = img.reshape(img.shape[0] // bin, bin, img.shape[1] // bin, bin)
            img = np.median(reshaped, axis=(1, 3)) 
            output[:,:,i] = img
    return output



def estimate_flatfield_and_constant(full_file_list):
    if len(full_file_list)<5:
        raise ValueError("Too few images to estimate flatfield and constant")
    stk = stkread(full_file_list)
    stk_ds = fast_median_bin(stk)

    # Calc constnt by taking bottom 1% and smoothing with iternative gaussian
    stk_tensor = torch.Tensor(stk_ds.astype(np.float32))
    n = np.min([stk_tensor.shape[2],100])
    Mraw_tensor = torch.kthvalue(stk_tensor, k=stk_tensor.shape[2]//n, dim=2).values
    # Mraw_tensor = torch.min(Mraw_tensor, axis=2).values
    Mraw = np.array(Mraw_tensor)
    Mflt = iterative_gaussian_filter(Mraw,sigma=20,iter=5)

    # now subtract const
    M_tensor = torch.Tensor(Mflt.astype(np.float32))
    stk_m_M_tensor = stk_tensor - M_tensor.unsqueeze(2)

    # rescale so all images get the same weight
    avg_img_scale = torch.mean(stk_m_M_tensor, axis=(0, 1))
    avg_img_scale = avg_img_scale/avg_img_scale.mean()
    stk_rescaled_tensor = stk_m_M_tensor/avg_img_scale.unsqueeze(0).unsqueeze(1)

    # get Imed and smooth
    Imed_tensor = torch.kthvalue(stk_rescaled_tensor, k=stk_tensor.shape[2]//2, dim=2).values
    FF = iterative_gaussian_filter(np.array(Imed_tensor),sigma=20,iter=5)

    # convert to FF (1/FF and rescale)
    FF = 1/FF
    FF = FF/FF.mean()

    return (FF,Mflt)

from metadata import Metadata
from dredFISH.Utils import imageu
import multiprocessing
from functools import partial
from tqdm import tqdm
from dredFISH.Utils import fileu
import os
import matplotlib.pyplot as plt
import shutil
def wrapper(acq,image_metadata,channel,path=''):
    try:
        image_metadata = Metadata(os.path.join(image_metadata,acq))
        well = [i for i in image_metadata.base_pth.split('/') if not i==''][-2].split('_')[0]#image_metadata.posnames[0].split('-')[0]
        f = fileu.generate_filename(section=well,path=path,hybe=acq,channel=channel,file_type='FF')
        if os.path.exists(f):
            try:
                FF = fileu.load(section=well,path=path,hybe=acq,channel=channel,file_type='FF')
            except:
                FF = None
        else:
            FF = None

        f = fileu.generate_filename(section=well,path=path,hybe=acq,channel=channel,file_type='constant')
        if os.path.exists(f):
            try:
                C = fileu.load(section=well,path=path,hybe=acq,channel=channel,file_type='constant')
            except:
                C = None
        else:
            C = None
        if isinstance(C,type(None))|isinstance(FF,type(None)):
            file_list = image_metadata.stkread(Channel=channel,acq=acq,groupby='Channel',fnames_only = True)
            (FF, C) = imageu.estimate_flatfield_and_constant(file_list)
            fileu.save(FF,section=well,path=path,hybe=acq,channel=channel,file_type='FF')
            fileu.save(FF*1000,section=well,path=path,hybe=acq,channel=channel,file_type='image_FF')
            fileu.save(C,section=well,path=path,hybe=acq,channel=channel,file_type='constant')
            fileu.save(C,section=well,path=path,hybe=acq,channel=channel,file_type='image_constant')

        return well,acq,FF,C
    except Exception as e:
        print(f"{acq} Failed")
        print(e)
        return None,acq,None,None

def generate_image_parameters(base_path,overwrite=True,nthreads = 10):
    dataset = [i for i in base_path.split('/') if not i==''][-1]
    out_path = os.path.join(base_path,'microscope_parameters')
    if overwrite:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    image_metadata = Metadata(base_path)
    for channel in ['FarRed','DeepBlue']:
        Input = sorted([i for i in image_metadata.acqnames if ('ybe' in i)|('rip' in i)])
        np.random.shuffle(Input)
        pfunc = partial(wrapper,image_metadata=base_path,channel=channel,path=out_path)
        with multiprocessing.Pool(nthreads) as p:
            for well,acq,FF,C in tqdm(p.imap(pfunc,Input),total=len(Input),desc=f"{dataset} {channel}"):
                if isinstance(FF,type(None)):
                    continue
                fig,axs = plt.subplots(1,2,figsize=[12,4])
                fig.suptitle(f"{dataset} {acq} {well} {channel}")
                axs = axs.ravel()
                ax = axs[0]
                im = ax.imshow(C,cmap='jet')
                plt.colorbar(im,ax=ax)
                ax.axis('off')
                ax.set_title("const")
                ax = axs[1]
                im=ax.imshow(FF,cmap='jet')
                plt.colorbar(im,ax=ax)
                ax.axis('off')
                ax.set_title("FF")
                path = fileu.generate_filename(section=well,path=out_path,hybe=acq,channel=channel,file_type='Figure')
                plt.savefig(path)
                plt.close('all')