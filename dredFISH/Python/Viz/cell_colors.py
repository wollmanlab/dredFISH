import numpy as np
import pandas as pd

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.ndimage.interpolation import rotate

from utils import scale_vec

import pdb
import math


def colors_from_list(lab, mat, lum=None, rot=None):
    """
    Returns dict with pairs mapping labels to RGB colorspace

    Input
    -----
    lab : list of labels 
    mat : matrix containing items to be mapped to color space by their attributes
        e.g. gene expression matrix 
    lum : luminosity value for CIELAB color 
    rot : rotation of t-SNE transformation

    Output
    ------
    dict : mapping unique items to RGB color space
    """

    mat.index = lab
    mat = mat.groupby(mat.index).mean()
    
    if lum == None:
        color = get_tsne(mat, tsne_dim=3, rotation=rot)
        color[2] = scale_vec(color[2], 0, 100)
    else:
        color = get_tsne(mat, tsne_dim=2, rotation=rot)
        color[2] = lum
    color[0] = scale_vec(color[0], -128, 127)
    color[1] = scale_vec(color[1], -128, 127)
    color = color.rename(columns={0:"A", 1:"B", 2:"L"})
    
    color["vec"] = list(map(vec2lab, color.values))
    color["rgb"] = list(map(lab2rgb, color["vec"]))
    
    return {x: color.loc[x, "rgb"] for x in color.index}

def lab2rgb(x):
    """
    Convert CIELAB color into clamped RGB color tuple 

    Input
    -----
    x : CIELAB color

    Output
    ------
    tuple : RGB color bound to [0-1] in each dimension
    """
    rgb_color = convert_color(x, sRGBColor)
    return (rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b)

def vec2lab(x):
    """
    Convert 3D vector into CIELAB color object

    Input
    -----
    x : 3D vector representing color

    Output
    ------
    LabColor : CIELAB color object 
    """
    return LabColor(lab_l=x[2],lab_a=x[0],lab_b=x[1])

def get_tsne(mat, pca_dim=20, tsne_dim=2, rs=0, prplx=30, rotation=None):
    """
    Fit t-SNE to cell counts to get color vector 

    Input
    -----
    mat : attribute matrix to be dimensionally reduced 
    pca_dim : reduced dimensions in PCA
    tsne_dim : reduced dimensions in t-SNE
    rs : random seed for t-SNE
    prplx : perplexity for t-SNE
    rotation : rotation in radians of t-SNE transformation

    Output
    ------
    Pandas DataFrame : attribute matrix reduced to t-SNE dimensions 
    """
    pca = PCA(n_components=pca_dim)
    tsne = TSNE(n_components=tsne_dim, init="random", random_state=rs, perplexity=prplx, learning_rate='auto')

    pca_mat = pca.fit_transform(mat)
    tsne_mat = tsne.fit_transform(pca_mat)

    # rotation about the origin 
    if rotation != None:
        tsne_mat = list(map(lambda x : rotate((0, 0), x, math.radians(rotation)), tsne_mat.values))
        
    return pd.DataFrame(tsne_mat, columns=range(tsne_dim), index=mat.index)