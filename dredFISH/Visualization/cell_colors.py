import numpy as np
import pandas as pd

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.spatial.transform import Rotation

from dredFISH.Visualization.utils import scale_vec

import math


def colors_from_list(lab, mat, lum=None, rot=None, axis=2):
    """
    Returns dict with pairs mapping labels to RGB colorspace

    Input
    -----
    lab : list of labels 
    mat : matrix containing items to be mapped to color space by their attributes
        e.g. gene expression matrix 
    lum : luminosity value for CIELAB color bounded between 0-100
    rot : rotation of t-SNE transformation
    axis : axis of rotation

    Output
    ------
    dict : mapping unique items to RGB color space
    """

    mat.index = lab
    mat = mat.groupby(mat.index).mean()
    
    if lum == None:
        color = get_tsne(mat, tsne_dim=3, rotation=rot, axis=axis)
        color[2] = scale_vec(color[2], 0, 100)
    else:
        color = get_tsne(mat, tsne_dim=2, rotation=rot, axis=axis)
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

def get_tsne(mat, pca_dim=20, tsne_dim=2, rs=0, prplx=30, rotation=None, axis=2):
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
    axis : axis of rotation

    Output
    ------
    Pandas DataFrame : attribute matrix reduced to t-SNE dimensions 
    """
    pca = PCA(n_components=pca_dim)
    tsne = TSNE(n_components=tsne_dim, init="random", random_state=rs, perplexity=prplx, learning_rate='auto')

    pca_mat = pca.fit_transform(mat)
    tsne_mat = tsne.fit_transform(pca_mat)

    # rotation about the origin of some axis 
    # if t-SNE has two dimensions, then the axis is always Z
    if rotation != None:
        if tsne_dim == 2:
            tsne_mat = np.append(tsne_mat, np.zeros((np.shape(tsne_mat)[0], 1)), 1)
            rotation_vector = math.radians(rotation) * np.array([0, 0, 1])
        elif axis == 2:
            rotation_vector = math.radians(rotation) * np.array([0, 0, 1])
        elif axis == 1:
            rotation_vector = math.radians(rotation) * np.array([0, 1, 0])
        elif axis == 0:
            rotation_vector = math.radians(rotation) * np.array([1, 0, 0])
        else:
            raise ValueError("Invalid axis value")

        tsne_mat = Rotation.from_rotvec(rotation_vector).apply(tsne_mat)
        
        if tsne_dim == 2:
            tsne_mat = tsne_mat[:, :-1]
        
    return pd.DataFrame(tsne_mat, columns=range(tsne_dim), index=mat.index)