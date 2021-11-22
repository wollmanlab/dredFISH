import numpy as np
import pandas as pd

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from copy import copy
import pdb

def colors_from_list(lab, mat, lum=50):
    """
    Returns dict with pairs mapping labels to RGB colorspace

    Input
    -----
    lab : list of labels 
    mat : matrix containing items to be mapped to color space by their attributes
        e.g. gene expression matrix 
    lum : luminosity value for CIELAB color 

    Output
    ------
    dict : mapping unique items to RGB color space
    """

    mat.index = lab
    mat = mat.groupby(mat.index).mean()
    
    color = get_tsne(mat, tsne_dim=2)
    color[0] = scale_vec(color[0], -128, 127)
    color[1] = scale_vec(color[1], -128, 127)
    color = color.rename(columns={0:"A",1:"B"})
    color["L"] = lum
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

def scale_vec(x, low, high):
    """
    Scale vector between bounds

    Input
    -----
    x : 1D vector
    low : minimum value after rescaling
    high : maximum value after rescaling

    Output
    ------
    NumPy array : rescaled values 
    """
    return ((high - low) * (x - x.min()) / (x.max() - x.min())) + low 

def get_tsne(mat, pca_dim=20, tsne_dim=2, rs=0, prplx=30):
    """
    Fit t-SNE to cell counts to get color vector 

    Input
    -----
    mat : attribute matrix to be dimensionally reduced 
    pca_dim : reduced dimensions in PCA
    tsne_dim : reduced dimensions in t-SNE
    rs : random seed for t-SNE
    prplx : perplexity for t-SNE

    Output
    ------
    Pandas DataFrame : attribute matrix reduced to t-SNE dimensions 
    """
    pca = PCA(n_components=pca_dim)
    tsne = TSNE(n_components=tsne_dim, init="random", random_state=rs, perplexity=prplx, learning_rate='auto')

    pca_mat = pca.fit_transform(mat)
    tsne_mat = tsne.fit_transform(pca_mat)

    return pd.DataFrame(tsne_mat, columns=range(tsne_dim), index=mat.index)

def get_aggregate_counts(fpath, cmpr='gzip', by_cell = True, by_log = True):
    """
    Get cell counts and average over cell type

    Input
    -----
    fpath : file path 
    cmpr : compression of file
    by_cell : normalize rows by sum 
    by_log : normalize by counts to log scale

    Output
    ------
    Pandas DataFrame : normalized counts for some attribute matrix
    """
    counts = pd.read_csv(fpath, compression=cmpr, index_col=0)

    if by_cell:
        counts = (counts.T / counts.sum(1).T).T
    if by_log:
        counts = np.log2(counts + 1)
    
    return counts

def cut_dend(dend, lab, level):
    """
    Cut dendrogram at some level and return adjusted labels 

    Input
    -----
    dend : dendrogram of labels with each row corresponding to two merging elements from columns A and B into column C at height H 
    lab : list of labels to be adjusted at tree height 
    level : level of dendrogram cut 

    Output
    ------
    NumPy array : list of modified labels 
    """
    num_typs = sum(dend["H"] <= level)
    for i in range(num_typs):
        cut = dend.iloc[i]
        if cut["A"] in lab:
            lab[lab == cut["A"]] = cut["C"]
        if cut["B"] in lab:
            lab[lab == cut["B"]] = cut["C"]
    return lab

def ct2subct(lab, sub):
    """
    Map cell type to sub-cell type

    Input
    -----
    lab : list of labels
    sub : list of sub-labels 

    Output
    ------
    dict : mapping labels to sub-labels 
    """
    cell_map = {}
    for i in range(len(lab)):
        if lab[i] not in cell_map:
            cell_map[lab[i]] = sub[i]

    return cell_map

def traverse_dend(lab, dend):
    """
    Find base labels for some label in dendrogram recursively. Nodes that are not leaves in the dendrogram start with a lowercase n and then some integer 
        e.g. n123

    Input
    -----
    lab : a cell type label
    dend : dendrogram in table format

    Output
    ------
    list : all leaf nodes in the dendrogram that share the same label at higher level
    """
    if lab[0] != "n":
        return [lab]
    else:
        right, left=dend.loc[dend.C == lab, ["A","B"]].values[0]
        return traverse_dend(right, dend) + traverse_dend(left, dend)

def avg_dend(dend, counts):
    """
    Find average gene counts for all nodes in dendrogram 
    
    Input
    -----
    dend : dendrogram in table format
    counts : attribute matrix 

    Output
    ------
    Pandas DataFrame with average counts for some cell type 
    """
    # all nodes in dendrogram 
    dend_labels = set(list(dend["C"].values) + list(dend["A"].values) + list(dend["B"].values))
    dend_counts = pd.DataFrame(columns=counts.keys())

    for x in dend_labels:
        labels = traverse_dend(x, dend)
        dend_counts.loc[x] = counts.loc[[True if x in labels else False for x in counts.index]].mean()
    
    return dend_counts.dropna()