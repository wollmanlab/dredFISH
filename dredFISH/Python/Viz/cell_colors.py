import numpy as np
import pandas as pd

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from copy import copy

def colors_from_list(lab, mat, lum=50):
    """
    Returns dict with pairs mapping labels to RGB colorspace
    """
    mat.index = lab
    mat = mat.groupby(mat.index).mean()
    
    color = get_tsne(mat, tsne_dim=2)
    color[0] = scale_vec(color[0], -128, 127)
    color[1] = scale_vec(color[1], -128, 127)
    color = color.rename(columns={0:"A",1:"B"})
    color["L"] = lum

    color["vec"] = list(map(vec2lab, color.values))
    # get clamped RGB colors after converting from CIELAB to sRGB
    color["rgb"] = [(x.clamped_rgb_r, x.clamped_rgb_g, x.clamped_rgb_b) for x in list(map(lambda x: convert_color(x, sRGBColor), color["vec"]))]
     
    return {x: color.loc[x, "rgb"] for x in color.index}

def vec2lab(x):
    """
    Convert 3D vector into CIELAB color object
    """
    return LabColor(lab_l=x[2],lab_a=x[0],lab_b=x[1])

def scale_vec(x, low, high):
    """
    Scale vector between bounds
    """
    return ((high - low) * (x - x.min()) / (x.max() - x.min())) + low 

def get_tsne(mat, pca_dim=20, tsne_dim=2, rs=0, prplx=30):
    """
    Fit t-SNE to cell counts to get color vector 
    """
    pca = PCA(n_components=pca_dim)
    tsne = TSNE(n_components=tsne_dim, init="random", random_state=rs, perplexity=prplx, learning_rate='auto')

    pca_mat = pca.fit_transform(mat)
    tsne_mat = tsne.fit_transform(pca_mat)

    return pd.DataFrame(tsne_mat, columns=range(tsne_dim), index=mat.index)

def get_aggregate_counts(fpath, cmpr='gzip', by_cell = True, by_log = True):
    """
    Get cell counts and average over cell type
    """
    counts = pd.read_csv(fpath, compression=cmpr, index_col=0)

    if by_cell:
        counts = (counts.T / counts.sum(1).T).T
    if by_log:
        counts = np.log2(counts + 1)
    
    return counts

def cut_dend(dend, labels, level):
    """
    Cut dendrogram at some level and return adjusted labels 
    """
    labels = copy(labels)
    num_typs = sum(dend["H"] <= level)
    for i in range(num_typs):
        cut = dend.iloc[i]
        if cut["A"] in labels.values:
            labels.loc[labels.values == cut["A"]] = cut["C"]
        if cut["B"] in labels.values:
            labels.loc[labels.values == cut["B"]] = cut["C"]
    return labels

def ct2subct(labels, sub):
    """
    Map cell type to sub-cell type
    """
    cell_map = {}
    for idx, x in enumerate(labels):
        if x not in cell_map:
            cell_map[x] = sub.iloc[idx]
    return cell_map

def traverse_dend(labels, dend):
    """
    Find base labels for some label in dendrogram
    """
    if labels[0] != "n":
        return [labels]
    else:
        right, left=dend.loc[dend.C == labels, ["A","B"]].values[0]
        return traverse_dend(right, dend) + traverse_dend(left, dend)

def avg_dend(dend, counts):
    """
    Find average gene counts for all nodes in dendrogram 
    """
    dend_labels = set(list(dend["C"].values) + list(dend["A"].values) + list(dend["B"].values))
    dend_counts = pd.DataFrame(columns=counts.keys())

    for x in dend_labels:
        labels = traverse_dend(x, dend)
        dend_counts.loc[x] = counts.loc[[True if x in labels else False for x in counts.index]].mean()
    
    return dend_counts.dropna()