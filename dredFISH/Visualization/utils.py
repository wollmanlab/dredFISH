import pandas as pd
import numpy as np

import math

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

    assert type(lab) == np.ndarray, "input label must be NumPy array" 

    num_typs = sum(dend["H"] <= level)
    for i in range(num_typs):
        cut = dend.iloc[i]
        if cut["A"] in lab:
            lab[lab == cut["A"]] = cut["C"]
        if cut["B"] in lab:
            lab[lab == cut["B"]] = cut["C"]
            
    return lab

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

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.

    https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python

    Input
    -----
    origin : origin over which rotation takes place
    point : points to be rotated
    angle: angle of rotation

    Output
    ------
    tuple : rotated points 
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
