from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cie1976
from scipy.cluster import hierarchy 
import numpy as np

import umap
import xycmap

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.spatial.distance import jensenshannon, pdist, squareform
from dredFISH.Utils.distu import *

def color_diff(clr1,clr2, mode = "RGB",de = "1976"): 
    """
    clr1/2 are 3 elements only
    """
    if mode=="RGB": 
        clr1_rgb = sRGBColor(clr1[0],clr1[1],clr1[2]);
        clr2_rgb = sRGBColor(clr2[0],clr2[1],clr2[2]);

        # Convert from RGB to Lab Color Space
        clr1_lab = convert_color(clr1_rgb, LabColor);

        # Convert from RGB to Lab Color Space
        clr2_lab = convert_color(clr2_rgb, LabColor);
    
    # Find the color difference
    if de == "2000":
        delta_e = delta_e_cie2000(clr1_lab, clr2_lab);
    elif de == "1976": 
        delta_e = delta_e_cie1976(clr1_lab, clr2_lab);
    else: 
        raise ValueError("de must be 1976 or 2000 (strings)")
    
    return delta_e

def color_diff_vec(clr1,clr2,mode = "RGB",de = "1976"): 
    """
    clr1/2 are numpy array nx3
    """
    if mode =="Lab" and de == "1976": 
        delta_e = np.sqrt(np.sum(np.power(clr1 - clr2, 2), axis=1))
    elif mode=="RGB": 
        delta_e = np.zeros((clr1.shape[0],1))

        # convert to colormath objects
        for i in range(clr1.shape[0]):
            # Find the color difference
            delta_e[i] = color_diff(clr1[i,:],clr2[i,:],mode="RGB")
    
    return delta_e

def convert_lab01_2rgb(clr_best):
    if clr_best.shape[1] != 3:
        sz = int(len(clr_best)/3)
        clr_best = np.reshape(clr_best,(sz,3))
    clr_rgb = np.zeros(clr_best.shape)
    for i in range(clr_rgb.shape[0]):
        clr_lab = LabColor(clr_best[i,0]*100,(clr_best[i,1]-0.5)*255,(clr_best[i,2]-0.5)*255)
        clr_rgb[i,:] = np.array(convert_color(clr_lab,sRGBColor).get_value_tuple())

    clr_rgb = np.clip(clr_rgb,0,1)
    return clr_rgb


def type_color_using_supervized_umap(data,target):
    reducer = umap.UMAP(n_components = 3, metric = "cosine")
    embedding = reducer.fit_transform(data, y = target)
    L = embedding[:,0]
    L = (L-L.min())/(L.max()-L.min())*100
    a = embedding[:,1]
    a = ((a-a.min())/(a.max()-a.min())-0.5)*255
    b = embedding[:,2]
    b = ((b-b.min())/(b.max()-b.min())-0.5)*255
    Lab = np.hstack((L[:,np.newaxis],a[:,np.newaxis],b[:,np.newaxis]))
    rgb_by_type = convert_lab01_2rgb(Lab)
    return rgb_by_type

def type_color_using_linkage(data,cmap,metric = "cosine"):
    dvec = pdist(data,metric)
    z = hierarchy.linkage(dvec,method='average')
    ordr = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(z,dvec))
    rgb_by_type = cmap(np.linspace(0,1,data.shape[0]+1))
    rgb_by_type = rgb_by_type[1:,:]
    rgb_by_type = rgb_by_type[ordr,:]
    return rgb_by_type

def merge_colormaps(colormap_names,range = (0,1),res = 128):
    """
    Merge multiple matplotlib colormaps into one. 
    range: either a tuple (same for all colormaps, or an Nx2 array with ranges to use. 
           full range is 0-1, so to clip any side just use partial range (0.1,1)
    """

    # make colormap_names into a list (if it's just a string name)
    if not isinstance(colormap_names,list):
        colormap_names=[colormap_names]

    # process / verify the range input
    if not isinstance(range, np.ndarray): 
        range = np.tile(range,(len(colormap_names),1))
    assert range.shape[0]==len(colormap_names), "ranges dimension doesn't match colormap names"

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors = []
    for i,cmap_name in enumerate(colormap_names): 
        cmap = cm.get_cmap(cmap_name, res)
        colors.append(cmap(np.linspace(range[i,0],range[i,1],res)))
    colors = np.vstack(colors)
    mymap = LinearSegmentedColormap.from_list('my_colormap', colors)
    return mymap