"""Visualize the cell layer of a TMG
- does some extra analysis as well
"""
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                datefmt='%m-%d %H:%M:%S', 
                level=logging.INFO,
                )
import numpy as np
import pandas as pd
import umap
import tqdm

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from dredFISH.Utils.miscu import leiden
from dredFISH.Analysis import TissueGraph
from dredFISH.Analysis import Classification
from dredFISH.Utils import powerplots
from . import compile_tex
# from ..Utils.__init__plots import *

### split hemisphere --- 
def split_hemisphere(XY, line_seg, consistency='large_x'):
    """
    Consistency=None: does not care left vs right
     = 'large_x': always select the right half
    """
    x, y = XY[:,0], XY[:,1]
    [[p1x, p1y], [p2x, p2y]] = line_seg

    vx = p2x-p1x
    vy = p2y-p1y
    vn = np.array([-vy, vx]) # normal to the line
    v = np.vstack([x-p1x,y-p1y]).T
    
    cond = v.dot(vn) < 0 # sign split points into left and right
    
    if consistency is None:
        return cond
    elif consistency == "large_x": # select the right hemisphere (bigger x)
        if np.mean(x[cond]) < np.mean(x[~cond]):
            cond = ~cond
        return cond

def adjust_XY_by_ybaseline(XY, line_seg):
    """
    """
    [[p1x, p1y], [p2x, p2y]] = line_seg
    # line direction
    v = np.array([p2x-p1x, p2y-p1y])
    v = v/np.linalg.norm(v, 2)
    vx, vy = v
    # always points up
    if vy < 0:
        v = -v
    # theta
    theta = np.arccos(v.dot([0,1]))
    if vx < 0:
        theta = -theta
    
    # rotate counter clock wise by theta
    R = np.array([
        [np.cos(theta), -np.sin(theta),], 
        [np.sin(theta),  np.cos(theta),], 
        ])
    XYnew = XY.dot(R.T)
    return XYnew

def preview_hemisphere(split_lines, basepth=None, XY=None, no_plot=False):
    """
    Visualize XY and tentative split_lines
    """
    # define a line to split things into hemi-coronal sections
    # split and adjust
    line_segs = split_lines
    if XY is None:
        # get XY from TMG
        # load TMG - with cell layer obs only
        TMG = TissueGraph.TissueMultiGraph(basepath=basepth, 
                                        redo=False, # load existing 
                                        quick_load_cell_obs=True,
                                        )
        # spatial coordinates
        layer = TMG.Layers[0]
        XY = layer.XY
        x, y = XY[:,0], XY[:,1]

    else:
        pass # ignore basepth

    cond = split_hemisphere(XY, line_segs[0])    
    XYnew = adjust_XY_by_ybaseline(XY, line_segs[0])
    x, y = XY[:,0], XY[:,1]
    xnew, ynew = XYnew[:,0], XYnew[:,1]

    if no_plot:
        return cond, XYnew

    mosaic="""
    AAB
    CCD
    """
    fig = plt.figure(figsize=(20,20), constrained_layout=True)
    axs_dict = fig.subplot_mosaic(mosaic)
    ncolors = len(line_segs)
    colors = sns.color_palette("tab10", ncolors)
    for i, (key, ax) in enumerate(axs_dict.items()):
        if i == 0:
            ax.scatter(x, y, s=1, color='black', edgecolor='none')
            ax.grid(True)
            lc = LineCollection(line_segs, linewidth=2, colors=colors) 
            ax.add_collection(lc)
        elif i == 1:
            ax.scatter(x[cond], y[cond], s=1, color='black', edgecolor='none')
            ax.grid(True)
            lc = LineCollection(line_segs, linewidth=2, colors=colors) 
            ax.add_collection(lc)
        elif i == 2:
            ax.scatter(xnew, ynew, s=1, color='black', edgecolor='none')
            ax.grid(True)
        elif i == 3:
            ax.scatter(xnew[cond], ynew[cond], s=1, color='black', edgecolor='none')
            ax.grid(True)
        ax.set_aspect('equal')
    plt.show()

    return cond, XYnew

### end split hemisphere --- 
def generate_default_analysis(
    split_lines,
    basepth, 
    XY=None,
    output_df="default_analysis.csv"):
    """
        TMG from basepth
        output goes into basepth
    """
    output_df = os.path.join(basepth, output_df)
    assert os.path.isdir(os.path.dirname(output_df))

    logging.info(f"Load TMG from {basepth}")
    TMG = TissueGraph.TissueMultiGraph(basepath=basepth, 
                                    redo=False, # load existing 
                                    )
    # unpack relevant stuff
    layer = TMG.Layers[0]
    N = layer.N
    if XY is None:
        XY = layer.XY
    else:
        pass # overwrite from outside

    x, y = XY[:,0], XY[:,1]

    # measured basis
    ftrs_mat = layer.feature_mat
    G = layer.FG
    cells = layer.adata.obs.index.values

    logging.info(f"split hemisphere...")
    # split hemisphere
    cond, XYnew = preview_hemisphere(split_lines, XY=XY, no_plot=True)

    logging.info(f"generate UMAP...")
    # UMAP
    umap_mat = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(ftrs_mat)

    logging.info(f"identify known cell types...")
    # known cell types
    allen_classifier = Classification.KnownCellTypeClassifier(
        layer, 
        tax_name='Allen_types',
        ref='allen_smrt_dpnmf',
        ref_levels=['class_label', 'neighborhood_label', 'subclass_label'], #, 'cluster_label'], 
        model='knn',
    )
    allen_classifier.train(verbose=True)
    type_mat = allen_classifier.classify()

    logging.info(f"cell clustering (unsupervised types)...")
    # clustering
    resolutions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1,2,5,10]
    clst_mat = []
    for i, r in tqdm.tqdm(enumerate(resolutions)):
        types = leiden(G, cells, resolution=r)
        # add to a df 
        clst_mat.append(types)
    
    logging.info(f"organizing results...")
    # add results to a df 
    # basics
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['x2'] = XYnew[:,0]
    df['y2'] = XYnew[:,1]
    df['hemi'] = cond.astype(int)

    # basis
    for i in range(24):
        df[f'b{i}'] = ftrs_mat[:,i]

    # umap
    df['umap_x'] = umap_mat[:,0]
    df['umap_y'] = umap_mat[:,1]

    # ktype
    for i in range(3):
        df[f'ktype_L{i+1}'] = type_mat[:,i]
        
    # type
    for i, r in enumerate(resolutions):
        types = clst_mat[i]
        df[f'type_r{r}'] = np.char.add('t', np.array(types).astype(str))
        
    # save
    df.to_csv(output_df, header=True, index=True)
    logging.info(f"saved results to: {output_df}")

    return df 

def generate_default_views(
    df, respth,):
    """
    """
    output = os.path.join(respth, 'fig1_basis_space.pdf')
    powerplots.plot_basis_spatial(df, output=output)

    dfsub = df[df['hemi']==1]
    output = os.path.join(respth, 'fig1-2_basis_space_righthalf.pdf')
    powerplots.plot_basis_spatial(dfsub, pmode='right_half', output=output)

    output = os.path.join(respth, 'fig2_basis_umap.pdf')
    powerplots.plot_basis_umap(df, output=output)

    typecols = df.filter(regex='^type_r', axis=1).columns
    for i, col in enumerate(typecols):
        hue = col
        output = os.path.join(respth, f'fig3-{i}_{col}.pdf')
        powerplots.plot_type_spatial_umap(df, hue, output=output)
    return 

def main(mode, 
        basepth, split_lines, 
        respth=None,
        compile_pdf=True,
        pdf_kwargs={'title': 'dredFISH default analysis', 
                    'author':'Fangming',
                    },
        tmg_kwargs=dict(
                    norm='default',
                    norm_cell=True,
                    norm_basis=True,
                    ),
        redo=False,
        ):
    """
    `respth` is for figures
    dataframe results goes to `basepth` `default_analysis.csv`
    """
    # House keeping
    assert mode in ['preview', 'view', 'analysis-only', 'plot-only',  'compile-only'] # choose from these options
    tmg_pth = os.path.join(basepth, 'TMG.json')
    if redo or not os.path.isfile(tmg_pth):  
        logging.info(f"TMG does not exist, generating from scratch (cell layer only)")
        TMG = TissueGraph.TissueMultiGraph(
                                basepath=basepth, 
                                redo=True, # create an empty one
                                ) 
        TMG.create_cell_layer(metric='cosine', **tmg_kwargs)
        TMG.save()

    if respth is None:
        respth = os.path.join(basepth, 'figures')
        if not os.path.isdir(respth):
            os.mkdir(respth)

    # MAIN
    if mode == 'preview': # 
        cond, XYnew = preview_hemisphere(split_lines, basepth=basepth, XY=None)
        return cond, XYnew

    elif mode == 'view': # analysis + plot
        cond, XYnew = preview_hemisphere(split_lines, basepth=basepth, XY=None)
        df = generate_default_analysis(split_lines, basepth, XY=None,
                output_df="default_analysis.csv")
        generate_default_views(df, respth)
        if compile_pdf:
            compile_tex.main(basepth, **pdf_kwargs)
        return df

    elif mode == 'analysis-only': # 
        cond, XYnew = preview_hemisphere(split_lines, basepth=basepth, XY=None)
        df = generate_default_analysis(split_lines, basepth, XY=None,
                output_df="default_analysis.csv")
        return df

    elif mode == 'plot-only': # 
        dfpth = os.path.join(basepth, 'default_analysis.csv')
        df = pd.read_csv(dfpth, index_col=0)
        generate_default_views(df, respth)
        if compile_pdf:
            compile_tex.main(basepth, **pdf_kwargs)
        return 
    elif mode == 'compile-only': # 
        compile_tex.main(basepth, **pdf_kwargs)
        return 

if __name__ == '__main__':
    # mode = 'preview'
    mode = 'view'
    # TMG
    basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t4'
    # define a line to split things into hemi-coronal sections
    split_lines = [
        # [(0,0),(1,1)],
        [(550, -6000), (200, 2000)],
        [(200, 2000), (550, -6000)],
        # [(300, 2000), (550, -6000)],
    ]

    main(mode, basepth, split_lines)

