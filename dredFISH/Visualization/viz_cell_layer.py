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
import glob
import subprocess
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from dredFISH.Utils.miscu import leiden
from dredFISH.Utils.miscu import is_in_polygon 
from dredFISH.Utils import tmgu
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
    theta = np.arcsin(vx)
    
    # rotate counter clock wise by theta
    R = np.array([
        [np.cos(theta), -np.sin(theta),], 
        [np.sin(theta),  np.cos(theta),], 
        ])
    XYnew = XY.dot(R.T)
    return XYnew

def rotate(XY, theta):
    """
    """
    # rotate counter clock wise by theta
    R = np.array([
        [np.cos(theta), -np.sin(theta),], 
        [np.sin(theta),  np.cos(theta),], 
        ])
    XYnew = XY.dot(R.T)
    return XYnew

def draw_control_points(points, ax):
    """
    """
    line_segs = [
        [points[i], points[(i+1)%len(points)]]
        for i in range(len(points))
    ]

    pm = np.asarray(points)
    ax.scatter(pm[:,0], pm[:,1], color='r')
    for i, p in enumerate(points):
        ax.text(p[0], p[1], i)
    lc = LineCollection(line_segs, linewidth=1, colors='r')
    ax.add_collection(lc)
    return line_segs

def preview_hemisphere(split_lines, basepth=None, XY=None, bounding_points=None, no_plot=False, title=None):
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

    if bounding_points is not None:
        isinpoly = is_in_polygon(bounding_points, XY)
    else:
        isinpoly = np.repeat(True, len(x))

    if no_plot:
        return cond, isinpoly, XYnew

    mosaic="""
    AAB
    CCD
    """
    fig = plt.figure(figsize=(20,20), constrained_layout=True)
    axs_dict = fig.subplot_mosaic(mosaic)
    ncolors = len(line_segs)
    # colors = sns.color_palette("tab10", ncolors)
    colors = ['r']*ncolors
    for i, (key, ax) in enumerate(axs_dict.items()):
        if i == 0:
            ax.scatter(x, y, s=1, color='black', edgecolor='none')
            ax.grid(True)
            ax.set_aspect('equal')
            lc = LineCollection(line_segs, linewidth=2, colors=colors) 
            ax.add_collection(lc)

            if bounding_points is not None:
                ax.scatter(x[isinpoly], y[isinpoly], s=1, color='C0', edgecolor='none')
                draw_control_points(bounding_points, ax)

            if title is not None:
                ax.set_title(title)

        elif i == 1:
            ax.scatter(x[cond], y[cond], s=1, color='black', edgecolor='none')
            ax.grid(True)
            ax.set_aspect('equal')
            lc = LineCollection(line_segs, linewidth=2, colors=colors) 
            ax.add_collection(lc)

            if bounding_points is not None:
                _tmpcond = np.logical_and(isinpoly, cond)
                ax.scatter(x[_tmpcond], y[_tmpcond], s=1, color='C0', edgecolor='none')

        elif i == 2:
            ax.scatter(xnew, ynew, s=1, color='black', edgecolor='none')
            ax.grid(True)
            ax.set_aspect('equal')

            if bounding_points is not None:
                ax.scatter(xnew[isinpoly], ynew[isinpoly], s=1, color='C0', edgecolor='none')

        elif i == 3:
            ax.scatter(xnew[cond], ynew[cond], s=1, color='black', edgecolor='none')
            ax.grid(True)
            ax.set_aspect('equal')

            if bounding_points is not None:
                _tmpcond = np.logical_and(isinpoly, cond)
                ax.scatter(xnew[_tmpcond], ynew[_tmpcond], s=1, color='C0', edgecolor='none')

    plt.show()
    return cond, isinpoly, XYnew

### end split hemisphere --- 


def generate_default_analysis(
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
                                       quick_load_cell_obs=True,
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
    G  = layer.FG
    SG = layer.SG
    cells = layer.adata.obs.index.values

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

    # region type (local cell type abundances)
    typebasis = type_mat[:,1] # known cell types Level 2
    env_mat = tmgu.get_local_type_abundance(typebasis, SG=SG)
    k_kms = [5, 10, 20, 50] 
    reg_mat = []
    for k_km in tqdm.tqdm(k_kms):
        kmeans = KMeans(n_clusters=k_km, random_state=1)
        reg_clsts = kmeans.fit_predict(env_mat)
        reg_mat.append(reg_clsts)
    
    logging.info(f"organizing results...")
    # add results to a df 
    # basics
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['hemi'] = layer.adata.obs['hemi'].values # important because the index doesn't match

    # basis
    for i in range(24):
        df[f'b{i}'] = ftrs_mat[:,i]

    # umap
    df['umap_x'] = umap_mat[:,0]
    df['umap_y'] = umap_mat[:,1]

    # ktype (known cell types)
    for i in range(3):
        df[f'ktype_L{i+1}'] = type_mat[:,i]
        
    # type
    for i, r in enumerate(resolutions):
        types = clst_mat[i]
        df[f'type_r{r}'] = np.char.add('t', np.array(types).astype(str))
        
    # region
    for i, k_km in enumerate(k_kms):
        df[f'regtype_allenL2basis_k{k_km}'] = np.char.add('reg', np.array(reg_mat[i]).astype(str))

    # save
    df.to_csv(output_df, header=True, index=True)
    logging.info(f"saved results to: {output_df}")

    return df 

def generate_default_views(
    df, respth, title=None):
    """
    """
    # 24 basis
    output = os.path.join(respth, 'fig1_basis_space.pdf')
    powerplots.plot_basis_spatial(df, title=title, output=output)
    plt.close()

    dfsub = df[df['hemi']==1]
    output = os.path.join(respth, 'fig1-2_basis_space_righthalf.pdf')
    powerplots.plot_basis_spatial(dfsub, pmode='right_half', title=title, output=output)
    plt.close()

    output = os.path.join(respth, 'fig2_basis_umap.pdf')
    powerplots.plot_basis_umap(df, title=title, output=output)
    plt.close()

    # clusters
    typecols = df.filter(regex='^type_r', axis=1).columns
    for i, col in enumerate(typecols):
        hue = col
        output = os.path.join(respth, f'fig3-{i}_{col}.pdf')
        powerplots.plot_type_spatial_umap(df, hue, title=title, output=output)
        plt.close()
    
    # known types
    ktypecols = df.filter(regex='^ktype_L', axis=1).columns
    for i, col in enumerate(ktypecols):
        hue = col
        output = os.path.join(respth, f'fig4-{i}_{col}.pdf')
        powerplots.plot_type_spatial_umap(df, hue, title=title, output=output)
        plt.close()

    # region
    regtypecols = df.filter(regex='^regtype_allenL2basis_k', axis=1).columns
    for i, col in enumerate(regtypecols):
        hue = col
        output = os.path.join(respth, f'fig5-{i}_{col}.pdf')
        powerplots.plot_type_spatial_umap(df, hue, title=title, output=output)
        plt.close()

    # region polygons
    regtypecols = df.filter(regex='^regtype_allenL2basis_k', axis=1).columns
    for i, col in enumerate(regtypecols):
        hue = col
        output = os.path.join(respth, f'fig6-{i}_{col}.pdf')
        _c,l = pd.factorize(df[hue]) 
        _xy = df[['x', 'y']].values
        powerplots.plot_colored_polygons(_xy, _c, title=title, output=output)
        plt.close()

    return 

def main(mode, 
        basepth, 
        split_lines=None, 
        rotate_theta=None,
        bounding_points=None,
        title=None,
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
    assert mode in ['preview', 'preview-save', 
                    'view', 'analysis-only', 'plot-only',  'compile-only'] # choose from these options
    logging.info(f"mode = {mode}")
    # bypassing TMG
    if mode.startswith('preview'): # 
        logging.info(f"entering {mode} mode")
        assert split_lines is not None
        # search for metadata files
        fs = glob.glob(os.path.join(basepth, f"*_metadata.csv"))
        assert len(fs) == 1
        meta_file = fs[0]
        df = pd.read_csv(meta_file)
        XY = df[['tmp_x', 'tmp_y']].values
        if rotate_theta is not None:
            XY = rotate(XY, rotate_theta)
        cond, isinpoly, XYnew = preview_hemisphere(split_lines, basepth=basepth, 
                                                    XY=XY, 
                                                    bounding_points=bounding_points,
                                                    title=title,
                                                    )

        if mode.endswith('save'):
            # subprocess.run(['chmod', '644', meta_file])
            # save it -- 
            df[['stage_x', 'stage_y']] = XYnew
            df['hemi'] = cond.astype(int)
            df['isinpoly'] = isinpoly.astype(int)
            df = df[df['isinpoly']==1] # select only those in the polygons
            df.to_csv(meta_file, index=False)
            logging.info(f"Updated {meta_file}")

            if np.sum(~isinpoly) > 0:
                # search for metadata files
                fs = glob.glob(os.path.join(basepth, f"*_matrix.csv"))
                assert len(fs) == 1
                mat_file = fs[0]
                df2 = pd.read_csv(mat_file)
                df2 = df2[isinpoly]
                df2.to_csv(mat_file, index=False)
                logging.info(f"Updated {mat_file}")

            # logging.info(f"{df.head()}")
            # subprocess.run(['chmod', '444', meta_file])
        return cond, XYnew

    # house keeping - TMGs 
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

    # branching off
    if mode == 'view': # analysis + plot
        logging.info(f"entering {mode} mode")
        df = generate_default_analysis(basepth, XY=None,
                output_df="default_analysis.csv")
        generate_default_views(df, respth, title=title)
        if compile_pdf:
            compile_tex.main(basepth, **pdf_kwargs)
        return df
    elif mode == 'analysis-only': # 
        logging.info(f"entering {mode} mode")
        df = generate_default_analysis(basepth, XY=None,
                output_df="default_analysis.csv")
        return df
    elif mode == 'plot-only': # 
        logging.info(f"entering {mode} mode")
        dfpth = os.path.join(basepth, 'default_analysis.csv')
        df = pd.read_csv(dfpth, index_col=0)
        generate_default_views(df, respth, title=title)
        if compile_pdf:
            compile_tex.main(basepth, **pdf_kwargs)
        return 
    elif mode == 'compile-only': # 
        logging.info(f"entering {mode} mode")
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

