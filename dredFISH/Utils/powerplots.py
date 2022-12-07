"""Plotting utility
"""
import numpy as np
import datashader as ds
import colorcet
import json
import logging
from datetime import datetime

from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

from .__init__plots import *
from dredFISH.Utils import geomu

class PlotScale:
    """
    arguments: rangex, rangey, [npxlx, npxly, pxl_scale]
    
    one of the three in [] will be required
    """
    def __init__(self, rangex, rangey, npxlx=0, npxly=0, pxl_scale=0):
        """
        rangex(y) - range of the x(y) axis (in micron)
        pxl_scale - number of microns per pixel
        npxlx(y) - number of pixels on the x(y)axis
        """
        # 1 of the three optional args need to be set
        assert (np.array([npxlx, npxly, pxl_scale])==0).sum() == 2 
        self.rangex = rangex
        self.rangey = rangey
        
        if pxl_scale:
            self.pxl_scale = pxl_scale
            self.npxlx = int(self.rangex/self.pxl_scale)
            self.npxly = int(self.rangey/self.pxl_scale)
        if npxlx:
            assert isinstance(npxlx, int)
            self.npxlx = npxlx
            self.pxl_scale = self.rangex/self.npxlx 
            self.npxly = int(self.rangey/self.pxl_scale)
        if npxly:
            assert isinstance(npxly, int)
            self.npxly = npxly
            self.pxl_scale = self.rangey/self.npxly 
            self.npxlx = int(self.rangex/self.pxl_scale)

        self.num_pxl = self.npxlx*self.npxly
        self.pxl_scale_ux = self.rangex/self.npxlx
        self.pxl_scale_uy = self.rangey/self.npxly 
        self.check_dim()
    
    def check_dim(self):
        """
        """
        num_pixel_limit = 1e6
        assert self.npxlx > 0
        assert self.npxly > 0
        assert self.num_pxl < num_pixel_limit
        return
        
    def len2pixel(self, length):
        """
        """
        return int(length/self.pxl_scale) 
    
    def pixel2len(self, npixel):
        """
        """
        return npixel*self.pxl_scale

def savefig(fig, path):
    """
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
    return 

def savefig_autodate(fig, path):
    """
    """
    today = datetime.today().date()
    suffix = path[-3:]
    assert suffix in ['pdf', 'png', 'jpg']
    path = path.replace(f'.{suffix}', f'_{today}.{suffix}')
    savefig(fig, path)
    print(f"saved the figure to: {path}")
    return 

def rgb_to_hex(r, g, b):
    """
    """
    assert max(r,g,b) <= 256
    return ('#{0:02X}{1:02X}{2:02X}').format(r, g, b)

def hex_to_rgb(hex_string):
    """
    """
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)

def plot_hybrid_mat_mask(_mat, _mask, axs):
    """show matrix and masks in left and right halves
    """
    m, n = _mat.shape

    ax = axs[0]
    ax.imshow(_mat[:,:int(n/2)])
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1]
    ax.imshow(_mask[:,:int(n/2)][:,::-1])
    # ax.imshow(_mask[:,int(n/2):])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    return
    
# For dredFISH
def plot_basis_spatial(df, xcol='x', ycol='y', pmode='full', vmin=-3, vmax=3, 
    title=None,
    output=None):
    if pmode == 'full':
        nx, ny = 6, 4
        panel_x, panel_y = 6, 5
        wspace, hspace = 0.05, 0
        title_loc = 'left'
        title_y = 0.9
    elif pmode == 'left_half':
        nx, ny = 6, 4
        panel_x, panel_y = 3, 5
        wspace, hspace = 0.05, 0
        title_loc = 'left'
        title_y = 0.9
    elif pmode == 'right_half':
        nx, ny = 6, 4
        panel_x, panel_y = 3, 5
        wspace, hspace = 0.05, 0
        title_loc = 'right'
        title_y = 0.9
    else:
        raise ValueError("No such mode")

    P = PlotScale(df[xcol].max()-df[xcol].min(), 
                  df[ycol].max()-df[ycol].min(),
                  npxlx=300,
                #   pxl_scale=20,
                )
    logging.info(f"Num pixels: {(P.npxlx, P.npxly)}")

    fig, axs = plt.subplots(ny, nx, figsize=(nx*panel_x, ny*panel_y))
    for i in range(24):
        if f'b{i}' not in df.columns:
            continue
        ax = axs.flat[i]
        aggdata = ds.Canvas(P.npxlx, P.npxly).points(df, xcol, ycol, agg=ds.mean(f'b{i}'))
        ax.imshow(aggdata, origin='lower', aspect='equal', cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(f'b{i}', loc=title_loc, y=title_y)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    if title is not None:
        fig.suptitle(title)
    if output is not None:
        savefig_autodate(fig, output)
        logging.info(f"saved: {output}")
    plt.show()

def plot_basis_umap(df, 
    title=None,
    output=None):
    x, y = 'umap_x', 'umap_y'
    P = PlotScale(df[x].max()-df[x].min(), 
                  df[y].max()-df[y].min(),
                  npxlx=300,
                  )
    logging.info(f"Num pixels: {(P.npxlx, P.npxly)}")

    nx, ny = 6, 4
    fig, axs = plt.subplots(ny, nx, figsize=(nx*5, ny*4))
    for i in range(24):
        ax = axs.flat[i]
        aggdata = ds.Canvas(P.npxlx, P.npxly).points(df, x, y, agg=ds.mean(f'b{i}'))
        ax.imshow(aggdata, origin='lower', aspect='equal', cmap='coolwarm', vmin=-3, vmax=3, interpolation='none')
        ax.set_title(f'b{i}', loc='left', y=0.9)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if title is not None:
        fig.suptitle(title)
    if output is not None:
        savefig_autodate(fig, output)
        logging.info(f"saved: {output}")
    plt.show()
    
def plot_type_spatial_umap(
    df, hue, 
    x='x', y='y', 
    umap_x='umap_x', umap_y='umap_y', 
    title=None,
    output=None
    ):
    """
    """
    hue_order = np.sort(np.unique(df[hue]))
    ntypes = len(hue_order)

    fig, axs = plt.subplots(1, 2, figsize=(8*2,6))
    fig.suptitle(f"{hue}; n={ntypes}")
    ax = axs[0]
    sns.scatterplot(data=df, x=x, y=y, 
                    hue=hue, hue_order=hue_order, 
                    s=0.5, edgecolor=None, 
                    legend=False,
                    rasterized=True,
                    ax=ax)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1), ncol=5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax = axs[1]
    sns.scatterplot(data=df, x=umap_x, y=umap_y, 
                    hue=hue, hue_order=hue_order, 
                    s=0.5, edgecolor=None, 
                    legend=False,
                    rasterized=True,
                    ax=ax)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1), ncol=5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(wspace=0)
    if title is not None:
        fig.suptitle(title)
    if output is not None:
        savefig_autodate(fig, output)
        logging.info(f"saved: {output}")
    plt.show()

def prep_polygons(XY, label_ids, cmap='tab20'):
    """
    """
    unq_ids = np.unique(label_ids)
    ncolors = len(unq_ids)
    colors = sns.color_palette(cmap, ncolors)
    cmap = ListedColormap(colors)
    
    xmin, ymin = np.min(XY, axis=0)
    xmax, ymax = np.max(XY, axis=0)

    vor = Voronoi(XY, furthest_site=False)
    poly = np.array([vor.vertices[vor.regions[item]] for item in vor.point_region], dtype=object)

    # filter poly by max span
    maxspans = []
    for item in poly:
        if len(item) > 0:
            maxspan = max(np.max(item, axis=0) - np.min(item, axis=0))
        else:
            maxspan = 0
        maxspans.append(maxspan)
    maxspans = np.array(maxspans)
    th = np.percentile(maxspans, 99)
    
    # prep polygons
    p = PolyCollection(poly[maxspans<th], #[:5], #geom['poly'], 
                       cmap=cmap,
                       edgecolors='none',
                      )
    p.set_array(label_ids[maxspans<th]) #[:5])
    
    return p, (xmin, xmax, ymin, ymax)

def plot_colored_polygons(XY, c, title=None, output=None):
    """
    """
    fig, ax = plt.subplots(figsize=(10,8))
    p, bbox = prep_polygons(XY, c)
    ax.add_collection(p) 
    ax.set_xlim(bbox[:2])
    ax.set_ylim(bbox[2:])
    ax.set_aspect('equal')
    ax.grid(False)
    if title is not None:
        fig.suptitle(title)
    if output is not None:
        savefig_autodate(fig, output)
        logging.info(f"saved: {output}")
    plt.show()