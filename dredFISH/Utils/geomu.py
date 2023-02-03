import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from copy import copy

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu 
import skimage

from scipy.ndimage import binary_fill_holes, uniform_filter
from scipy.spatial import Voronoi
from scipy.signal import convolve2d
import skimage.morphology
from skimage.transform import rescale
import scipy.ndimage.measurements

from scipy.spatial import Delaunay, Voronoi

import math

import igraph

import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.ops import voronoi_diagram
import shapely.geometry
import shapely.validation
import shapely.ops

import rasterio
import rasterio.features

import os.path 
import glob

from matplotlib.collections import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.path import Path

import pdb

# Commented this for now, replaced with create_voronoi
# keeping it ghere commented till more testing will be done on create_voronoi
def voronoi_polygons(XY):
    """
    Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.
    
    https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region
    
    Input
    -----
    voronoi : SciPy Voronoi object 
    diameter : bound for infinite polygons 
    
    Output
    ------
    list : Polygon objects that make up Voronoi diagram
    """    
    voronoi = Voronoi(XY)
    xmax = XY.max(0)[0]
    xmin = XY.min(0)[0]
    ymax = XY.max(0)[1]
    ymin = XY.min(0)[1]
    diameter = ((xmax-xmin)**2+(ymax-ymin)**2)**0.5

    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    vor_polygons = []
    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            vor_polygons.append(Polygon(voronoi.vertices[region]))
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        vor_polygons.append(Polygon(np.concatenate((finite_part, extra_edge))))
       
    return vor_polygons


def mask_voronoi(vor_polys,mask_polys):
    """
    masks voronoi polygons based on masks. Both inputs are lists
    """
    # First, find which subset of polygons are on the boundary and need to be trimmed
    vrts = get_polygons_vertices(vor_polys)[0]
    all_vrts = np.vstack(vrts)
    vor_ids = list()
    for i in range(len(vrts)):
        vor_ids.append(np.ones(vrts[i].shape[0])*i)
    vor_ids = np.hstack(vor_ids)
    bnd = points_in_polygons(mask_polys,all_vrts)
    bnd = np.any(bnd,axis=1)
    bnd_vor_ids = np.unique(vor_ids[np.logical_not(bnd)])

    mask_polys = shapely.geometry.MultiPolygon(mask_polys)
    # after intersections, there are cases where shapely doesn't return a single polygons
    # the code below deals with it by checking for edge cases: MultiPolygon, GeometryCollection    
    for pix in bnd_vor_ids:
        pix = int(pix)
        v = vor_polys[pix]
        vp = v.intersection(mask_polys)
        if isinstance(vp,shapely.geometry.MultiPolygon):
            allparts = list(vp.geoms)
            areas = np.array([p.area for p in allparts])
            vor_polys[pix] = allparts[np.argmax(areas)]
        elif isinstance(vp,shapely.geometry.collection.GeometryCollection):
            # first get only the polygons from this geometry collection
            g_ispolygon = np.array([isinstance(g,shapely.geometry.polygon.Polygon) for g in list(vp.geoms)])
            allparts = np.array(list(vp.geoms))
            allparts = allparts[g_ispolygon]
            # then chose the largest one
            areas = np.array([p.area for p in list(allparts)])
            vor_polys[pix] = allparts[np.argmax(areas)]
        else:
            vor_polys[pix] = vp

    return vor_polys


def create_mask(lbl,scl = 0.1,rds = 150,flip = False):
    """
    Create a mask that includes ALL cells, returns a list of polygons

    Input
    -----
    lbl : a labeled matrix of all cells
    scl : to speed calculation will scale image by scl first down and then up by 1/scl
    rds : radius of disk to extend around all cells (in original pixel units)
    """
    msk = lbl>0
    # rescale down to ease computation
    msk_sml = rescale(msk,scl)
    k = skimage.morphology.disk(rds*scl)
    msk_dil = convolve2d(msk_sml, k, mode='same', boundary='symm')
    msk_dil = msk_dil>0
    lbl_dil = scipy.ndimage.measurements.label(msk_dil)
    lbl_dil = lbl_dil[0]
    # rescale up using nearest neighbors (order = 0) to preserve labels
    lbl_dil  = rescale(lbl_dil,1/scl,order=0) 
    mask_polygons = vectorize_labeled_matrix_to_polygons(lbl_dil.astype('int32'))

    if flip:
        msk_flipped = list()
        for i,p in enumerate(mask_polygons): 
            msk_flipped.append(shapely.ops.transform(lambda x, y: (y, x), p))
        mask_polygons = msk_flipped

    return mask_polygons

def vectorize_labeled_matrix_to_polygons(imgmat,tolerance = 2):
    """
    """

    # First, we use rasterio to generate all polygons. It is possible that cells are represented 
    # as non-continous in the raster image. So number of polygons rasterio generates is not necessarily 
    # the same as the number of cells. 
    polygons = []
    ids = []
    # assuming here that if input it not a numpy array it must be a pytorch tensor so converting to numpy
    if not isinstance(imgmat, np. ndarray): 
        imgmat = imgmat.numpy() 
    mask = imgmat!=0 
    for shape, value in rasterio.features.shapes(imgmat,mask = mask,connectivity = 8):
        polygons.append(shapely.geometry.shape(shape))
        ids.append(value)
    polygons = np.asarray(polygons,dtype="object")
    ids = np.array(ids,dtype='int32')

    # Once we have the polygons, we want to merge them to that each cell gets one polygons
    # this requires lookping over all polygons and "fixing" (merging) them
    # the code works on the (unique sorted) ids one at a time, so it also make sure we return 
    # the list of polygons according to ids (unique sorted) order. 
    unq = np.unique(ids) 
    unq_poly = []
    verts = []
    max_iter = 5
    for i in range(len(unq)):
        ix = np.flatnonzero(ids==unq[i])
        multipoly=shapely.geometry.MultiPolygon(list(polygons[ix]))
        if not multipoly.is_valid:
            multipoly = shapely.validation.make_valid(multipoly)
        poly = shapely.ops.unary_union(multipoly)
        # it is possible that poly is not really a polygon as union can't work if polygons are seperated. 
        # to deal with this the code does 2 things:  
        # 1. try to buffer(1) up to max_iter times and see if the union becomes a single polygon 
        # 2. choose the largest of the polygons in the multipolygon. 
        cnt=0
        while cnt<max_iter and isinstance(poly,shapely.geometry.multipolygon.MultiPolygon):
            poly.buffer(1)
            poly = shapely.ops.unary_union(poly)
            cnt=cnt+1
        # if the first solutions doesn't work, just get the largest polygon by area
        if isinstance(poly,shapely.geometry.multipolygon.MultiPolygon):
            allparts = [p.buffer(0) for p in poly.geoms]
            areas = np.array([p.area for p in poly.geoms])
            poly = allparts[np.argmax(areas)]

        poly.simplify(tolerance)
        unq_poly.append(poly)

    return unq_poly

def get_polygons_vertices(polygons):
    """
    simple utility to create list of verticies (that can be drawn with matplotlib PolygonColleciton)
    from alist of shapely polygons. 
    The methods returns a tuple of exterior vertices for all polygons (one per polygon) and another list of all the holes. 
    It also returns a list of lists for the "holes", i.e. inner vertices. 
    The size of lists within the main list is variable and depends on data input. 
    """ 
    xy=[]; 
    verts=[];
    inner_verts = []; 
    for i,poly in enumerate(polygons):
        xy = poly.exterior.xy
        verts.append(np.array(xy).T)
        inner_verts.append(list())
        for LinearRing in poly.interiors:
            xi,yi = LinearRing.xy
            xy = np.transpose(np.vstack((xi,yi)))
            inner_verts[i].append(xy)

    return (verts,inner_verts)

def plot_polygon_collection(verts_or_polys,
                            rgb_faces = None, # one per polygon
                            rgb_edges = None, # 
                            ax = None,
                            xlm = None,
                            ylm = None,
                            background_color = (1,1,1), # defaults to white background
                            transpose = False):
    """
    utility function that gets vertices (list of XYs) or polygons (list of shapley polygons)
    and plots them using matplotlib PolygonCollection. If polygons have holes, will "fake it" and plot the holes on top of the 
    existing polygons with background color (defaults to white rgb=(1,1,1))
    """

    # First - convert verts_or_polys to just be verts
    if isinstance(verts_or_polys[0],shapely.geometry.Polygon): 
        (verts,hole_verts) = get_polygons_vertices(verts_or_polys)
    else: 
        verts = verts_or_polys[0]
        hole_verts = verts_or_polys[1]

    if transpose:
        verts = [np.fliplr(v) for v in verts]

    # verify rgb inputs and deal with cases where we only want edges
    assert rgb_edges is not None or rgb_faces is not None,"To plot either pleaes provide RGB array (nx3) for either edges or faces "
    if rgb_edges is not None and rgb_faces is None:
        bck_color = np.array(background_color)
        rgb_faces = np.repeat(bck_color.T,len(verts),0)

    # Create the PolyCollection from vertices and set face/edge colors
    p = PolyCollection(verts)
    p.set(array=None, facecolors=rgb_faces,edgecolors = rgb_edges)
    if ax is None: 
        fig = plt.figure(figsize = (8,10))
        ax = fig.add_subplot()
    ax.add_collection(p)

    # merge the different inner polygons as they will all be plotted with the same color
    all_inr = list()
    for i in range(len(hole_verts)):
        for j in range(len(hole_verts[i])):
            all_inr.append(hole_verts[i][j])

    # Create a second PolyCollection for the holes
    p2 = PolyCollection(all_inr)
    p2.set(array=None, facecolors=background_color,edgecolors = background_color)
    ax.add_collection(p2)

    # identify boundaries for ax
    xy = np.vstack(np.array(verts))
    mx = np.max(xy,axis=0)
    mn = np.min(xy,axis=0)
    if xlm is None: 
        xlm = (mn[0],mx[0])
    if ylm is None: 
        ylm = (mn[1],mx[1])

    ax.set_xlim(xlm[0],xlm[1])
    ax.set_ylim(ylm[0],ylm[1])

    ax.set_aspect('equal', 'box')
    ax.axis('off')

def plot_point_collection(pnts,sizes,rgb_faces = None,rgb_edges = None,ax = None,xlim=None,ylm = None):
    # Create the PollyCollection from vertices and set face/edge colors
    assert rgb_edges is not None or rgb_faces is not None,"To plot either pleaes provide RGB array (nx3) for either edges or faces "

    # get XY from points
    xy = [(p.x,p.y) for p in pnts]
    xy = np.array(xy)

    if ax is None: 
        fig = plt.figure(figsize = (8,10))
        ax = fig.add_subplot()

    CC = CircleCollection(sizes,offsets=xy,transOffset=ax.transData)
    CC.set(array=None, facecolors=rgb_faces,edgecolors = rgb_edges)

    ax.add_collection(CC)
    ax.set_xlim(xlm[0],xlm[1])
    ax.set_ylim(ylm[0],ylm[1])

    ax.set_aspect('equal', 'box')
    ax.axis('off')

    mx = np.max(xy,axis=0)
    mn = np.min(xy,axis=0)
    if xlm is None: 
        xlm = (mn[0],mx[0])
    if ylm is None: 
        ylm = (mn[1],mx[1])

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

def load_section_geometries(unqS,basepath):
    """
    Loads slices geometries from a given folder (basepath) for all unique slices in unqS
    It assumes that the files are wkt files and that their pattern is: /basepath/Geom_{geom_type}_Section_{section_num}.wkt

    """
    Geoms = list()
    # read the wkt files to figure out which geoms we have
    wkt_files = glob.glob(basepath + '/*.wkt', recursive=False)
    geom_types = set()
    for i,wf in enumerate(wkt_files): 
        wf_file = wf.split('/')[-1] 
        parts = wf_file.split('_')
        geom_types.add(parts[1])
    geom_types=list(geom_types)

    # look over all slices and geom_types to load all geom types
    for i,s in enumerate(unqS):
        # adds an empty dict
        Geoms.append(dict())
        for j,gt in enumerate(geom_types):
            fname = "Geom_" + gt + "_Section_" + str(int(s)) + ".wkt"
            fname = os.path.join(basepath,fname)
            Geoms[i][gt] = {'poly' : list()}
            with open(fname) as file:
                for line in file:
                    wktstr = line.rstrip()
                    gm = shapely.wkt.loads(wktstr)
                    Geoms[i][gt]['poly'].append(gm)
            if gt == 'point':
                Pnts = shapely.geometry.MultiPoint(Geoms[i][gt]['poly'])
                Geoms[i][gt]['vert'] = np.vstack([np.array(p.xy).T for p in list(Pnts.geoms)])
            else: 
                Geoms[i][gt]['vert'] = get_polygons_vertices(Geoms[i][gt]['poly'])
    return Geoms

def swap_mask(mat, lookup_o2n):
    """
    create from the old mask matrix a new matrix with the swapped labels according to the lookup table (pd.Series)
    """
    i, j = np.nonzero(mat)
    unq, inv = np.unique(mat[i,j], return_inverse=True)
    # assert np.all(unq[inv] == mat[i,j]) # unq[inv] should recreates the original one
    
    newmat = mat.copy()
    newmat[i,j] = lookup_o2n.loc[unq].values[inv]
    return newmat



def points_in_polygons(polys,XY,check_holes = False):
    """
    for a list of polygons check if points XY are in each of the polygons
    by defaults it ignores holes, as they add a lot of runtime

    returns a matrix of size NxM with N = XY.shape[0] and M is len(polys) 
    """
    result = np.ones((XY.shape[0],len(polys)),dtype=bool)
    ext_verts,in_verts = get_polygons_vertices(polys)
    for i,p in enumerate(polys):
        # first find if points are in the external polygon
        path = Path(ext_verts[i])
        result[:,i] = path.contains_points(XY)

        if check_holes:
            # now exclude innter points
            for j,iv in enumerate(in_verts):
                path = Path(iv)
                result[:,i] = np.logical_and(result[:,i],np.logical_not(path.contains_points(XY)))
    
    return result


def spatial_graph_from_XY(XY,max_dist = 300):
    """
    Create spatial graph from XY, allow edges up to max_dist from each other. 
    """
    dd = Delaunay(XY)

    # create Graph from edge list
    EL = np.zeros((dd.simplices.shape[0]*3,2),dtype=np.int64)
    for i in range(dd.simplices.shape[0]): 
        EL[i*3,  :] = [dd.simplices[i,0], dd.simplices[i,1]]
        EL[i*3+1,:] = [dd.simplices[i,0], dd.simplices[i,2]]
        EL[i*3+2,:] = [dd.simplices[i,1], dd.simplices[i,2]]

    # calcualte distances along edges and keep only ones that are less tham max_dist
    edge_spatial_dist  = np.sqrt((XY[EL[:,0],0]-XY[EL[:,1],0])**2 + 
                                 (XY[EL[:,0],1]-XY[EL[:,1],1])**2)

    EL = EL[edge_spatial_dist<max_dist,:]
            
    # update vertices numbers to account for previously added nodes (cnt)
    SG = igraph.Graph(n=XY.shape[0], edges=EL, directed=False).simplify()

    return SG


def in_graph_large_connected_components(XY,Section = None,max_dist = 300,large_comp_def = 0,plot_comp = False):
    """
    Checks components size for neighbor graph given by XY, and max_dist
    returns components id and if large_comp_def is not None a boolean to say it it should be included
    for all cells using spatial graph analysis
    """
    if Section is None: 
        Section = np.ones((XY.shape[0],))
    
    unqS = np.unique(Section)
    all_in_large_comp = list()
    all_cnt = list()
    for i,s in enumerate(unqS): 
        SG = spatial_graph_from_XY(XY[Section == s,:],max_dist = max_dist)
        cmp = np.asarray(SG.components().membership)
        unq_cmp,cnt = np.unique(cmp,return_counts=True)
        cnt = cnt/cnt.sum()
        in_large_comp = np.isin(cmp,unq_cmp[cnt>large_comp_def])
        all_in_large_comp.append(in_large_comp)
        all_cnt.append(cnt)

    in_large_comp = np.hstack(all_in_large_comp)

    if plot_comp:
        if len(unqS) == 1: #single section, make sense to plot XY
            fig,ax = plt.subplots(1,2,figsize=(12,10))
            ax[0].scatter(XY[:,0],XY[:,1],c = cmp,s = 0.1,cmap = 'jet')
            clr = cm.jet(np.arange(0,1,len(unq_cmp)))
            ax[0].set_xticks = []
            ax[0].set_yticks = []
            ax[0].set_aspect('equal', 'box')
            ax[1].bar(unq_cmp,cnt,color = clr)
            ax[1].set_ylabel("Number of cells per mask")
            ax[1].set_xlabel("masks")
        else: # multi-section, just show fraction of filtered (subplot(1,2,1)) and fractions in large components
            fig,ax = plt.subplots(1,2,figsize=(16, 8))
            filt_per_sec = [1-x.mean() for x in all_in_large_comp]
            ax[0].bar(np.arange(len(unqS)),filt_per_sec)
            ax[0].set_xticks(np.arange(len(unqS)));
            ax[0].set_xticklabels(unqS);
            ax[0].set_xlabel("Section")
            ax[0].set_ylabel("Fraction of filtered cells")

            all_cnt_filt = [x[x > 0.05] for x in all_cnt]
            sz = np.full([len(all_cnt), np.max([len(x) for x in all_cnt_filt])], np.nan)
            for i in range(len(all_cnt_filt)):
                sz[i,0:len(all_cnt_filt[i])] = all_cnt_filt[i]
            df = pd.DataFrame(sz)
            df['Section'] = unqS
            df.plot.bar(x='Section', stacked=True, title='Fraction of cells in components (filtered)',ax=ax[1])
            plt.gca().get_legend().remove()

    return(in_large_comp)

def merge_polygons_by_ids(polys,ids,max_buff = 1000):
    """
    merges nearby poygons based on id vectors. 
    if polygons are not really touching, it will expand them slightsly until they do. 
    """
    unq_id  = np.unique(ids)
    all_merged_polys = list()
    for j,uid in enumerate(unq_id):
        ix = np.flatnonzero(ids == uid)
        poly_list = list()
        for i in range(len(ix)):
            poly_list.append(polys[ix[i]])
        merged_poly = shapely.ops.unary_union(poly_list)
        all_merged_polys.append(merged_poly)
    
    # expands polygons until they touch and merge (convetrs multi to poly)
    not_reg_poly = np.array([not isinstance(mp,shapely.geometry.polygon.Polygon) for mp in all_merged_polys])
    not_reg_poly_ix = np.flatnonzero(not_reg_poly)
    for i,ix in enumerate(not_reg_poly_ix):
        mp = all_merged_polys[ix]
        bf=0
        while not isinstance(mp,shapely.geometry.polygon.Polygon) and bf < max_buff:
            bf+=1
            mp = mp.buffer(bf)
        if bf == max_buff:
            raise ValueError("Issue with merging polygons, they are too far apart")
        all_merged_polys[ix] = mp
    
    return all_merged_polys

