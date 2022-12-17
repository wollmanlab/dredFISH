import numpy as np

from collections import defaultdict, Counter
from copy import copy

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu 
import skimage

from scipy.ndimage import binary_fill_holes, uniform_filter
from scipy.spatial import Voronoi

import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import shapely.geometry
import shapely.validation
import shapely.ops

import rasterio
import rasterio.features


from matplotlib.collections import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib.cm as cm

import pdb

def voronoi_polygons(XY,mask_polygons):
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

        # fix polygons that were split into multiples parts by taking largest one
        # mask input is given as list of polygons, make it into a MultiPolygon collection 
        # so that it is easier to use intersection
        mask_polygons = shapely.geometry.MultiPolygon(mask_polygons)
        vor_polygons = [p.intersection(mask_polygons) for p in vor_polygons]
        
        for i in range(len(vor_polygons)):
            if isinstance(vor_polygons[i],shapely.geometry.MultiPolygon):
                allparts = [p.buffer(0) for p in vor_polygons[i].geoms]
                areas = np.array([p.area for p in vor_polygons[i].geoms])
                vor_polygons[i] = allparts[np.argmax(areas)]
        
        return vor_polygons

def create_mask(XY, n_xbins=50, n_ybins=50, padding=10, size_thr=8, small=100, fill_holes=True, hole_thr=10):
    """
    Create a mask (shapley MultiPolygon) with or without holes
    Input
    -----
    XY : xy coordinates
    n_xbins : number of X bins in histogram
    n_ybins : number of Y bins in histogram
    padding : padding to bounding box
    size_thr : size of uniform convolution 
    small : size of small objects to exclude
    fill_holes : whether to fill holes
    hole_thr : circumference threshold for holes 

    Output
    ------ 
    diamter : diameter of bounding box
    polygon : polygon bounding box
    """

    xmax = XY.max(0)[0]
    xmin = XY.min(0)[0]
    ymax = XY.max(0)[1]
    ymin = XY.min(0)[1]

    xbins = np.arange(xmin, xmax, n_xbins)
    ybins = np.arange(ymin, ymax, n_ybins)

    # 2D histogram
    hist = np.histogram2d(XY[:, 0], XY[:, 1], bins=(xbins, ybins))

    # moved to uniform filter to detect holes
    # Otsu's method for thresholding bw 
    mask = np.array(hist[0] > 0)
    mask_rso = remove_small_objects(mask, small) * 1
    mask_sml = 1 - uniform_filter(mask_rso.astype(float), size=size_thr)
    otsu_threshold = threshold_otsu(mask_sml)
    mask_sml = mask_sml < otsu_threshold

    if fill_holes:
        mask_sml = binary_fill_holes(mask_sml)

    # resize the grid
    mask = skimage.transform.resize(mask_sml,((round(xmax-xmin),round(ymax-ymin))))

    mask_polygons = vectorize_labeled_matrix_to_polygons(mask)

    return mask_polygons

def vectorize_labeled_matrix_to_polygons(imgmat,tolerance = 2):
    """
    """

    # First, we use rasterio to generate all polygons. It is possible that cells are represented 
    # as non-continous in the raster image. So number of polygons rasterio generates is not necessarily 
    # the same as the number of cells. 
    polygons = []
    ids = []
    mask = imgmat!=0 
    for shape, value in rasterio.features.shapes(imgmat.numpy(),mask = mask.numpy(),connectivity = 8):
        polygons.append(shapely.geometry.shape(shape))
        ids.append(value)
    polygons = np.asarray(polygons,dtype="object")
    ids = np.array(ids,dtype='int32')

    # Once we have the polygons, we want to merge them to that each cell gets one polygons
    # this requires lookping over all polygons and "fixing" (merging) them
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
            poly = shapely.ops.unary_union(multipoly)
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
    from alist of shapely polygons
    """ 
    xy=[]; 
    verts=[];
    for i,poly in enumerate(polygons):
        xy = poly.exterior.xy
        verts.append(np.array(xy).T)
    return verts

def plot_polygon_collection(verts_or_polys,rgb_faces = None,rgb_edges = None,ax = None,xlm = None,ylm = None):
    """
    utility function that gets vertices (list of XYs) or polygons (list of shapley polygons)
    and plots them using matplotlib PolygonCollection
    """

    # First - convert verts_or_polys to just be verts
    if isinstance(verts_or_polys[0],shapely.geometry.Polygon): 
        verts = get_polygons_vertices(verts_or_polys)
    else: 
        verts = verts_or_polys

    # Create the PollyCollection from vertices and set face/edge colors
    assert rgb_edges is not None or rgb_faces is not None,"To plot either pleaes provide RGB array (nx3) for either edges or faces "
    p = PolyCollection(verts)
    p.set(array=None, facecolors=rgb_faces,edgecolors = rgb_edges)
    if ax is None: 
        fig = plt.figure(figsize = (8,10))
        ax = fig.add_subplot()
    ax.add_collection(p)

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
    # process / verify the range input
    if not isinstance(range, np.ndarray): 
        range = np.tile(range,(len(colormap_names,1)))
    assert range.shape[0]==len(colormap_names), "ranges dimension doesn't match colormap names"
    
    if not isinstance(colormap_names,list):
        colormap_names=[colormap_names]

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors = []
    for i,cmap_name in enumerate(colormap_names): 
        cmap = cm.get_cmap(cmap_name, res)
        colors.append(cmap(np.linspace(range[0],range[1],res)))
    colors = np.vstack(colors)
    print(colors.shape)
    mymap = LinearSegmentedColormap.from_list('my_colormap', colors)
    return mymap

