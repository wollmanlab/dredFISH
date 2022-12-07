import numpy as np

from collections import defaultdict, Counter
from copy import copy

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu 

from scipy.ndimage import binary_fill_holes, uniform_filter

from shapely.geometry import Polygon
from shapely.ops import unary_union

import pdb

def voronoi_polygons(voronoi, diameter):
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

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
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
        yield Polygon(np.concatenate((finite_part, extra_edge)))


def find_neigh(pos, xy_coords, redun, r=1):
    """
    Find neighbors within r radius excluding some set

    Input
    -----
    pos : position 
    xy_coords : XY coordinates
    redun : redundant set for exclusion 
    r : radius 

    Output
    ------
    list : neighbors in radius 
    """
    cand = np.array([[(pos[0] + i, pos[1] + j) for i in range(0 - r, 1 + r)] for j in range(0 - r, 1 + r)])
    cand=[x for y in cand for x in y]
    return [x for x in cand if 2 in np.sum(xy_coords == x, 1) if tuple(x) not in redun]

def find_order(xy_coords, path_steps):
    """
    Find order of coordinates given that some subset have been traversed already

    Input
    -----
    xy_coords : XY coordinates
    path_steps : tracking previous positions walk to some start

    Output:
    order cycle through coordinates 
    """

    # arbitrary start that ideally avoids inner cycles on first iteration 
    index = np.where(xy_coords[:, 0] == np.min(xy_coords, axis=0)[0])[0]
    index = index[np.argmin(xy_coords[index, :], axis=0)[1]]

    start = xy_coords[index]
    path_steps[tuple(start)] = [index]
    pos = [start]
    idx = 0
    # once we have run out of candidates, we have covered all cells within a cycle 
    while len(pos) > idx:
        cand = find_neigh(pos[idx], xy_coords, path_steps)
        for c in cand:
            if tuple(c) not in path_steps:
                idx_count = Counter(np.where(xy_coords == c)[0])
                path_steps[tuple(c)] = path_steps[tuple(pos[idx])] + [x for x in idx_count if idx_count[x] == 2]
                pos += [c]
        idx += 1

    # find set of cells connecting two neighbors to start
    path_dict = {}
    keys = [tuple(list(x)) for x in pos]
    key_dict = {x: idx for idx, x in enumerate(keys)}

    for x in pos:
        x = tuple(list(x))
        for y in find_neigh(x, xy_coords, {}):
            y = tuple(y)
            if not np.product(x == y):
                idx = key_dict[x]
                idy = key_dict[y]
                # if two paths are distinct, then the set of their combined elements will be relatively large 
                path_dict[(idx, idy)] = len(set(path_steps[tuple(x)] + path_steps[tuple(y)]))

    # empty cell
    if len(path_dict) == 0:
        return path_steps[keys[0]]

    # find longest path 
    max_x, max_y = max(path_dict, key = path_dict.get)

    order = path_steps[keys[max_x]] + path_steps[keys[max_y]][::-1][:-1]  

    # find longest substring with no duplicates 
    if len(order) != len(set(order)):
        val, ct = np.unique(order, return_counts=True)
        dupl = np.where([True if x in val[ct > 1] else False for x in order])[0]
        order = order[dupl[np.argmax(np.diff(dupl))]: dupl[np.argmax(np.diff(dupl)) + 1]]

    return order 

def find_hist_mapping(hist, padding):
    """
    Adjust histogram for padding

    Input
    -----
    hist : 2D histogram result
    padding : padding quantity 

    Output
    ------
    coordinates from bin indexes to space 
    """
    # construct mapping of grid with padding back to coordinates
    # 2D hist contains three dimensions: 2D counts and x/y coords
    hist2coord = {}

    # inner points, average between bins 
    for i in range(hist[0].shape[0]):
        for j in range(hist[0].shape[1]):
            hist2coord[(i + padding, j + padding)] = ((hist[1][i] + hist[1][i + 1]) / 2, (hist[2][j] + hist[2][j + 1]) / 2)

    # [0, padding] x difference, inner y 
    for i in range(padding)[::-1]:
        for j in range(hist[0].shape[1]):
                a = hist2coord[(i + 1, j + padding)]
                b = hist2coord[(i + 2, j + padding)]
                hist2coord[(i, j + padding)] = np.array(a) + np.array(a) - np.array(b)

    # [xrange + padding, xrange + padding * 2] x difference, inner y 
    for i in range(hist[0].shape[0], hist[0].shape[0] + padding * 2):
        for j in range(hist[0].shape[1]):
                a=hist2coord[(i - 1, j + padding)]
                b=hist2coord[(i - 2, j + padding)]
                hist2coord[(i, j + padding)] = np.array(a) + np.array(a) - np.array(b)

    # all of x, [0, padding] y difference
    for i in range(hist[0].shape[0] + padding * 2):
        for j in range(padding)[::-1]:
            a=hist2coord[(i, j + 1)]
            b=hist2coord[(i, j + 2)]
            hist2coord[(i, j)] = np.array(a) + np.array(a) - np.array(b)

    # all of x, [yrange + padding, yrange + padding * 2] y difference
    for i in range(hist[0].shape[0] + padding * 2):
        for j in range(hist[0].shape[1], hist[0].shape[1] + padding * 2):
            a = hist2coord[(i, j - 1)]
            b = hist2coord[(i, j - 2)]
            hist2coord[(i, j)] = np.array(a) + np.array(a) - np.array(b)
    return hist2coord

def bounding_box(XY, n_xbins=50, n_ybins=50, padding=10, size_thr=8, small=100, fill_holes=True, hole_thr=10):
    """
    Find bounding box with or without holes
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
    grid = 1 - uniform_filter(mask_rso.astype(float), size=size_thr)
    otsu_threshold = threshold_otsu(grid)
    grid = grid < otsu_threshold

    if fill_holes:
        grid = binary_fill_holes(grid)

    # pad the grid 
    grid_pad = np.zeros(grid.shape + np.array((padding * 2, padding * 2)))
    grid_pad[padding: (grid_pad.shape[0] - padding), padding : (grid_pad.shape[1] - padding)] = grid
    grid = grid_pad

    # detect boundary points 
    border_grid = copy(grid)
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            if grid[i, j] == 0 and sum([sum([grid[k + i, l + j] for k in range(-1, 2)]) for l in range(-1, 2)]):
                border_grid[i, j] = 2
    
    # we need to sort all of the border points into cycles
    # if fill_holes == True, then this is easy, there is just one cycle
    # if otherwise, then we have to iteratively find cycles from the remaining cells
    x, y = np.where(border_grid == 2)
    xy_coords = np.array(list(zip(x, y)))
    cycles=[]
    path_steps={}

    # global stopping condition is when we have covered distance from start(s) to every cell
    while len(path_steps)!=len(xy_coords):
        order = find_order(xy_coords, path_steps)
        # add to cycle set and update coords to remove cells from cycle
        cycles += [(order, np.array([xy_coords[i] for i in order]))]
        xy_coords = np.array([list(xy) for xy in xy_coords if tuple(xy) not in path_steps])

        if len(xy_coords) == 1:
            cycles += [([0], xy_coords)]
            break
        if len(xy_coords) == 0:
            break

    hist2coord = find_hist_mapping(hist, padding)
    bb = list(map(lambda xy_coords: np.array([hist2coord[tuple(x)] for x in xy_coords]), [x[1] for x in cycles]))
    bb = [x for x in bb if len(x) >= hole_thr]
    diameter = np.linalg.norm(bb[0].ptp(axis=0))

    return diameter, Polygon(bb[0]) - unary_union([Polygon(bb[i]) for i in range(1, len(bb))])
