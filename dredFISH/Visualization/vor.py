import numpy as np

from shapely.geometry import Polygon

from collections import defaultdict, Counter
from copy import copy

from scipy.spatial import Voronoi
from scipy.ndimage import binary_fill_holes, gaussian_filter

def voronoi_intersect_box(XY):
    """
    Find Voronoi points within bounding box

    Input
    -----
    XY : XY coordinates

    Output
    ------
    list : set of polygons from Voronoi diagram intersecting bounding box

    """
    _, bb = bounding_box_grid(XY)
    diameter = np.linalg.norm(bb.ptp(axis=0))
    boundary_polygon = Polygon(bb)
    
    vp = list(voronoi_polygons(Voronoi(XY), diameter))
    return [p.intersection(boundary_polygon) for p in vp]

def bounding_box_grid(XY, scale=25, padding=10, threshold=50, sd=0.1, nofill=False):
    """
    Find a bounding box over XY coordinates using a grid 
    
    Input
    -----
    XY : XY coordinates
    scale : difference between max and min values used to determine grid shape 
    padding : padding on sides of grid 
    threshold : distance from observation where bounding box is valid
    sd : standard deviation for Gaussian filter
    nofill : whether empty space within the object should be filled 

    Output
    ------
    segments : ordered line segments of bounding box
    bb : ordered coordinates of bounding box 
    """

    # construct grid using min and max values for X and Y values 
    xmax = XY.max(0)[0]
    xmin = XY.min(0)[0]
    ymax = XY.max(0)[1]
    ymin = XY.min(0)[1]

    grid = np.zeros([((xmax - xmin) / scale).astype(int) + padding,((ymax - ymin) / scale).astype(int) + padding])

    # mappings for grid indices to the XY plane
    i2x = lambda i : (i * scale + xmin - scale * padding / 2)
    j2y = lambda j : (j * scale + ymin - scale * padding / 2)

    # identify all grid elements where there is a cell within distance of grid mapping to XY 
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            x = i2x(i)
            y = j2y(j)
            if np.min(np.sum(np.abs(XY - (x, y)), 1)) > threshold:
                grid[i, j] = 1

    if not nofill:
        grid = 1 - binary_fill_holes(grid == 0)
    if sd != None:
        grid = gaussian_filter(grid, sd)

    # find all grid elements next to a grid element that contains a cell 
    border_grid = copy(grid)
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            if grid[i, j] == 0 and sum([sum([grid[k + i, l + j] for k in range(-1, 2)]) for l in range(-1, 2)]):
                border_grid[i, j] = 2

    # coordinates 
    x, y = np.where(border_grid == 2)
    xy_coords = np.array(list(zip(x, y)))

    # order coordinates 
    order = find_order(xy_coords)
    xy_coords = np.array([xy_coords[i] for i in order])

    bb = np.array([x for x in zip(list(map(i2x, xy_coords[:, 0])), list(map(j2y, xy_coords[:, 1])))])

    segments = []

    for i in range(np.shape(bb)[0] - 1):
        segments.append([list(bb[i, :]), list(bb[i + 1, :])])
    segments.append([list(bb[i, :]), list(bb[0, :])])

    return np.array(segments), bb

def find_order(xy_coords):
    """
    Use dynamic programming to find longest cycle in graph 
    
    Input
    -----
    xy_coords : XY coordinates 

    Output
    ------
    order : order of XY coordinates such that they make a cycle 
    """

    # start site 
    start = xy_coords[0]

    # helper for finding neighbors in radius r of obs
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

    # find path for each coordinate to start site 
    path_steps = {tuple(start): [0]}
    pos = [start]
    idx = 0
    while len(path_steps)!=len(xy_coords):
        cand = find_neigh(pos[idx], xy_coords, path_steps)
        for c in cand:
            if tuple(c) not in path_steps:
                idx_count = Counter(np.where(xy_coords == c)[0])
                path_steps[tuple(c)] = path_steps[tuple(pos[idx])] + [x for x in idx_count if idx_count[x] == 2]
                pos += [c]
        idx += 1

    # find longest path joining two neighbors 
    path_dict = {}
    keys = list(path_steps.keys())
    key_dict = {x: idx for idx, x in enumerate(list(path_steps.keys()))}

    for x in path_steps:
        for y in find_neigh(x, xy_coords, {}):
            y = tuple(y)
            if not np.product(x == y):
                idx = key_dict[x]
                idy = key_dict[y]
                path_dict[(idx, idy)] = len(set(path_steps[tuple(x)] + path_steps[tuple(y)]))

    max_x, max_y = max(path_dict, key = path_dict.get)
    order = path_steps[keys[max_x]] + path_steps[keys[max_y]][::-1][:-1]
    return order


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

def bounding_box(XY, bins=100):
    """
    Find bounding box from a set of XY coordinates using X values 
    
    Input
    -----
    XY : XY coodinates
    bins : number of bins for X values 

    Output
    ------
    segments : ordered line segments of bounding box
    bb : ordered coordinates of bounding box 
    """
    bb = np.zeros([2 * bins, 2])

    xmax=XY.max(0)[0]
    xmin=XY.min(0)[0]

    # x values go from min to max to min
    bb[:, 0] = list(np.arange(xmin, xmax, step = (xmax - xmin) / (bins))) + list(np.arange(xmax, xmin, step = (xmin-xmax) / (bins)))

    # find the indices within some x range and then find their min and max y values 
    for i in range(bins):
        ix = (XY[:, 0] > bb[i, 0]) & (XY[:, 0] <= bb[i + 1, 0])
        bb[i, 1] = np.min(XY[ix, 1])
        bb[2 * bins - i - 1, 1] = np.max(XY[ix, 1])

    segments = []

    for i in range(np.shape(bb)[0] - 1):
        segments.append([list(bb[i, :]), list(bb[i + 1, :])])
    segments.append([list(bb[i, :]), list(bb[0, :])])

    return np.array(segments), np.array(bb)  
