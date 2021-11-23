import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict

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
    Find bounding box from a set of XY coordinates 
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
