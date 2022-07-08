"""TissueMultiGraph Analysis module.

The module contains the three main classes required graph based tissue analysis: 
TissueMultiGraph: the key organizing class used to create and manage graphs across layers (hence multi)
TissueGraph: Each layers (cells, zones, regions) is defined using spatial and feature graphs that are both part ot a single tissuegraph
Taxonomy: a container class that stores information about the taxonomical units (cell types, region types).  

Note
----
In current implementation each TMG object is stored as multiple files in a single directory. 
That is the same directory that stores all the input files used by TMG to create it's different objects. 
Only a single TMG object can exist in each directory. To create two TMG objects from the same data, just copy the raw data. 

Example
-------
TMG is created in a folder with csv file(s) that have cells transcriptional state (\*_matrix.csv) and other metadata information such as x,y, section information in (\*_metadata.csv) files. 
The creation of TMG follows the methods to create different layers (create_cell_layer, create_zone_layer, create_region_layer) and creating Geoms using create_geoms. 
"""

# dependencies
from cmath import nan
import functools
from textwrap import indent
import ipyparallel as ipp
from collections import Counter
import numpy as np
rng = np.random.default_rng()
import pandas as pd
import itertools
import logging
import os.path 
import glob
import json
import pickle

import igraph
import pynndescent 
from scipy.spatial import Delaunay,Voronoi
from scipy.sparse.csgraph import dijkstra

import anndata
# to create geomtries
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString

from dredFISH.Utils import basicu 
from dredFISH.Utils import tmgu
from dredFISH.Utils.geomu import voronoi_polygons, bounding_box

# for debuding mostly
import warnings
import time
from IPython import embed

class TissueMultiGraph: 
    """Main class used to manage the creation of multi-layer graph representation of tissues. 
    
    TissueMultiGraph (TMG) acts as a factory to create objects that represents the state of different biospatial units in tissue, 
    and their relationships. Examples of biospatial units are cells, isozones, regions. 
    The main objects that TMG creates and stores are TissueGraphs, Taxonomies, and Geoms.    

    
    Attributes
    ----------
    Layers : list 
        This is the main spatial/feature data storage representing biospatial units (cells, isozones, and regions)
        
    Taxonomies : list 
        The taxonomical representation of the different types biospatial units can have. 
        There is a many-to-one relationship between TissueGraphs Layers and Taxonomies. 
        Multiple layers can have the same taxnomy (cells and isozones both have the same taxonomy). 
        The Taxonomies store type related information (full names, feature_type_mats, relationship between types, etc). 
    
    Geoms : dict
        Geometrical aspects of the TMG required for Vizualization. 
        
    basepath : str
        The base path with all TMG related files.
        
    layers_graph : list of tuples
        Stores relatioship between layers. 
        For example, [(1,2),(1,3)] inducates that layer 2 used layer 1 (isozones are build on cells) and that layer 3 (regions) uses layer 1. 
    
    layer_taxonomy_mapping : list of tuples
        Stores relationship between TG layers and Taxonomies
        
 """
    def __init__(self, basepath = None, redo = False):
        """Create a TMG object
        
        There could only be a single TMG object in basepath. 
        if one already exists (and redo is False) it will be loaded. If not a new empty one will be created. 
        
        Parameters
        ----------
            basepath : str
                The system path that stores all the data to be used.
                This is the same folder where all additional TMG files will be stored. 
                
            redo : bool (default False)
                If the object was already created in the past, the default behavior is to just load the object. 
                This can be overruled when redo is set to True. If this is the first time a TMG object is created 
            
        Raises
        ------
            basepath must exists (os.path.exists) or an error will be raised
            
        """
        
        if basepath is None or not os.path.exists(basepath):
            raise ValueError(f"Path {basepath} doesn't exist")
            
        self.basepath = basepath 
        
        # check to see if a TMG.json file exists in that folder
        # is not, create an empty TMG. 
        if redo or not os.path.exists(os.path.join(basepath,"TMG.json")):
            self.Layers = list() # a list of TissueGraphs
            self.layers_graph = list() # a list of tuples that keep track of the relationship between different layers 
            
            self.Geoms = list()
            
            self.Taxonomies = list() # a list of Taxonomies
            self.layer_taxonomy_mapping = dict() # list() # a list of tuples that keep tracks of which TissueGraph (index into Layer) 
                                                 # uses which taxonomy (index into Taxonomies
                                                 # FIXME -- has to be a dictinary 

        else: 
            # load from drive
            with open(os.path.join(basepath,"TMG.json")) as fh:
                self._config = json.load(fh)
            
            # start with Taxonomies: 
            TaxNameList = self._config["Taxonomies"]
            self.Taxonomies = [None]*len(TaxNameList)
            for i in range(len(TaxNameList)): 
                self.Taxonomies[i] = Taxonomy(TaxNameList[i])
                    
            # convert string key to int key (fixing an artifects of JSON dump and load)
            ltm = self._config["layer_taxonomy_mapping"]
            ltm = {int(layer_ix): tax_ix for layer_ix, tax_ix in ltm.items()}
            self.layer_taxonomy_mapping = ltm 
            
            LayerNameList = self._config["Layers"]
            self.Layers = [None]*len(LayerNameList)
            for i in range(len(LayerNameList)): 
                tax_ix = self.layer_taxonomy_mapping[i]
                self.Layers[i] = TissueGraph(basepath=basepath, 
                                             layer_type=LayerNameList[i], 
                                             tax=self.Taxonomies[tax_ix], 
                                             redo=False)
                
            self.layers_graph = self._config["layers_graph"]

            # at this point, not saving geoms to file so recalculate them here:             
            self.Geoms = list()
            self.add_geoms()
        return None
    
    def save(self):
        """ create the TMG.json and save everything.
        
        Saving scheme: 
        TissueGraphs in Layers and Taxonomies in Taxonomies are stored in a single anndata file per object according to theor own .save() method
        Geoms are currently not saved and are recreated on load
        mapping between layers and the names of different objects in layers (to be loaded and saved) are saved in a simple TMG.json file. 
        
        """
        self._config = {"layers_graph" : self.layers_graph, 
                       "layer_taxonomy_mapping" : self.layer_taxonomy_mapping, 
                       "Taxonomies" : [tx.name for tx in self.Taxonomies], 
                       "Layers" : [tg.layer_type for tg in self.Layers]}
        
        with open(os.path.join(self.basepath, "TMG.json"), 'w') as json_file:
            json.dump(self._config, json_file)

        # with open(os.path.join(self.basepath,"Geoms.pkl"), 'w') as pickle_file:
        #     pickle.dump(self.Geoms, pickle_file, pickle.HIGHEST_PROTOCOL)
            
        for i in range(len(self.Layers)): 
            self.Layers[i].save()
        
        for i in range(len(self.Taxonomies)): 
            self.Taxonomies[i].save(self.basepath)
            
        logging.info(f"saved")
        
    def add_type_information(self, layer_id, type_vec, tax): 
        """Adds type information to TMG
        
        Bookeeping method to add type information and update taxonomies etc. 
        
        Parameters
        ----------
        layer_id : int 
            what layers are we adding type info to? should be a `cell` layer 
        type_vec : numpy 1D array / list
            the integer codes of type 
        tax : int / Taxonomy
            either an integer that will be intepreted as an index to existing taxonomies or a Taxonomy object
        
        """
        if len(self.Layers) < layer_id or layer_id is None or layer_id < 0: 
            raise ValueError(f"requested layer id: {layer_id} doesn't exist")

        if self.Layers[layer_id].Type is not None:
            print("!! .Type already exists in that layer; return ...")
            return

        # update .Type  
        self.Layers[layer_id].Type = type_vec

        # add Tax
        # add a reference of which taxonomy index belongs to this TG
        if isinstance(tax, Taxonomy): # add to the pool and use an index to represent it
            self.Taxonomies.append(tax)
            tax = len(self.Taxonomies)-1
        self.Layers[layer_id].tax = tax

        # update mapping between layers and taxonomies to TMG (self)
        # this decides on which type to move on next...
        self.layer_taxonomy_mapping[layer_id] = tax #.append((layer_id,tax))
        return 
    
    def create_cell_layer(self, metric='cosine'): 
        """Creating cell layer from raw data. 
        TODO: Fix documentaion after finishing Taxonomy class. 
        
        Cell layer is unique as it's the only one where spatial information is directly used with Voronoi
        
        Parameters
        ----------
            
        path_to_raw_data - path to folder with all raw-data (multiple slices)
        celltypes_org - labels for cells. 
        expand_types - id of cell types to expand. If they are all the last values, numbering will be continous 
        
         
        """
        for TG in self.Layers:
            if TG.layer_type == "cell":
                print("!!`cell` layer already exists; return...")
                return 

        logging.info('In TMG.create_cell_layer')
        # load all data - create the FISHbasis and XYS variables
        logging.info('Started reading matrices and metadata')
        metadata_files = glob.glob(f"{self.basepath}//*_metadata.csv")
        metadata_files.sort()
        XYS = np.zeros((0,3))
        for i in range(len(metadata_files)): 
            df = pd.read_csv(metadata_files[i])
            xys_i = np.array(df[["stage_x","stage_y","section_index"]])
            XYS = np.vstack((XYS,xys_i))
        
        matrix_files = glob.glob(f"{self.basepath}//*_matrix.csv")
        matrix_files.sort()
        df = pd.read_csv(matrix_files[0],index_col=0)
        FISHbasis = df.to_numpy()
        cellnames = list(df.index)
        for i in range(1,len(matrix_files)):
            df = pd.read_csv(matrix_files[i],index_col=0)
            FISHbasis = np.vstack((FISHbasis,df.to_numpy()))
            cellnames = cellnames + list(df.index)
        
        logging.info('done reading files')
        # return FISHbasis
        
        # FISH basis is the raw count matrices from imaging; normalize data
        FISHbasis = basicu.normalize_fishdata(FISHbasis, norm_cell=True, norm_basis=True)
        
        # creating first layer - cell tissue graph
        TG = TissueGraph(feature_mat=FISHbasis,
                         basepath=self.basepath,
                         layer_type="cell", 
                         redo=True)
        
        # add observations and init size to 1 for all cells
        TG.node_size = np.ones((FISHbasis.shape[0],1))

        # add XY and slice information 
        TG.XY = XYS[:,0:2]
        TG.Slice = XYS[:,2]

        # build two key graphs
        logging.info('building spatial graphs')
        TG.build_spatial_graph(XYS)
        logging.info('building feature graphs')
        TG.build_feature_graph(FISHbasis, metric=metric)
        
        
        # add layer
        self.Layers.append(TG)
        logging.info('done with create_cell_layer')
        return
        
        # Add taxonomy (type) information. 
        # There are three possiblities, unsupervized, supervized, and hybrid
        
#         # completely unsupervised:
#         if celltypes_org is None: 
#             # cluster cell types optimally - all cells from scratch
#             celltypes,optres = TG.multilayer_Leiden_with_cond_entropy(return_res = True)
            
#             cell_taxonomy = Taxonomy()
#             cell_taxonomy.add_labels(feature_mat = FISHbasis,labels = celltypes)
        
#         # hybrid: 
#         elif expand_types is not None : 
#             celltypes = celltypes_org.copy()
#             mx_subtypes = 10000
#             celltypes = TG.multi_optim_Leiden_from_existing_types(base_types = celltypes,
#                                                                         types_to_expand = expand_types,
#                                                                         FeatureMat = FeatureMat,
#                                                                         max_subtypes = mx_subtypes)
#             ix_exanded_types = np.isin(celltypes_org,expand_types)
#             cell_taxonomy.add_types(feature_mat = FISHbasis[ix_exanded_types,:],labels = celltypes[ix_exanded_types])
#         # completely supervised
#         else: 
#             celltypes = celltypes_org.copy()
               
        
#         # add types and key data

    def create_isozone_layer(self, cell_layer = 0):
        """Creates isozones layer using cell types. 
        = IsoZoneLayer.tax 
        Contract cell types to create isozone graph. 
        
        """
        for TG in self.Layers:
            if TG.layer_type == "isozone":
                print("!!`isozone` layer already exists; return...")
                return 
        IsoZoneLayer = self.Layers[cell_layer].contract_graph()
        self.Layers.append(IsoZoneLayer)
        layer_id = len(self.Layers)-1                             
        self.layer_taxonomy_mapping[layer_id] = IsoZoneLayer.tax # .append((layer_id,IsoZoneLayer.tax))
        self.layers_graph.append((cell_layer,layer_id))
    
    def create_region_layer(self, topics, region_tax, cell_layer=0):
        """Add region layor given cells' region types (topics)
        
        Parameters
        ----------
        topics : numpy array / list
            An array with local type environment 
        region_tax : Taxonomy
            A Taxonomy object that contains the region classification scheme
        cell_layer : int (default,0)
            Which layer in TMG is the cell layer?          

        """
        for TG in self.Layers:
            if TG.layer_type == "region":
                print("!!`region` layer already exists; return...")
                return 

        # create region layers through graph contraction
        CG = self.Layers[cell_layer].contract_graph(topics)
        
        # contraction assumes that the feature_mat and taxonomy of the contracted layers are
        # inherited from the layer used for contraction. This is not true for regions so we need to update these
        # feature_mat is updated here and tax is updated by calling add_type_information
        Env = self.Layers[cell_layer].extract_environments(typevec = CG.Upstream)
        row_sums = Env.sum(axis=1)
        row_sums = row_sums[:,None]
        Env = Env/row_sums
        
        # create the region layer merging information from contracted graph and environments
        RegionLayer = TissueGraph(feature_mat=Env, basepath=self.basepath, layer_type="region", redo=True)
        RegionLayer.SG = CG.SG.copy()
        RegionLayer.node_size = CG.node_size.copy()
        RegionLayer.Upstream = CG.Upstream.copy()

        self.Layers.append(RegionLayer)
        current_layer_id = len(self.Layers)-1
        self.add_type_information(current_layer_id,CG.Type,region_tax)

        # update the layers graph to show that regions are created from cells
        self.layers_graph.append((cell_layer,current_layer_id))

    def fill_holes(self,lvl_to_fill,min_node_size):
        """EXPERIMENTAL: merges small biospatial units with their neighbors. 

        The goal of this method is to allow filling up "holes", i.e. chunks in the tissue graph that we 
        are unhappy about their type using local neighbor types. 

        Note
        ----
        As of 6/8/2022 this method is not ready for production use. 
        """
        update_feature_mat_flag=False
        if self.Layers[lvl_to_fill].feature_mat is not None: 
            feature_mat = self.Layers[lvl_to_fill].feature_mat.copy()
            update_feature_mat_flag = True

        # get types (with holes)
        region_types = self.Layers[lvl_to_fill].Type.copy()

        # mark all verticies with small node size
        region_types[self.Layers[lvl_to_fill].node_size < min_node_size]=-1
        fixed = region_types > -1

        # renumber region types in case removing some made numbering discontinuous
        unq,ix = np.unique(region_types[fixed],return_inverse=True)
        cont_region_types = region_types.copy()
        cont_region_types[fixed] = ix

        # map to level-0 to do the label propagation at the cell level: 
        fixed_celllvl = self.map_to_cell_level(lvl_to_fill,fixed)
        contregion_celllvl = self.map_to_cell_level(lvl_to_fill,cont_region_types)

        # fill holes through label propagation
        lblprp = self.Layers[0].SG.community_label_propagation(weights=None, initial=contregion_celllvl, fixed=fixed_celllvl)
        region_type_filled = np.array(lblprp.membership)

        # shrink indexes from level-0 to requested level-1 
        _,ix = self.map_to_cell_level(lvl_to_fill-1,return_ix=True)
        _, ix_zero_to_two = np.unique(ix, return_index=True)
        new_type = region_type_filled[ix_zero_to_two]

        # recreate the layer
        upstream_layer_ix = self.find_upstream_layer(lvl_to_fill)
        NewLayer = self.Layers[lvl_to_fill-1].contract_graph(new_type)
        self.Layers[lvl_to_fill] = NewLayer
        if update_feature_mat_flag:
            self.Layers[lvl_to_fill].feature_mat = feature_mat[fixed,:]

    def find_upstream_layer(self,layer_id):
        #TODO: remove hardcoded cell is always "root"
        upstream_layer_id=0
        return upstream_layer_id
    
    def map_to_cell_level(self,lvl,VecToMap = None,return_ix = False):
        """
        Maps values to first layer of the graph, mostly used for plotting. 
        lvl is the level we want to map all the way to 
        """
        # if VecToMap is not supplied, will use the layer Type as default thing to map to lower levels
        if VecToMap is None: 
            VecToMap = self.Layers[lvl].Type.astype(np.int64)
        elif len(VecToMap) != self.Layers[lvl].N:
            raise ValueError("Number of elements in VecToMap doesn't match requested Layer size")
            
        # if needed (i.e. not already at cell level) expand indexing backwards in layers following layers_graph
        ix=self.Layers[lvl].Upstream
        while lvl>0:
            lvl = self.find_upstream_layer(lvl)
            ix=ix[self.Layers[lvl].Upstream]
        
        VecToMap = VecToMap[ix].flatten()
        
        if return_ix: 
            return (VecToMap,ix)
        else: 
            return VecToMap
    
    def find_regions_edge_level(self, region_layer=2):
        """ finds cells graph edges edges that are also region edges
        """
        
        # create edge list with sorted tuples (the geom convention)
        edge_list = list()
        for e in self.Layers[0].SG.es:
            t=e.tuple
            if t[0]>t[1]:
                t=(t[1],t[0])
            edge_list.append(t)
        
        np_edge_list = np.asarray(edge_list)
        region_id = self.map_to_cell_level(region_layer,np.arange(self.N[region_layer]))
        np_edge_list = np_edge_list[region_id[np_edge_list[:,0]] != region_id[np_edge_list[:,1]],:]
        region_edge_list = [(np_edge_list[i,0],np_edge_list[i,1]) for i in range(np_edge_list.shape[0])]            
        return region_edge_list
    
    @property
    def N(self):
        """list : Number of cells in each layer of TMG."""
        return([L.N for L in self.Layers])
    
    @property
    def Ntypes(self):
        """list : Number of types in each layer of TMG."""
        return([L.Ntypes for L in self.Layers])
    
    def add_geoms(self):
        """Creates the geometies needed (boundingbox, lines, points, and polygons) to be used in views to create maps. 
        """
        
        # Bounding box geometry 
        allXY = self.Layers[0].XY
        Slices = self.Layers[0].Slice
        unqS = np.unique(Slices)

        for i in range(self.Layers[0].Nslices):
            slice_geoms = {}
            ix = np.flatnonzero(Slices==unqS[i])
            XY = allXY[ix,:]    
            diameter, bb = bounding_box(XY, fill_holes=False)
            slice_geoms['BoundingBox'] = bb

            # Polygon geometry 
            # this geom is just the voronoi polygons
            # after intersection with the bounding box
            # if the intersection splits a polygon into two, take the one with largest area
            vor = Voronoi(XY)
            vp = list(voronoi_polygons(vor, diameter))
            vp = [p.intersection(slice_geoms['BoundingBox']) for p in vp]
        
            verts = list()
            for i in range(len(vp)):
                if isinstance(vp[i],MultiPolygon):
                    allparts = [p.buffer(0) for p in vp[i].geoms]
                    areas = np.array([p.area for p in vp[i].geoms])
                    vp[i] = allparts[np.argmax(areas)]
            
                xy = vp[i].exterior.xy
                verts.append(np.array(xy).T)
        
            slice_geoms['poly'] = verts
        
            # Line Geometry
            # Next section deals with edge line between any two voroni polygons. 
            # Overall it relies on vor.ridge_* attributes, but need to deal 
            # with bounding box and with lines that goes to infinity
            # 
            # This geom maps to edges so things are stored using dict with (v1,v2) tuples as keys
            # the tuples are always sorted from low o high as a convension. 
            mx = np.max(np.array(slice_geoms['BoundingBox'].exterior.xy).T,axis=0)
            mn = np.min(np.array(slice_geoms['BoundingBox'].exterior.xy).T,axis=0)

            mins = np.tile(mn, (vor.vertices.shape[0], 1))
            bounded_vertices = np.max((vor.vertices, mins), axis=0)
            maxs = np.tile(mx, (vor.vertices.shape[0], 1))
            bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

            segs = list()
            center = XY.mean(axis=0)
            for i in range(len(vor.ridge_vertices)):
                pointidx = vor.ridge_points[i,:]
                simplex = np.asarray(vor.ridge_vertices[i])
                if np.all(simplex >= 0):
                    line=[(bounded_vertices[simplex[0], 0], bounded_vertices[simplex[0], 1]),
                        (bounded_vertices[simplex[1], 0], bounded_vertices[simplex[1], 1])]
                else:
                    i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                    t = XY[pointidx[1]] - XY[pointidx[0]]  # tangent
                    t = t / np.linalg.norm(t)
                    n = np.array([-t[1], t[0]]) # normal
                    midpoint = XY[pointidx].mean(axis=0)
                    far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                    line=[(vor.vertices[i,:]),(far_point)]
        
                LS = LineString(line)
                LS = LS.intersection(slice_geoms['BoundingBox'])
                if isinstance(LS,MultiLineString):
                    allparts = list(LS.intersection(slice_geoms['BoundingBox']).geoms)
                    lengths = [l.length for l in allparts]
                    LS = allparts[np.argmax(lengths)]
                    
                xy = np.asarray(LS.xy)
                line=xy.T
                segs.append(line)
            
            # make sure the keys for edges are always sorted, a convenstion that will make life easier. 
            ridge_points_sorted = np.sort(vor.ridge_points,axis=1)
            keys=list()
            for i in range(ridge_points_sorted.shape[0]):
                keys.append(tuple(ridge_points_sorted[i,:]))

            slice_geoms['line'] = dict(zip(keys, segs))
        
            # Geom Points are easy, just use XY, the order is correct :)
            slice_geoms['point'] = XY

            self.Geoms.append(slice_geoms)
class TissueGraph:
    """Representation of transcriptional state of biospatial units as a graph. 
    
    TissueGraph (TG) is the core class used to analyze tissues using Graph representation. 
    TG stores a two layer graph G = {V,Es,Ef} where Es are spatial edges (physical neighbors) and Ef are feature neighnors. 
    TG stores information on position (XYS) and features that used to created these graphs using anndata object representation.  
    Each TG has a reference to a Taxonomy object that contains information on the types ( 
    
    Note
    ----
    TissueGraph objects are typically not created on their own, but using method calls of TMG (create_{cell,isozones,regions}_layer)
    
    Attributes
    ----------
        tax : Taxonomy
            a Taxonomy object that contain labels, type stats, and relationship between types. 
        SG : iGraph
            Graph representation of the spatial relationship between biospatiual units (i.e. cells, zones, regions). 
            SG might include multiple componennts for multiple slice data and is non-weighted graph (1 - neighbors, 0 not neighbors). 
        FG : iGraph 
            Graph representation of feature similarity between biospatial units.  
               
        
    main methods:
        * contract_graph - find zones/region, i.e. spatially continous areas in the graph the same (cell/microenvironment) type
        * cond_entopy - calculates the conditional entropy of a graph given types (or uses defaults existing types)
        * watershed - devide into regions based on watershed


    """
    def __init__(self, feature_mat=None, basepath=None, layer_type=None, tax=None, redo=False):
        """Create a TissueGraph object
        
        Parameters
        ----------
        feature_mat : numpy 2D arrary or tuple of matrix size
            Matrix of the spatial units features (expression, composition, etc). 
            As alternative to the full matrix, input could be a tuple of matrix size (samples x features)
        basepath : str
            Where to read/write files related to this TG
        layer_type : str
            Name of the type of layer (cells, isozones, regions)
        """

        # validate input
        if basepath is None: 
            raise ValueError("missing basepath in TissueGraph constructor")
        if layer_type is None: 
            raise ValueError("Missing layer type information in TissueGraph constructor")
        if layer_type not in ['cell', 'isozone', 'region']:
            raise ValueError("Invalid layer type")

        # what is stored in this layer (cells, zones, regions, etc)
        self.layer_type = layer_type # label layers by their type
        self.basepath = basepath
        self.filename = os.path.join(basepath,f"{layer_type}.h5ad")
        
        # if anndata file exists and redo is False, just read the file. 
        print(self.filename)
        if not redo and os.path.exists(self.filename): 
            self.adata = anndata.read_h5ad(self.filename)
            if "SG" in self.adata.obsp.keys():
                # create SG and FG from Anndata
                sg = self.adata.obsp["SG"] # csr matrix
                self.SG = tmgu.adjacency_to_igraph(sg, directed=False) # leiden requires undirected
            if "FG" in self.adata.obsp.keys():
                fg = self.adata.obsp["FG"] # csr matrix
                self.FG = tmgu.adjacency_to_igraph(fg, directed=False)
            
        else: # create an object from given feature_mat data
            # validate input
            if feature_mat is None: 
                raise ValueError("Missing feature_mat in TisseGraph constructor")
            # if feautre_mat is a tuple, replace with NaNs
            if isinstance(feature_mat,tuple): 
                feature_mat = np.empty(feature_mat)
                feature_mat[:]=np.nan

            # The main data container is an anndata, initalize with feature_mat  
            self.adata = anndata.AnnData(feature_mat) # the tissuegraph AnnData object
            
            # Key graphs - spatial and feature based
            self.SG = None # spatial graph (created by build_spatial_graph, or load)
            self.FG = None # Feature graph (created by build_feature_graph, or load)
            
        # Taxonomy object - if it exists, provides a pointer to the object, None by default: 
        self.tax = tax
            
        # this dict stores the defaults field names in the anndata objects that maps to TissueGraph properties
        # this allows storing different versions (i.e. different cell type assignment) in the anndata object 
        # while still maintaining a "clean" interfact, i.e. i can still call for TG.Type and get a type vector without 
        # knowing anything about anndata. 
        # To see where in AnnData everything is stored, check comment in rows below 
        self.adata_mapping = {"Type": "Type", #obs
                              "node_size": "node_size", #obs
                              "name" : "name", #obs
                              "XY" : "XY", #obsm
                              "Slice" : "Slice"} #obs
        # Note: a these mapping are not used for few attributes such as SG/FG/Upstream that are "hard coded" 
        # as much as possible, the only memory footprint is in the anndata object, the exceptions are SG/FG that 
        # are large objects that we want to keep as iGraph in mem. 
        return None
    
    def is_empty(self):
        """Determines if the TG object is empty
        
        Checks if internal adata is None of empty
        """ 
        if self.adata is None or self.adata.shape[0]==0: 
            return True
        else: 
            return False
    
    def save(self):
        """save TG to file"""
        if not self.is_empty():
            self.adata.write(self.filename)
    
    @property
    def names(self):
        """list : observation names"""
        if self.is_empty():
            return None
        return self.adata.obs[self.adata_mapping["name"]]
    
    @names.setter
    def names(self,names):
        self.adata.obs[self.adata_mapping["name"]]=names
    
    @property
    def Upstream(self):
        """list : mapping between current TG layer (self) and upstream layer
        Return value has he length of upsteam level and index values of current layer""" 
        if self.is_empty():
            return None
        # Upstream is stored as ups in adata: 
        return self.adata.uns["Upstream"]
    
    @Upstream.setter
    def Upstream(self,V):
        self.adata.uns["Upstream"]=V
    
    @property
    def feature_mat(self):
        """matrix : the feature values for this TG observations
        
        The feature_mat is stored in the underlying anndata object and is required to properly init it. 
        """
        # if adata is still None, return None
        if self.is_empty():
            return None
        # otherwide, feature_mat is stored as the main data in adata
        return(self.adata.X)
    
    @feature_mat.setter
    def feature_mat(self,X):
        self.adata.X = X
    
    @property 
    def Type(self): 
        """Type
        """
        if self.is_empty():
            return None
        elif self.adata_mapping["Type"] not in self.adata.obs.columns.values.tolist(): 
            return None
            # raise ValueError("Mapping of type to AnnData is broken, please check!")
        else:
            typ = self.adata.obs[self.adata_mapping["Type"]]
            typ = np.array(typ) 
            return typ
        
    @Type.setter
    def Type(self,Type):
        """list : list (or 1D np array) of integer values that reference a Taxonomy object types""" 
        self.adata.obs[self.adata_mapping["Type"]] = Type
        
    @property
    def N(self):
        """int : Size of the tissue graph
            internally stored as igraph size
        """
        if not self.is_empty():
            return(self.adata.shape[0])
        else: 
            raise ValueError('TissueGraph does not contain an AnnData object, please verify!')
    
    @property
    def node_size(self):
        if self.is_empty():
            return None
        elif self.adata_mapping["node_size"] not in self.adata.obs.columns.values.tolist(): 
            raise ValueError("Mapping of type to AnnData is broken, please check!")
        else: 
            return self.adata.obs[self.adata_mapping["node_size"]]
    
    @node_size.setter
    def node_size(self,Nsz):
        self.adata.obs[self.adata_mapping["node_size"]] = list(Nsz)
    
    @property
    def Slice(self):
        """
            Slice : dependent property - will query info from anndata and return
        """
        if self.adata is None:
            return None
        elif self.adata_mapping["Slice"] not in self.adata.obs.columns.values.tolist(): 
            raise ValueError("Mapping of type to AnnData is broken, please check!")
        else: 
            return self.adata.obs[self.adata_mapping["Slice"]]

    @Slice.setter
    def Slice(self,Slice):
        self.adata.obs[self.adata_mapping["Slice"]]=Slice

    @property
    def XY(self):
        """
            XY : dependent property - will query info from anndata and return
        """
        if self.adata is None:
            return None
        elif self.adata_mapping["XY"] not in self.adata.obsm.keys(): 
            raise ValueError("Mapping of XY to AnnData is broken, please check!")
        else: 
            return self.adata.obsm[self.adata_mapping["XY"]]

    @XY.setter
    def XY(self,XY): 
        self.adata.obsm[self.adata_mapping["XY"]]=XY
        
    @property    
    def X(self):
        """
            X : dependent property - will query info from Graph and return
        """
        return(self.XY[:,0])
        
    @property
    def Y(self):
        """Y : dependent property - will query info from Graph and return
        """
        return(self.XY[:,1])
    
    @property    
    def Ntypes(self): 
        """ 
            Ntypes: returns number of unique types in the graph
        """ 
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count how many")
        return(len(np.unique(self.Type)))
    
    @property
    def Nslices(self):
        """
            Nslices : returns number of unique slices in TG
        """
        unqS = np.unique(self.Slice)
        return(len(unqS))
    
    def build_feature_graph(self,X,n_neighbors = 15,metric = None,accuracy = {'prob' : 1, 'extras' : 1.5},metric_kwds = {},return_graph = False):
        """construct k-graph based on feature similarity

        Create a kNN graph (an igraph object) based on feature similarity. The core of this method is the calculation on how to find neighbors. 
        If metric is "precomputed" the distances are assumed to be known and we're almost done. 
        For all other metric values, we use pynndescent 

        Parameters
        ----------
        X : numpy array
            Either a distance matrix, i.e. squareform(pdist(feature_mat)) if metric = 'precomputed'.  ) or just a feature_mat
        n_neighbors : int
            How many neighbors (k) should we use in the knn graph
        metric : str
            either "precomputed", "random", or one of the MANY metrics supported by pynndescent. Random is for debugging only. 
        accuracy : dict with fields: 'prob' and 'extras'
            a dictionary with accuracy options for pynndescent. 'prob' should be in [0,1] and 'extras' is typically >1. 
            accuracy['prob'] conrols the 'diversify_prob' and accuracy['extra'] the 'pruning_degree_multplier' 
        metric_kwds : dict
            passthrough kwds that will be sent to pynndescent. 
        return_graph : bool
            will return the graph instead of updating self.FG

        Note
        ----
        There are LOTS of metric implemnted in pynndescent. 
        Many are not updated in the readthedocs so check the sources code! 
        """
    
        logging.info(f"building feature graph using {metric}")
        if metric is None:
            raise ValueError('metric was not specified')

        # checks if we have enough rows 
        n_neighbors = min(X.shape[0]-1,n_neighbors)

        if metric == 'precomputed':
            indices = np.argsort(X,axis=1)
            distances = np.sort(X,axis=1)
        elif metric == 'random': 
            indices = np.random.randint(X.shape[0],size=(X.shape[0],n_neighbors+1))
            distances = np.ones((X.shape[0],n_neighbors+1))
        else:
            # perform nn search (using accuracy x number of neighbors to improve accuracy)
            knn = pynndescent.NNDescent(X,n_neighbors = n_neighbors,
                                          metric = metric,
                                          diversify_prob = accuracy['prob'],
                                          pruning_degree_multiplier = accuracy['extras'],
                                          metric_kwds = metric_kwds)

            # get indices and remove self. 
            (indices,distances) = knn.neighbor_graph

        # take the first K values remove first self similarities    
        indices = indices[:,1:n_neighbors+1]
        distances = distances[:,1:n_neighbors+1]

        id_from = np.tile(np.arange(indices.shape[0]),indices.shape[1])
        id_to = indices.flatten(order='F')

        # build graph
        edgeList = np.vstack((id_from,id_to)).T
        G = igraph.Graph(n=X.shape[0], edges=edgeList)
        G.simplify()
        if return_graph:
            return G
        else:
            self.adata.obsp["FG"]=  G.get_adjacency_sparse()
            self.FG = G
            return self
    
    def build_spatial_graph(self,XYS,names = None):
        """construct graph based on Delauny neighbors
        
        build_spatial_graph will create an igrah using Delaunay triangulation

        Parameters
        ----------
            XYS : numpy Nx2 or Nx3 array 
            centroid regions to build a graph around, data is assumed to be Nx3 with X,Y and S (slice) data
                  if it's only Nx2 assuming a single S=0

        """
        # validate input
        if not isinstance(XYS, np.ndarray):
            raise ValueError('XY must be a numpy array')
        
        if XYS.shape[1]==2:
            XYS = np.hstack((XYS,np.zeros((XYS.shape[0],1))))
        
        if not XYS.shape[1]==3:
            raise ValueError('XYS must have either two (XY) or three (XYS) columns')
        
        unqS = np.unique(XYS[:,2])
        logging.info(f"Building spatial graphs for {len(unqS)} sections")
        el = list()
        cnt=0
        for s in range(len(unqS)): 
            # get XY for a given slice
            XY = XYS[XYS[:,2]==unqS[s],0:2]
            # start with triangulation
            dd=Delaunay(XY)

            # create Graph from edge list
            EL = np.zeros((dd.simplices.shape[0]*3,2),dtype=np.int64)
            for i in range(dd.simplices.shape[0]): 
                EL[i*3,:]=[dd.simplices[i,0],dd.simplices[i,1]]
                EL[i*3+1,:]=[dd.simplices[i,0],dd.simplices[i,2]]
                EL[i*3+2,:]=[dd.simplices[i,1],dd.simplices[i,2]]

            # update vertices numbers to account for previously added nodes (cnt)
            EL = EL + cnt
            # update cnt for next round
            cnt = cnt + XY.shape[0]

            # convert to list of tuples to make igraph happy and add them. 
            el = el + list(zip(EL[:,0], EL[:,1]))
        
        self.SG  = igraph.Graph(n=XYS.shape[0],edges=el,directed=False).simplify()
        logging.info("updating anndata")
        self.adata.obsp["SG"] = self.SG.get_adjacency_sparse()
        self.adata.obsm[self.adata_mapping["XY"]]=XYS[:,0:2]
        self.adata.obs[self.adata_mapping["node_size"]] = np.ones(XYS.shape[0])
        if names is None: 
            self.adata.obs[self.adata_mapping["name"]] = list(range(self.N))
        else: 
            self.adata.obs[self.adata_mapping["name"]] = names
        logging.info("done building spatial graph")
    
    def contract_graph(self, TypeVec=None):
        """find zones/region, i.e. spatially continous areas in the graph the same (cell/microenvironment) type
        
        reduce graph size by merging spatial neighbors of same type. 
        Given a vector of types, will contract the graph to merge vertices that are both next to each other and of the same type. 
        
        Parameters
        ----------
        TypeVec : 1D numpy array with dtype int (default value is self.Type)
            a vector of Types for each node. If None, will use self.Type

        Note
        ----
        Default behavior is to assign the contracted TG the same taxonomy as the original graph. 
        
        Returns
        -------
        TissueGraph 
            A TG object after vertices merging. 
        """

        # Figure out which type to use
        if TypeVec is None: 
            TypeVec = self.Type
        
        # get edge list - work with names and not indexes in case things shift around (they shouldn't),     
        EL = np.asarray(self.SG.get_edgelist()).astype("int")
        nm = self.adata.obs["name"]
        EL[:,0] = np.take(nm,EL[:,0])
        EL[:,1] = np.take(nm,EL[:,1])
        
        # only keep edges where neighbors are of same types
        EL = EL[np.take(TypeVec,EL[:,0]) == np.take(TypeVec,EL[:,1]),:]
        
        # remake a graph with potentially many components
        IsoZonesGraph = igraph.Graph(n=self.N, edges=EL, directed = False)
        IsoZonesGraph = IsoZonesGraph.as_undirected().simplify()

        # because we used both type and proximity, the original graph (based only on proximity)
        # that was a single component graph will be broken down to multiple components 
        # finding clusters for each component. 
        cmp = IsoZonesGraph.components()
        
        IxMapping = np.asarray(cmp.membership)
        
        ZoneName, ZoneSingleIx = np.unique(IxMapping, return_index=True)
        
        # zone size sums the current graph zone size per each aggregate (i.e. zone or microenv)
        df = pd.DataFrame(data = self.node_size)
        df['type'] = IxMapping
        ZoneSize = df.groupby(['type']).sum()
        ZoneSize = np.array(ZoneSize).flatten()
        
        # calculate zones feature_mat
        # if all values are Nan, just replace with tuple of the required size
        if np.all(np.isnan(self.feature_mat)): 
            zone_feat_mat = (len(ZoneSize),self.feature_mat.shape[1])
        else: 
            df = pd.DataFrame(data = self.feature_mat)
            df['type']=IxMapping
            zone_feat_mat = np.array(df.groupby(['type']).mean())
            
        # create new SG for zones 
        ZSG = self.SG.copy()
        
        comb = {"X" : "mean",
               "Y" : "mean",
               "Type" : "ignore",
               "name" : "ignore"}
        
        ZSG.contract_vertices(IxMapping,combine_attrs=comb)
        ZSG.simplify()

        # create a new Tissue graph by copying existing one, contracting, and updating XY
        ZoneGraph = TissueGraph(feature_mat=zone_feat_mat, 
                                basepath=self.basepath,
                                layer_type="isozone",
                                redo=True,
                                )

        ZoneGraph.SG = ZSG
        ZoneGraph.names = ZoneName
        ZoneGraph.node_size = ZoneSize
        ZoneGraph.Type = TypeVec[ZoneSingleIx]
        ZoneGraph.Upstream = IxMapping
        ZoneGraph.tax = self.tax
        
        return(ZoneGraph)
                             
    def type_freq(self): 
        """return the catogorical probability for each type in TG
        
        Probabilities are weighted by the node_size
        """
        if self.Type is None: 
            raise ValueError("Type not yet assigned, can't count frequencies")
        unqTypes = np.unique(self.Type)
        Ptypes = tmgu.count_values(self.Type,unqTypes,self.node_size)
        
        return Ptypes,unqTypes
    
    def cond_entropy(self):
        """calculate conditional entropy of the tissue graph
           
           cond entropy is the difference between graph entropy based on zones and type entropy
        """
        Pzones = self.node_size
        Pzones = Pzones/np.sum(Pzones)
        Entropy_Zone = -np.sum(Pzones*np.log2(Pzones))
        
        # validate that type exists
        if self.Type is None: 
            raise ValueError("Can't calculate cond-entropy without Types, please check")
            
        Ptypes = self.type_freq()[0] 
        Entropy_Types=-np.sum(Ptypes*np.log2(Ptypes))
        
        cond_entropy = Entropy_Zone-Entropy_Types
        return(cond_entropy)
    
    def extract_environments(self,ordr = None,typevec = None):
        """returns the categorical distribution of neighbors. 
        
        Depending on input there could be two uses, 
            usage 1: if ordr is not None returns local neighberhood defined as nodes up to distance ordr on the graph for all vertices. 
            usage 2: if typevec is not None returns local env based on typevec, will return one env for each unique type in typevec
            
        Return
        ------
        numpy array
            Array with Type frequency for all local environments for all types in TG. 
        """
        unqlbl = np.unique(self.Type)
        
        # arrange the indexes for the environments. 
        # if we use ordr this is neighborhood defined by iGraph
        # if we provide types, than indexes of each type. 
        if ordr is not None and typevec is None:
            ind = self.SG.neighborhood(order = ordr)
        elif typevec is not None and ordr is None:
            ind = list()
            for i in range(len(np.unique(typevec))):
                ind.append(np.flatnonzero(typevec==i))
        else: 
            raise ValueError('either order or typevec must be provided, not both (or none)')
        
        unqlbl = np.unique(self.Type)
        Env = np.zeros((len(ind),len(unqlbl)),dtype=np.int64)
        ndsz = self.node_size.copy().astype(np.int64)
        int_types = self.Type.astype(np.int64)
        AllTypes = [int_types[ix] for ix in ind]
        AllSizes = [ndsz[ix] for ix in ind]
        for i in range(Env.shape[0]):
            Env[i,:]=np.bincount(AllTypes[i],weights = AllSizes[i],minlength=len(unqlbl))
        
        # should be the same as above, but much slower... keeping it here for now till more testing is done. 
        # for i in range(len(ind)):
        #     Env[i,:]=count_values(self.Type[ind[i]],unqlbl,ndsz[ind[i]],norm_to_one = False)
            
        return(Env)
    
    def graph_local_avg(self,VecToSmooth,ordr = 3,kernel = np.array([0.4,0.3,0.2,0.1])):
        """Simple local average of a Vec based on local neighborgood
        
        Parameters
        ----------
            VecToSmooth : numpy array
                The values we want to smooth. len(VecToSmooth) must be self.N
        """
        
        if len(VecToSmooth) is not self.N: 
            raise ValueError(f"Length of input vector {len(VecToSmooth)} doesn't match TG.N which is {self.N}")
        
        Smoothed = np.zeros((len(VecToSmooth),ordr+1))
        Smoothed[:,0] = VecToSmooth
        for i in range(ordr):
            ind = self.SG.neighborhood(order = i+1,mindist=i)
            for j in range(len(ind)):
                ix = np.array(ind[j],dtype=np.int64)
                Smoothed[j,i+1] = np.nanmean(VecToSmooth[ix])

        kernel = kernel[None,:]
        kernel = np.repeat(kernel,len(VecToSmooth),axis=0)
        ix_nan = np.isnan(Smoothed)
        kernel[ix_nan]=0
        sum_of_rows = kernel.sum(axis=1)
        kernel = kernel / sum_of_rows[:, None]
        Smoothed[ix_nan] = 0
        Smoothed = np.multiply(Smoothed,kernel)
        Smoothed = Smoothed.sum(axis=1)
        return(Smoothed)
    
    def watershed(self,InputVec):
        """Watershed segmentation based on InputVec values
        
        Watershed on the TG spatial graph. 
        First finds local peaks and then assigns all nodes to their closest local peak using dijkstra
        
        Parameters
        ----------
        
        Return
        ------
        (id,dist) 
            tuple with id and distance to cloest zone. 
        """
        is_peak = np.zeros(InputVec.shape).astype('bool')
        ind = self.SG.neighborhood(order = 1,mindist=1)
        for i in range(len(ind)):
            is_peak[i] = np.all(InputVec[i]>InputVec[ind[i]])
        peaks = np.flatnonzero(is_peak)  

        Adj = self.SG.get_adjacency_sparse()
        Dij_min, predecessors,ClosestPeak = dijkstra(Adj, directed=False, 
                                                          indices=peaks, 
                                                          return_predecessors=True, 
                                                          unweighted=False, 
                                                          limit=np.inf, 
                                                          min_only=True)
        
        # renumber all closest peak continously
        u,ix_rev = np.unique(ClosestPeak, return_inverse=True)
        ClosestPeak=u[ix_rev]
        
        # relabel HoodId in case we have a heterozone that was split along the way
        # by contracting and expanding where each contracted zone gets a unique ID we
        # basically garantuee that Ntypes = N (for now...)
        CG = self.contract_graph(TypeVec = ClosestPeak)
        Id = np.arange(CG.N)
        ClosestPeak = Id[CG.Upstream]
        
        return (ClosestPeak,Dij_min)
        

    def calc_entropy_at_different_Leiden_resolutions(self,Rvec = np.logspace(-1,2.5,100)): 
        Ent = np.zeros(Rvec.shape)
        Ntypes = np.zeros(Rvec.shape)
        for i in range(len(Rvec)):
            TypeVec = self.FG.community_leiden(resolution_parameter=Rvec[i],objective_function='modularity').membership
            TypeVec = np.array(TypeVec).astype(np.int64)
            Ent[i] = self.contract_graph(TypeVec).cond_entropy()
            Ntypes[i] = len(np.unique(TypeVec))
            
        self.cond_entropy_df = pd.DataFrame(data = {'Entropy' : Ent, 'Ntypes' : Ntypes, 'Resolution' : Rvec})     
    
    # def multi_optim_Leiden_from_existing_types(self,base_types,types_to_expand, 
    #                                            FeatureMat, max_subtypes = 10000, 
    #                                            opt_params = {'iters' : 10, 'n_consensus' : 50}):
        
    #     def ObjFunLeidenRes_FG(res,FGtosplit,ix,TypeVec,return_types = False):
    #         """
    #         Basic optimization routine for Leiden resolution parameter. 
    #         Implemented using igraph leiden community detection
    #         """
            
    #         mask = np.zeros(TypeVec.shape,dtype='bool')
    #         mask[ix] = True
    #         mx_id = TypeVec[~mask].max()+1
            
    #         # if asked to return types, run this only once, no need for averaging. 
    #         iter_to_avg = opt_params['iters']
    #         if return_types:
    #             iter_to_avg=1
            
    #         EntropyVec = np.zeros(opt_params['iters'])
    #         for i in range(iter_to_avg):
    #             # split cells in the provided graph
    #             SplitTypes = FGtosplit.community_leiden(resolution_parameter=res,objective_function='modularity').membership
    #             # adjust ids - make into numpy array and shift to account for existing types
    #             SplitTypes = np.array(SplitTypes).astype(np.int64) + mx_id
                
    #             # recreate type vector for the whole tissue
    #             TypeVec2 = TypeVec.copy()
    #             TypeVec2[ix] = TypeVec2[ix] + SplitTypes
    #             EntropyVec[i] = self.contract_graph(TypeVec2).cond_entropy()
                
    #         Entropy = EntropyVec.mean()
    #         if return_types: 
    #             return -Entropy,TypeVec2
    #         else:
    #             return(-Entropy)
            
    #     # to ease bookkeeping, multiply cell type integers with a large const so we could add subtypes later
    #     # ix_expand = np.isin(base_types,types_to_expand)
    #     # base_types[ix_expand] = base_types[ix_expand] * max_subtypes
    #     # types_to_expand = [x * max_subtypes for x in types_to_expand]
        
    #     # Build subgraphs    
    #     start = time.time()

    #     print(f"Optimize each type to see if it can be split further")
    #     for i in range(len(types_to_expand)):
            
    #         # get indexes of cells with this type
    #         ix = np.flatnonzero(base_types == types_to_expand[i])
            
    #         # create a subgraph for these cells 
    #         sub_FG = self.build_feature_graph(FeatureMat[ix,:],metric = 'cosine',accuracy=3,return_graph = True)
            
    #         # Cond entropy optimization only for these cells
    #         type_copy = base_types.copy()
    #         n_before = len(np.unique(base_types))
    #         sol = minimize_scalar(ObjFunLeidenRes_FG, bounds = (0.1,30), 
    #                                                   method='bounded',
    #                                                   args = (sub_FG,ix,type_copy),
    #                                                   options={'xatol': 1e-1, 'disp': 3})
    #         # get types
    #         opt_res = sol['x']
    #         ent,base_types = ObjFunLeidenRes_FG(opt_res,sub_FG,ix,base_types,return_types=True)
    #         n_after = len(np.unique(base_types))
    #         print(f'i: {i} time: {time.time()-start:.2f} type before: {n_before} added: {n_after-n_before}')
            
    #     return base_types
                
    
    # def multilayer_Leiden_with_cond_entropy(self,base_types = None, 
    #                                         FeatureMat = None, 
    #                                         return_res = False,
    #                                         opt_params = {'iters' : 10, 'n_consensus' : 50}): 
    #     """
    #         Find optimial clusters by peforming clustering on two-layer graph. 
            
    #         Input
    #         -----
    #         TG : A TissueGraph that has matching SpatialGraph (SG) and FeatureGraph (FG)
    #         optimization is done on resolution parameter  
            
    #     """
    #     start = time.time()
    #     if base_types is not None: 
    #         unq_types = np.unique(base_types)
    #         if FeatureMat is None: 
    #             raise ValueError('if types are supplied then a features matrix must be included')
    #         sub_FGs = list()
    #         all_ix = list()
    #         print(f"Building feature subgraphs for each type")
    #         for i in range(len(unq_types)):
    #             ix = np.flatnonzero(base_types == unq_types[i])
    #             all_ix.append(ix)
    #             # create a subgraph 
    #             sub_FGs.append(self.build_feature_graph(FeatureMat[ix,:],metric = 'cosine',accuracy=1,return_graph=True))
    #         print(f'done, time: {time.time()-start:.2f}')
            

    #     def ObjFunLeidenRes(res,return_types = False):
    #         """
    #         Basic optimization routine for Leiden resolution parameter. 
    #         Implemented using igraph leiden community detection
    #         """
    #         EntropyVec = np.zeros(opt_params['iters'])
    #         if base_types is not None:
    #             for i in range(opt_params['iters']):
    #                 all_sub_Types = list()
    #                 for j in range(len(unq_types)):
    #                     sub_TypeVec = sub_FGs[j].community_leiden(resolution_parameter=res,
    #                                                               objective_function='modularity').membership
    #                     sub_TypeVec = np.array(sub_TypeVec).astype(np.int64)
    #                     all_sub_Types.append(sub_TypeVec)
                        
    #                 # bookeeping: rename all clusters so that numbers are allways unique. 
    #                 # find the largest number of subtypes we need to add
    #                 # take log10 and ceil so that we find a place value that is larget that that
    #                 # multiply base_types with that value. 
    #                 mxdec = np.max(10**np.ceil(np.log10([len(np.unique(x)) for x in all_sub_Types])))
    #                 TypeVec = base_types * mxdec
    #                 for j in range(len(all_ix)): 
    #                     TypeVec[all_ix[j]] = TypeVec[all_ix[j]] + all_sub_Types[j]
    #                 EntropyVec[i] = self.contract_graph(TypeVec).cond_entropy()
    #             Entropy = EntropyVec.mean()
                    
    #         else:
    #             for i in range(opt_params['iters']):
    #                 TypeVec = self.FG.community_leiden(resolution_parameter=res,
    #                                                    objective_function='modularity').membership
    #                 TypeVec = np.array(TypeVec).astype(np.int64)
    #                 EntropyVec[i] = self.contract_graph(TypeVec).cond_entropy()
    #             Entropy = EntropyVec.mean()
    #         if return_types: 
    #             return -Entropy,TypeVec
    #         else:
    #             return(-Entropy)

    #     print(f"Calling initial optimization")
    #     sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
    #                                            method='bounded',
    #                                            options={'xatol': 1e-2, 'disp': 3})
    #     opt_res = sol['x']
        
    #     # consensus clustering
    #     TypeVec = np.zeros((self.N,opt_params['n_consensus']))
    #     for i in range(opt_params['n_consensus']):
    #         ent,TypeVec[:,i] = ObjFunLeidenRes(opt_res,return_types = True)
            
    #     if opt_params['n_consensus']>1:
    #         cmb = np.array(list(itertools.combinations(np.arange(opt_params['n_consensus']), r=2)))
    #         rand_scr = np.zeros(cmb.shape[0])
    #         for i in range(cmb.shape[0]):
    #             rand_scr[i] = adjusted_rand_score(TypeVec[:,cmb[i,0]],TypeVec[:,cmb[i,1]])
    #         rand_scr = squareform(rand_scr)
    #         total_rand_scr = rand_scr.sum(axis=0)
    #         TypeVec = TypeVec[:,np.argmax(total_rand_scr)]
                                                  

    #     print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {-sol['fun']} number of evals: {sol['nfev']}")
    #     if return_res: 
    #         return TypeVec,opt_res
    #     else: 
    #         return TypeVec
    
    def gradient_magnitude(self,V):
        """Spatial gradient based on spatial graph
        
        Calculate the gradient defined as sqrt(dV/dx^2+dV/dy^2) where dV/dx(y) is calcualted using simple trigo
        
        """
        EL = self.SG.get_edgelist()
        EL = np.array(EL,dtype='int')
        XY = self.XY
        dV = V[EL[:,1]]-V[EL[:,0]]
        dX = XY[EL[:,1],0]-XY[EL[:,0],0]
        dY = XY[EL[:,1],1]-XY[EL[:,0],1]                      
        alpha = np.arctan(dX/dY)
        dVdX = dV*np.cos(alpha)/dX
        dVdY = dV*np.sin(alpha)/dY 
        dVdXY = np.sqrt(dVdX**2 + dVdY**2)
        df = pd.DataFrame({'dVdXY' : np.hstack((dVdXY,-dVdXY))})
        df['type']=np.hstack((EL[:,0],EL[:,1]))
        gradmag = np.array(df.groupby(['type']).mean())
        gradmag = np.array(gradmag)
        return(gradmag)
    
    def graph_local_median(self,VecToSmooth,ordr = 3):
        """local median of VecToSmooth based in spatial graph
        """
        ind = self.SG.neighborhood(order = ordr)
        Smoothed = np.zeros(VecToSmooth.shape)
        for j in range(len(ind)):
            ix = np.array(ind[j],dtype=np.int64)
            Smoothed[j] = np.nanmedian(VecToSmooth[ix])
        return(Smoothed)
class Taxonomy:
    """Taxonomical system for different biospatial units (cells, isozones, regions)
    
    Attributes
    ----------
    name : str
        a name of the taxonomy (will be used to file io)
    
    """
    
    def __init__(self, name=None, 
        # Types=None, feature_mat=None
        ): 
        """Create a Taxonomy object

        Parameters
        ----------
        name : str
            the name of the taxonomy object
        Types : list 
            list of types names that will be used in this Taxonomy
        """
        if name is None: 
            raise ValueError("Taxonomy must get a name")
        self.name = name
            
        self._df = pd.DataFrame()
        self._feature_cols = list()
        return None
    
    @property
    def Type(self): 
        """list: Building blocks of this Taxonomy
        
        Setter verify that there are no duplications. 
        """
        return(list(self._df.index))
    
    @Type.setter
    def Type(self,Types):
        if len(Types) is not len(set(Types)):
            dups = [Types.count(t) for t in set(Types) if Types.count(t)>1]
            raise ValueError(f"Types must be unique. Found duplicates: {dups}")
        if self.is_empty():
            self._df.index = Types
        else: 
            if len(Types) is not self._df.shape[0]: 
                raise ValueError('Changing Types of Taxonomy with defined values is not allowed')
            self._df.index = Types

    @property
    def feature_mat(self): 
        """ndarray: feature values for all types in the taxonomy
        
        Setter can get as input either numpy array (ordered same as self.Type) or pandas dataframe
        """
        if self.is_empty(): 
            return None
        else: 
            return(self._df.loc[:,self._feature_cols].to_numpy())
        
    @feature_mat.setter
    def feature_mat(self,F):
        # first, make sure F is a DataFrame
        if isinstance(F, pd.DataFrame):
            df_features = F
        else: # assumes F is a matrix, make into a df and give col names
            df_features = pd.DataFrame(F)
            df_features.columns = [f"f_{i:03d}" for i in range(F.shape[1])]

        # either replace or concat columns based on their name
        for c in df_features.columns: 
            if c in self._feature_cols: 
                self._df.loc[:,c]=F.loc[:,c]
            else: 
                self._df = pd.concat((self._df,df_features.loc[:,c]),axis=1)
                self._feature_cols.append(c)
        
        self._feature_cols = df_features.columns
        
    def is_empty(self):
        """ determie if taxonomy is empty
        """
        return (self._df.empty)
        
    def save(self, basepath):
        """save to basepath using Taxonomy name
        """
        self._df.to_csv(os.path.join(basepath, f"Taxonomy_{self.name}.csv"))
        
    def load(self, basepath):
        """save from basepath using Taxonomy name
        """
        self._df = pd.read_csv(os.path.join(basepath, f"Taxonomy_{self.name}.csv"))
        
    def add_types(self,new_types,feature_mat):
        """add new types and their feature average to the Taxonomy
        """
        df_new = pd.DataFrame(feature_mat)
        df_new['type']=new_types
        type_feat_df = df_new.groupby(['type']).mean()
        missing_index = type_feat_df.index.difference(self._df.index)
        self._df = pd.concat((self._df,type_feat_df.iloc[missing_index,:]),axis=0)
        
        return None
