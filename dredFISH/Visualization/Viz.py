"""Viz module manages all TissueMultiGraph vizualization needs

The module contains two elements: View and the Panel class hierarchy. 
View acts as:
#. connection to specific TMG container,
#. manage figure related stuff (size, save/load, etc)  
#. container of different Panels. 

The Panel hierarchy is where the specific graphics are created. 
The base class (Panel) is an abstract class with an abstract plot method users have to overload: 

All Panels include a few important properties: 
#. V : a pointer to the View that this Panel belogs to. This also acts as the link to TMG (self.V.TMG)
#. Data : a dict container for any data required for Viz, 
#. ax
# either given as input during __init__ 

A specific type of Panel that is important in the hierarchy is Map which is dedicated to spatial plotting of a TMG section. 

Other panels (Scatter, Historgram, LogLogPlot) etc are there for convinent so that View maintains 
"""

"""
TODO: 
1. Finish with legends and pie/wedge legens. 
2. clean dependencies
3. move UMAP color to coloru
4. move ploting from geomu to powerplots (?)
"""

from matplotlib.gridspec import GridSpec, SubplotSpec
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import Rectangle as pRectangle
import colorcet as cc

import umap
import xycmap

import abc

import seaborn as sns

from scipy import optimize
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import jensenshannon, pdist, squareform

from dredFISH.Utils import geomu
from dredFISH.Utils import coloru 
from dredFISH.Analysis.TissueGraph import *

"""
TODO: 
* Add scale-bar to Map and all subclasses (pixel size)
* Add colorbar to Colorpleth (but not to RandomColorpleth...)
* Fix Circle-legend w/o wedges 
* Fix TypeMap to the new non-stype format
* Add RegionMaps that shows regions as boundaries and cells as types. 
* In TMG - add isozones / region geometry by merging polygons using type vector (use accessory function in geomu)
"""

class View:
    def __init__(self, TMG, name=None, figsize=(11,11)):
        
        # each view needs a unique name
        self.name = name
        
        # link to the TMG used to create the view
        self.TMG = TMG
        
        # list of all panels
        self.Panels = list()
        
        self.figsize = figsize
        self.fig = plt.figure(figsize = self.figsize)
        
    def show(self):
        """
        plot all panels. 
        """
        # Add all panels to figure
        for i in range(len(self.Panels)): 
            # add an axes
            if isinstance(self.Panels[i].pos,SubplotSpec):
                ax = self.fig.add_subplot(self.Panels[i].pos)
            else: 
                ax = self.fig.add_axes(self.Panels[i].pos)
            self.Panels[i].ax = ax
            plt.sca(ax)
            self.Panels[i].plot()

class BasisView(View):
    def __init__(self,TMG,section = None, basis = np.arange(24), qntl = (0.025,0.975),colormaps="jet",subplot_layout = [1,1],**kwargs):
        figsize = kwargs.get('figsize',(15,10))
        super().__init__(TMG,name = "View basis",figsize=figsize)

        # decide the subplot layout, that keep a 2x3 design
        # this will only run if the subplot_layout is smaller then needed 
        while np.prod(subplot_layout)<len(basis):
            subplot_layout[1] += 1
            if np.prod(subplot_layout)>=len(basis): 
                break
            if subplot_layout[1]/subplot_layout[0] > 1.5:
                subplot_layout[0] += 1

        # set up subpanels
        gs = self.fig.add_gridspec(subplot_layout[0],subplot_layout[1],wspace = 0.01,hspace = 0.01) 
        # add 24 Colorpleths
        feature_mat = self.TMG.Layers[0].get_feature_mat(section = section)
        feature_mat = feature_mat[:,basis]
        for i in range(len(basis)):
            P = Colorpleth(feature_mat[:,i],V = self,section = section,colormaps = colormaps,pos = gs[i],**kwargs,qntl = qntl)

class SingleMapView(View):
    """
    A view that has only a single map in it. 
    map_type is one of 'type', 'colorpleth', 'random'
    """
    def __init__(self, TMG, section = None,level_type = "cell",map_type = "type", val_to_map = None, figsize=(11, 11),**kwargs):
        super().__init__(TMG, "Map", figsize)

        if len(TMG.unqS)==1 and section is None: 
            section = TMG.unqS[0]
        
        geom_type = TMG.layer_to_geom_type_mapping[level_type]
        if map_type == 'type':
            Pmap = TypeMap(geom_type,V=self,section = section,**kwargs)
        elif map_type == 'colorpleth': 
            if val_to_map is None: 
                raise ValueError("Must provide val_to_map if map_type is colorpleth")
            Pmap = Colorpleth(val_to_map,V=self,section = section,**kwargs)
        elif map_type == 'random':
            Pmap = RandomColorMap(geom_type,V=self,section=section,**kwargs)
        else: 
            raise ValueError(f"value {map_type} is not a recognized map_type")
        

class UMAPwithSpatialMap(View):
    """
    TODO: 
    1. add support multi-section
    2. allow coloring by cell types taken from TMG. 
    """
    def __init__(self,TMG,section = None, qntl = (0,1),clp_embed = (0,1),**kwargs):
        super().__init__(TMG,name = "UMAP with spatial map",figsize=(16,8))

        # add two subplots
        gs = self.fig.add_gridspec(1,2,wspace = 0.01)
        basis = self.TMG.Layers[0].get_feature_mat(section = section)

        for i in range(basis.shape[1]):
            lmts = np.percentile(basis[:,i],np.array(qntl)*100)
            basis[:,i] = np.clip(basis[:,i],lmts[0],lmts[1])
        
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(basis)
        self.ump1 = np.clip(embedding[:,0],np.quantile(embedding[:,0],clp_embed[0]),np.quantile(embedding[:,0],clp_embed[1]))
        self.ump2 = np.clip(embedding[:,1],np.quantile(embedding[:,1],clp_embed[0]),np.quantile(embedding[:,1],clp_embed[1]))

        corner_colors = kwargs.get('corner_colors',["#d10f7c", "#53e6de", "#f4fc4e", "#4e99fc"]) #magenta, cyan, yellow, blue
        self.cmap = xycmap.custom_xycmap(corner_colors = corner_colors, n = (100, 100))



        colors = xycmap.bivariate_color(sx=self.ump1, sy=self.ump2, cmap=self.cmap)
        clr = np.vstack(colors)
        clr[clr>1]=1

        # add the scatter plot
        Pumap = Scatter(self,self.ump1,self.ump2,c = clr,s=0.1,name = 'umaps',pos = gs[0],
                        xlabel = 'UMAP-1',ylabel = 'UMAP-2',xtick = [],ytick=[],**kwargs)
        Pmap = Map(V = self,section = section,rgb_faces = clr,pos = gs[1],name = 'map with UMAP colors',**kwargs)

    def show(self):
        super().show()
         # add legend 
        self.legend_ax = self.fig.add_axes([0.15, 0.15, 0.05, 0.1])
        self.legend_ax = xycmap.bivariate_legend(ax=self.legend_ax , sx=self.ump1, sy=self.ump2, cmap=self.cmap)
        self.legend_ax.set_xticks([])
        self.legend_ax.set_yticks([])


class DapiValueDistributions(View):
    def __init__(self,TMG,figsize = (16,8),min_dapi_line = None,max_dapi_line = None):
        super().__init__(TMG,name = "Dapi per section violin",figsize = figsize)
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = TMG.Layers[0].adata.obs['dapi'],xlabel='Section',ylabel='Dapi')
            self.min_dapi_line=min_dapi_line
            self.max_dapi_line=max_dapi_line
        else: 
            P = Histogram(V = self,values_to_count = TMG.Layers[0].adata.obs['dapi'],xlabel='Dapi')
            self.min_dapi_line=min_dapi_line
            self.max_dapi_line=max_dapi_line

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_dapi_line is not None: 
                self.Panels[0].ax.axhline(self.min_dapi_line)
            if self.max_dapi_line is not None: 
                self.Panels[0].ax.axhline(self.max_dapi_line)
        else: 
            if self.min_dapi_line is not None: 
                self.Panels[0].ax.axvline(self.min_dapi_line)
            if self.max_dapi_line is not None: 
                self.Panels[0].ax.axvline(self.max_dapi_line)

class SumValueDistributions(View):
    def __init__(self,TMG,figsize = (16,8),min_sum_line = None,max_sum_line = None):
        super().__init__(TMG,name = "Log10 Sum per section violin",figsize = figsize)
        num_values = TMG.Layers[0].adata.X.sum(1)
        num_values[num_values<0] = 0
        num_values = np.log10(num_values+1)
        min_sum_line = np.log10(min_sum_line+1)
        max_sum_line = np.log10(max_sum_line+1)
        self.min_sum_line=min_sum_line
        self.max_sum_line=max_sum_line
        
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = num_values,xlabel='Section',ylabel='Sum')
        else: 
            P = Histogram(V = self,values_to_count = num_values,xlabel='Sum')

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_sum_line is not None: 
                self.Panels[0].ax.axhline(self.min_sum_line)
            if self.max_sum_line is not None: 
                self.Panels[0].ax.axhline(self.max_sum_line)
        else: 
            if self.min_sum_line is not None: 
                self.Panels[0].ax.axvline(self.min_sum_line)
            if self.max_sum_line is not None: 
                self.Panels[0].ax.axvline(self.max_sum_line)

class ValueDistributions(View):
    def __init__(self,TMG,num_values,log=False,title='',figsize = (16,8),min_line = None,max_line = None):
        super().__init__(TMG,name = title+" per section violin",figsize = figsize)
        if log:
            title = 'Log10 '+title
            num_values[num_values<0] = 0
            num_values = np.log10(num_values+1)
            if not isinstance(min_line,type(None)):
                min_line = np.log10(min_line+1)
            if not isinstance(max_line,type(None)):
                max_line = np.log10(max_line+1)
        self.min_line=min_line
        self.max_line=max_line
        
        if self.TMG.Nsections > 1:
            P = Violin(V = self, cat_values = TMG.Layers[0].Section,num_values = num_values,xlabel='Section',ylabel=title)
        else: 
            P = Histogram(V = self,values_to_count = num_values,xlabel=title)

    def show(self): 
        super().show()
        if self.TMG.Nsections > 1:
            if self.min_line is not None: 
                self.Panels[0].ax.axhline(self.min_line)
            if self.max_line is not None: 
                self.Panels[0].ax.axhline(self.max_line)
        else: 
            if self.min_line is not None: 
                self.Panels[0].ax.axvline(self.min_line)
            if self.max_line is not None: 
                self.Panels[0].ax.axvline(self.max_line)


""" class IsoZonesView(View):
    # A view with two panels: a isozone colorpleth with node_size and log-log hist
    # simple overloading of colorpleth, only different is that we're getting values from TMG 
    # since TMG is only defined after init we pass the Data during set_view instead. 

    def __init__(self, section=0, name="isozones", pos=(0,0,1,1)):
        values_to_map = None
        super().__init__(values_to_map, section=section, name=name, pos=pos)
    def set_view(self):
        self.Data['values_to_map'] = np.log10(self.V.TMG.Layers[1].node_size)
        super().set_view() """



class LocalCellDensity(View):
    """
    Creates a map of local cell density of a given section (or None for all)
    local density is estimated using k-nearest neighbors (default k=10)
    """
    def __init__(self,TMG,figsize = (16,8),section = None, qntl = (0,1),k = 10,**kwargs):
        super().__init__(TMG, "local cell density", figsize)

        # calculate cell densities
        ind = self.TMG.Layers[0].SG.neighborhood(order = k)

        # add two subplots
        gs = self.fig.add_gridspec(1,2,wspace = 0.01)

        XY = TMG.Layers[0].get_XY(section = section)

        local_dens = geomu.local_density(XY,k=k)
        
        Pmap = Colorpleth(local_dens,V=self,section = section,qntl = qntl,**kwargs)


class Panel(metaclass=abc.ABCMeta):
    def __init__(self, V = None, name = None, pos=(0,0,1,1)):
        if not isinstance(V,View):
            raise ValueError(f"The value of parameter V must be a View object, instead it was {type(V)}")
        self.V = V
        self.name = name
        self.pos = pos
        self.ax = None
        self.Data = {}
        self.clrmp = None

        # Add the Panel into the View it is linked to. 
        self.V.Panels.append(self)

    @abc.abstractmethod
    def plot(self):
        """
        actual plotting happens here, overload in subclasses! 
        """
        pass
        

class Map(Panel):
    """
    Basic map plotting functionality. 
    This panels can plots single type of geometry from TMG based on use supplied rgb (Nx3 array) for faces and/or edges
    It does not calcualte any of the rgb maps. 
    """
    def __init__(self, V = None, section = None, name = None, 
                       pos=(0,0,1,1), geom_type = "voronoi",rgb_faces = None,
                       rgb_edges = None, **kwargs):
        super().__init__(V,name, pos)
        
        # set default section to be the first in unqS, useful is there is only single section
        if section is None:
            section = self.V.TMG.unqS[0]

        # section information: which section in TMG are we plotting
        self.section = section
        
        # Make sure one color was provided
        assert rgb_edges is not None or rgb_faces is not None,"To plot either pleaes provide RGB array (nx3) for either edges or faces "
        self.rgb_edges = rgb_edges
        self.rgb_faces = rgb_faces
        self.geom_type = geom_type

        # get limits
        if self.V.TMG.unqS.count(self.section) == 0:
            raise ValueError(f"section {self.section} is not found in TMG.unqS {self.V.TMG.unqS}")
        else: 
            section_ix = self.V.TMG.unqS.index(self.section)

        self.xlim = kwargs.get('xlim',None)
        self.ylim = kwargs.get('ylim',None)
        self.rotation = kwargs.get('rotation','auto')

        # get the geom collection saved in appropriate TMG Geom
        self.geom_collection = self.V.TMG.Geoms[section_ix][geom_type].verts

        return
        
    def plot(self):
        """
        plot a map (list of shapely points in colors) 
        """
        if self.geom_type == "points":
            self.sizes = None # TODO - make sure to fix self.sizes to something real...
            geomu.plot_point_collection(self.geom_collection,
                                        self.sizes,
                                        rgb_faces = self.rgb_faces,
                                        rgb_edges = self.rgb_edges, 
                                        ax = self.ax,
                                        xlm = self.xlim,ylm = self.ylim,
                                        rotation = self.rotation)
        else: 
            geomu.plot_polygon_collection(self.geom_collection,
                                          rgb_faces = self.rgb_faces,
                                          rgb_edges = self.rgb_edges, 
                                          ax = self.ax,
                                          xlm = self.xlim,ylm = self.ylim,
                                          rotation = self.rotation)
            

class TypeMap(Map):
    """
    plots types of given geom using UMAP color projection of types
    """
    def __init__(self, geom_type, V = None,section = None, name = None, 
                                      pos = (0,0,1,1), color_assign_method = 'supervised_umap',**kwargs):
        super().__init__(V = V,section = section, geom_type = geom_type, name = name, pos = pos,rgb_faces=[1,1,1],**kwargs)
        
        # find layer and tax ids
        layer = self.V.TMG.geom_to_layer_type_mapping[geom_type]
        layer_ix = self.V.TMG.find_layer_by_name(layer)
        tax_ix = self.V.TMG.layer_taxonomy_mapping[layer_ix]

        # get data and rename for simplicity
        self.Data['data'] = self.V.TMG.Taxonomies[tax_ix].feature_mat
        self.Data['tax'] = self.V.TMG.Taxonomies[tax_ix].Type
        self.Data['type'] = self.V.TMG.Layers[layer_ix].Type[self.V.TMG.Layers[layer_ix].Section==section]
        types = self.Data['type']
        target = self.Data['tax']
        _, target_index = np.unique(target, return_index=True)
        data =  self.Data['data']

        # get colors and assign for each unit (cell, isozone, regions)
        if color_assign_method == 'taxonomy': 
            self.rgb_by_type = self.V.TMG.Taxonomies[tax_ix].RGB
        elif color_assign_method == 'supervised_umap': 
            self.rgb_by_type = coloru.type_color_using_supervized_umap(data,target_index)
        elif color_assign_method == 'linkage':
            metric = kwargs.get("metric","cosine")
            self.cmap_names = kwargs.get('colormaps', "hot")
            self.cmap = coloru.merge_colormaps(self.cmap_names,range=(0.05,1))
            self.rgb_by_type = coloru.type_color_using_linkage(data,self.cmap,metric = metric)
        self.rgb_faces = self.rgb_by_type[types.astype(int),:]


class RandomColorMap(Map):
    """
    plots types of given geom using UMAP color projection of types
    """
    def __init__(self, geom_type, V = None,cmap_list = ['jet'],section = None, name = None, 
                                      pos = (0,0,1,1), **kwargs):
        super().__init__(V = V,section = section, geom_type = geom_type, name = name, pos = pos,rgb_faces=[1,1,1],**kwargs)
        
        # set colormap
        self.cmap_names = kwargs.get('colormaps', "jet")
        self.cmap = coloru.merge_colormaps(self.cmap_names)
        # find layer and tax ids
        layer = self.V.TMG.geom_to_layer_type_mapping[geom_type]
        layer_ix = self.V.TMG.find_layer_by_name(layer)
        self.rgb_faces = self.cmap(np.linspace(0,1,self.V.TMG.N[layer_ix]))
        self.rgb_faces = self.rgb_faces[np.random.permutation(self.V.TMG.N[layer_ix]),:]
       
class Colorpleth(Map):
    """
    Show a user-provided vector color coded. 
    It will choose the geomtry (voronoi,isozones,regions) from the size of the use provided values_to_map vector. 
    """
    def __init__(self, values_to_map, V = None,section = None, name = None, 
                                      pos = (0,0,1,1), geom_type = None,qntl = (0,1),
                                      clp = (-np.inf,np.inf), **kwargs):
        # if geom_type is None chose it based on size of input vector
        if geom_type is None:         
            lvlarr = np.flatnonzero(np.equal(V.TMG.N,len(values_to_map)))
            if len(lvlarr)!=1:
                raise ValueError('number of items in values_to_map doesnt match any level size')
            if lvlarr[0] == 0: 
                geom_type = "voronoi"
            elif lvlarr[0] == 1: 
                geom_type = "isozones"
            elif lvlarr[0] == 2: 
                geom_type = "regions"
        super().__init__(V = V,section = section, name = name, pos = pos,rgb_faces=[1,1,1],geom_type=geom_type,**kwargs)
        self.Data['values_to_map'] = values_to_map
        
        self.cmap_names = kwargs.get('colormaps', "hot")
        self.colorbar = kwargs.get('colorbar', False)
        self.cmap = coloru.merge_colormaps(self.cmap_names)
        
        # Create the rgb_faces using self.clrmp and self.Data['values_to_map']
        scalar_mapping = self.Data['values_to_map']

        # clip extreme values by value
        scalar_mapping = np.clip(scalar_mapping,clp[0],clp[1])

        # rescale usign quantiles
        scalar_mapping[scalar_mapping < np.percentile(scalar_mapping,qntl[0]*100)]=np.percentile(scalar_mapping,qntl[0]*100)
        scalar_mapping[scalar_mapping > np.percentile(scalar_mapping,qntl[1]*100)]=np.percentile(scalar_mapping,qntl[1]*100)
        
        # convert to 0-1 range to maximize colormap dynamic range
        scalar_mapping = scalar_mapping-scalar_mapping.min()
        scalar_mapping = scalar_mapping/scalar_mapping.max()
        
        self.rgb_faces = self.cmap(scalar_mapping)
        
 
                 
    
""" class LegendWithCircles(Panel): 
    def __init__(self, map_panel, name=None, pos=(0,0,1,1), scale=300, **kwargs):
        super().__init__(name=name, pos=pos, **kwargs)
        self.map_panel = map_panel
        self.scale = scale
        
    def set_view(self, count_type='index'):
        lvl = self.map_panel.lvl
        if count_type == 'index':
            if lvl == 0:
                cell_mapped_index = np.arange(self.V.TMG.Layers[lvl].N)
            else:
                cell_mapped_index = self.V.TMG.Layers[lvl].Upstream
            unq, cnt = np.unique(cell_mapped_index, return_counts=True)
        elif count_type == 'type':
            # cell_mapped_types = self.map_panel.Data['values_to_map']
            cell_mapped_types = self.V.TMG.Layers[lvl].Type 
            unq, cnt = np.unique(cell_mapped_types, return_counts=True)
        self.sz = cnt.astype('float')
        self.sz = self.sz/self.sz.mean()*self.scale

        if count_type == 'index':
            layout = self.V.TMG.Layers[lvl].FG.layout_fruchterman_reingold()
        elif count_type == 'type':
            # groupby (not implemented yet)
            raise ValueError("not implemented")
            _TG = self.V.TMG.Layers[lvl]
            sumTG = _TG.contract_graph(_TG.Type) # summarize the current graph by type...
            layout = sumTG.FG.layout_fruchterman_reingold()

        xy = np.array(layout.coords)
        xy[:,0] = xy[:,0]-xy[:,0].min()
        xy[:,1] = xy[:,1]-xy[:,1].min()
        xy[:,0] = xy[:,0]/xy[:,0].max()
        xy[:,1] = xy[:,1]/xy[:,1].max()
        self.xy = xy

        self.clr = self.map_panel.basecolors[self.V.TMG.Layers[lvl].Type.astype(int)]
        self.clrmp = self.map_panel.clrmp # xxx
    
    def plot(self): 
        # plt.sca(self.ax)
        self.ax.scatter(
                    x=self.xy[:,0], 
                    y=self.xy[:,1],
                    c=self.clr,
                    s=self.sz, 
                    )
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])
        
class LegendWithCirclesAndWedges(LegendWithCircles):
    def __init__(self, map_panel, cell_map_panel, name=None, pos=(0,0,1,1), **kwargs):
        super().__init__(map_panel, name=name, pos=pos, **kwargs)
        self.cell_map_panel = cell_map_panel
                               
    def plot(self): 

        # get fractions and sort by type2
        feature_mat = self.V.TMG.Layers[self.map_panel.lvl].feature_mat
        # ordr = np.argsort(self.cell_map_panel.type2)
        # feature_mat = feature_mat[:,ordr]
        sum_of_rows = feature_mat.sum(axis=1)
        feature_mat = feature_mat / sum_of_rows[:, None]
        n_smpl, n_ftrs = feature_mat.shape 
            
        # scale between radi in points to xy that was normed to 0-1
        scale_factor = self.V.fig.dpi * self.V.figsize[0] 
            
        xy = self.xy*scale_factor
        radi = np.sqrt(self.sz/np.pi) * self.scale/100
        cdf_in_angles = np.cumsum(np.hstack((np.zeros((n_smpl,1)), feature_mat)), axis=1)*360

        wedges = list()
        wedge_width = 0.66
        region_lvl = self.map_panel.lvl
        region_types = self.V.TMG.Layers[region_lvl].Type.astype(int)
        cell_lvl = self.cell_map_panel.lvl
        cell_types = self.V.TMG.Layers[cell_lvl].Type.astype(int)

        for i in range(n_smpl):
            for j in range(n_ftrs):
                w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
                           cdf_in_angles[i,j+1],
                           width=wedge_width*radi[i], 
                           facecolor=self.cell_map_panel.basecolors[cell_types[j]]
                           )
                c = Circle((xy[i,0],xy[i,1]), wedge_width*radi[i], 
                            facecolor=self.map_panel.basecolors[region_types[i]],
                            # facecolor=self.map_panel.clr[i,:], 
                            fill=True) 
                wedges.append(w)
                wedges.append(c)

        p = PatchCollection(wedges, match_original=True)
        self.ax.add_collection(p)

        margins = 0.05
        self.ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_xticks([])
        self.ax.set_yticks([])




  commented section was used to plot Cond Entropy Vs resolution, not clear if that is a panel
    
        # add a panel with conditional entropy
        if hasattr(self.TMG.Layers[0], 'cond_entropy_df') and self.lvl==1:
            EntropyCalcsL1 = self.TMG.Layers[0].cond_entropy_df
            fig = plt.figure()
         
            ax1 = plt.gca()
            yopt = self.TMG.cond_entropy[1]
            xopt = self.TMG.Ntypes[1]
            ax1.plot(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
            ylm = ax1.get_ylim()
            ax1.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
            ax1.set_xlabel('# of types',fontsize=18)
            ax1.set_ylabel('H (Map | Type)',fontsize=18)
            fig = plt.gcf()
            left, bottom, width, height = [0.6, 0.55, 0.25, 0.25]
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.semilogx(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
            ylm = ax2.get_ylim()
            ax2.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
            
            fig = plt.figure()
            unq,cnt = np.unique(self.TMG.Layers[0].Type,return_counts=True)
            plt.hist(cnt,bins=15);
            plt.title("Cells per type")
            plt.xlabel("# Cells in a type")
            plt.ylabel("# of Types")


class RegionMap(CellMap): 
    def __init__(self,name = "region map"):
        super().__init__(TMG,name = name)
        self.lvl = 2
    
    def set_view(self):
        super().set_view()
        
    def plot(self,V1 = None,**kwargs):
        super().plot(**kwargs)
        if V1 is None:
            return
        # add another panel with piechart markers
        # start new figure (to calc size factor)
        fig = plt.figure(figsize = self.figsize)
        self.figs.append(fig)
        ax = plt.gca()
            
        # get fractions and sort by type2
        feature_type_mat = self.TMG.Layers[self.lvl].feature_type_mat
        ordr = np.argsort(V1.type2)
        feature_type_mat = feature_type_mat[:,ordr]
        sum_of_rows = feature_type_mat.sum(axis=1)
        feature_type_mat = feature_type_mat / sum_of_rows[:, None]
            
        # scale between radi in points to xy that was normed to 0-1
        scale_factor = fig.dpi * self.figsize[0]
            
        xy = self.xy*scale_factor
        radi = np.sqrt(self.sz/np.pi)
        cdf_in_angles = np.cumsum(np.hstack((np.zeros((feature_type_mat.shape[0],1)),feature_type_mat)),axis=1)*360

        wedges = list()
        wedge_width = 0.66
        for i in range(feature_type_mat.shape[0]):
            for j in range(feature_type_mat.shape[1]):
                w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
                           cdf_in_angles[i,j+1],width = wedge_width*radi[i], facecolor = V1.clr[ordr[j],:])
                c = Circle((xy[i,0],xy[i,1]),wedge_width*radi[i],facecolor = self.clr[i,:],fill = True) 
                wedges.append(w)
                wedges.append(c)


        p = PatchCollection(wedges,match_original=True)
        ax.add_collection(p)

        margins = 0.05
        ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
        ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
        ax.set_xticks([])
        ax.set_yticks([])

next section could be used to show mapping between regions and cell
        
        fig = plt.figure(figsize=(10,10))
        self.figs.append(fig)
        region_cell_types = self.TMG.Layers[2].feature_type_mat
        row_sums = region_cell_types.sum(axis=1)
        row_sums = row_sums[:,None]
        region_cell_types_nrm=region_cell_types/row_sums
        g = sns.clustermap(region_cell_types_nrm,method="ward", cmap="mako",col_colors=V1.clr,row_colors = self.clr)
        g.ax_heatmap.set_xticks(list())
        g.ax_heatmap.set_yticks(list())
        self.figs.append(fig)
"""



class Scatter(Panel):
    def __init__(self,V,x,y,c = [0.25, 0.3, 0.95] , s = 0.1,name = 'scatter',pos = (0,0,1,1), **kwargs):
        super().__init__(V,name=name, pos=pos)
        self.Data['x'] = x
        self.Data['y'] = y
        self.Data['c'] = c
        self.Data['s'] = s
        self.xtick = kwargs.get('xtick',None)
        self.ytick = kwargs.get('ytick',None)
        self.xlabel = kwargs.get('xlabel',None)
        self.ylabel = kwargs.get('ylabel',None)
        self.label_font_size = kwargs.get('label_font_size',20)
       
    def plot(self):
        self.ax.scatter(x=self.Data['x'],y=self.Data['y'],c=self.Data['c'],s = self.Data['s'])
        if self.xtick is not None: 
            self.ax.set_xticks(self.xtick)
        if self.ytick is not None: 
            self.ax.set_yticks(self.ytick)
        self.ax.set_xlabel(self.xlabel,fontsize = self.label_font_size)
        self.ax.set_ylabel(self.ylabel,fontsize = self.label_font_size)

class Violin(Panel):
    def __init__(self,V = None,cat_values = None, num_values = None,name = 'violin',pos=(0,0,1,1),**kwargs):
        super().__init__(V,name=name, pos=pos)
        if cat_values is None:
            raise ValueError("Must supply categorical variable")
        if num_values is None: 
            raise ValueError("Must supply numerical variable")

        self.Data['cat_values'] = cat_values
        self.Data['num_values'] = num_values
        self.xlabel = kwargs.get('xlabel',None)
        self.ylabel = kwargs.get('ylabel',None)

    def plot(self):
        df = pd.DataFrame({'cat' : self.Data['cat_values'],
                           'num' : self.Data['num_values']})
        sns.violinplot(ax = self.ax,data = df,x = "cat", y = "num")
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    
        

class Histogram(Panel):
    def __init__(self, V = None,values_to_count = None, name='hist', pos=(0,0,1,1), n_bins=50, **kwargs):
        if values_to_count is None: 
            raise ValueError("Must supply values to show their distribution")
        super().__init__(V = V,name=name, pos=pos)
        self.n_bins = n_bins 
        self.Data['values_to_count'] = values_to_count
        self.xlabel = kwargs.get('xlabel',None)

    def plot(self): 
        self.ax.hist(self.Data['values_to_count'], bins=self.n_bins)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_yticks([])

class LogLogPlot(Histogram):
    def __init__(self, V = None,values_to_count = None, name="loglog", pos=(0,0,1,1), n_bins=50, trend_line = None,**kwargs):
        super().__init__(values_to_count, name=name, pos=pos, n_bins=n_bins, **kwargs)
        
        mx_sz = self.Data['values_to_count'].max()
        bins = np.logspace(0, np.ceil(np.log10(mx_sz)), self.n_bins+1)

        # Calculate histogram
        hist = np.histogram(self.Data['values_to_count'], bins=bins)
        # normalize by bin width
        hist_norm = hist[0]/hist[0].sum()

        ix = hist_norm>0
        x=(bins[0:-1]+bins[1:])/2
        self.log_size=np.log10(x[ix])
        self.log_freq = np.log10(hist_norm[ix])

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

        self.trend_line = trend_line
        if trend_line is None:
            pass
        elif self.trend_line == "piecewise_linear":
            self.p, e = optimize.curve_fit(piecewise_linear, self.log_size, self.log_freq)
            self.exponents = self.p[2:4]
        elif self.trend_line == "linear": 
            self.p = np.polyfit(self.log_size, self.log_freq, 1)
            self.exponents = self.p[0]
            pass
        
    def plot(self):
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
        
        # plot it!
        self.ax.plot(10**self.log_size, 10**self.log_freq,'.')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Value',fontsize=18)
        self.ax.set_ylabel('Freq',fontsize=18)
        if self.trend_line == "piecewise_linear": 
            self.ax.plot(10**self.log_size, 10**piecewise_linear(self.log_size, *self.p),'r-')
            self.ax.set_title(f"Exponents: {self.exponents[0]:.2f} {self.exponents[1]:.2f}")
        elif self.trend_line == "linear": 
            self.ax.plot(10**self.log_size, 10**(self.p[0]*self.log_size+self.p[1]),'r-')
            self.ax.set_title(f"Exponent: {self.exponents[0]:.2f}")


class Zoom(Panel):
    def __init__(self, panel_to_zoom, zoom_coords=np.array([0,0,1,1]), name=None, pos=(0,0,1,1)):
        f = type(panel_to_zoom) # Type of the Panel
        self.ZP = f()
        
        for attr in dir(panel_to_zoom):
            try:
                setattr(self.ZP , attr, getattr(panel_to_zoom, attr))
            except:
                print(f"cannot copy attribute: {attr}")

        self.ZP.ax = None
        self.pos = pos
        self.name = name
        self.zoom_coords = zoom_coords
        self.panel_to_zoom = panel_to_zoom

        
    def plot(self):
        # add boundary around zoomed area
        self.panel_to_zoom.ax.add_patch(pRectangle((self.zoom_coords[0], self.zoom_coords[1]),
                                   self.zoom_coords[2], self.zoom_coords[3],
                                   fc ='none', 
                                   ec ="w",
                                   lw = 3))
        self.ZP.ax = self.ax
        self.ZP.plot()
        self.ax.set_xlim(self.zoom_coords[0],self.zoom_coords[0]+self.zoom_coords[2])
        self.ax.set_ylim(self.zoom_coords[1],self.zoom_coords[1]+self.zoom_coords[3])
