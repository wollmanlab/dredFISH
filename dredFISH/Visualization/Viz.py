"""
Module Viz deals with all TissueMultiGraph vizualization needs. 

Module has few accessory function and the View class hierarchy. V

"""

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import Rectangle as pRectangle
import colorcet as cc

import pickle
import io
import copy

import seaborn as sns

from scipy import optimize
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import jensenshannon, pdist, squareform

from sklearn.manifold import MDS
from rasterio.features import rasterize

# for debuding mostly
import warnings
import time
from IPython import embed

from dredFISH.Visualization.cell_colors import *
from dredFISH.Visualization.vor import * 

from dredFISH.Analysis.TissueGraph import *

from dataclasses import dataclass

class View:
    def __init__(self,TMG,name=None,figsize = (11,11), **kwargs):
        
        # each view needs a unique name
        self.name = name
        
        # link to the TMG used to create the view
        self.TMG = TMG
        
        # list of all panels
        self.Panels = list()
        
        self.figsize = figsize
        
    def add_panel(self,P):
        """
        add a panel to the view and initialized it by calling Panel's method set_view() 
        """
        P.V = self
        P.set_view()
        self.Panels.append(P)
                
            
    def show(self,**kwargs):
        """
        plot all panels. 
        """
        # set figures
        self.figsize = self.figsize
        self.fig = plt.figure(figsize = self.figsize)
        for i in range(len(self.Panels)): 
            # add an axes
            ax = self.fig.add_axes(self.Panels[i].pos)
            self.Panels[i].ax = ax
            plt.sca(ax)
            self.Panels[i].plot()


class Panel:
    def __init__(self,name = None,pos = (0,0,1,1)):
        self.V = None
        self.name = name
        self.pos = pos
        self.ax = None
        self.Data = {}
        self.Styles = {}
        self.clrmp = None

    def set_view(self):  
        """
        Key abstract method - has to be implemented in the subclass
        signature should always include the TMG (and other stuff if needed)
        """
        return
    
    def plot(self):
        """
        actual plotting happens here, overload in subclasses! 
        """
        return
        
class Map(Panel):
    """
    MapView extends View to allow plotting of maps, it adds the methods:
        plot_points
        plot_lines
        plot_bounding_box
        plot_polys
        plot_map
        plot_map_zoom
    """
    def __init__(self,name=None,pos = (0,0,1,1),**kwargs):
        super().__init__(name,pos,**kwargs)
        
        # Fundamentally, a view keeps tab of all the type for different geoms
        # and a dictionary that maps these ids to color/shape etc. 
        
        # types, maps each points, line, or polygon to a specific type key. 
        # in these dataframes, the index must match the TMG Geom and different columns store different attributes
        self.Styles['line'] = pd.DataFrame()
        self.Styles['point'] = pd.DataFrame()
        self.Styles['polygon'] = pd.DataFrame()
        self.Styles['boundingbox'] = pd.DataFrame()
        
        # colormap is avaliable in case some derived Views needs to use it (for coloring PolyCollection for example)
        self.clrmp = None
        
    def set_view(self): 
        """
        Key abstract method - has to be implemented in the subclass
        signature should always include the TMG (and other stuff if needed)
        """
        return
        
    
    def plot_boundingbox(self): 
        xy=np.array(self.V.TMG.Geoms['BoundingBox'].exterior.xy).T
        self.ax.plot(xy[:,0],xy[:,1],color=self.Styles['boundingbox']['color'])
    
    
    def plot_points(self): 
        self.ax.scatter(x=self.Layers[0].X,
                    y=self.Layers[0].Y,
                    s=self.Styles['point']['size'],
                    c=self.Styles['point']['color'])
        
    def plot_polys(self): 
        p = PolyCollection(self.V.TMG.Geoms['poly'],cmap=self.clrmp)
        p.set_array(self.Styles['polygon']['scalar'])
        self.ax.add_collection(p)
        
    def plot_lines(self): 
        # get regions lines (subset of line geom)
        region_edge = self.V.TMG.find_regions_edge_level()
        segs = [self.V.TMG.Geoms['line'][edges] for edges in region_edge]
        segs = np.array(segs)
        
        line_segments = LineCollection(segs,
                                       linewidths=self.Styles['line']['width'],
                                        colors=self.Styles['line']['color'])
        self.ax.add_collection(line_segments)
            
    def plot(self,**kwargs):
        """
        plot a map. 
        """
        if not self.Styles['boundingbox'].empty:
            self.plot_boundingbox()
        
        if not self.Styles['point'].empty: 
            self.plot_points()
            
        if not self.Styles['polygon'].empty:
            self.plot_polys()
            
        if not self.Styles['line'].empty:
            self.plot_lines()
        
        # set up limits and remove frame
        mx = np.max(np.array(self.V.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        mn = np.min(np.array(self.V.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        self.ax.set_xlim(mn[0],mx[0])
        self.ax.set_ylim(mn[1],mx[1])
        self.ax.axis('off')
        
class Zoom(Panel):
    def __init__(self,panel_to_zoom,zoom_coords = np.array([0,0,1,1]),name = None,pos = (0,0,1,1)):
        f=type(panel_to_zoom)
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
        self.panel_to_zoom = panel_to_zoom;
        
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

 
        
            
#         def set_view(self):
#             super().set_view()
            
#         def plot_map_zoom(self,**kwargs): 
#             ix_axes_to_zoom = kwargs.get('ax_to_zooom',0)
#             zoom_coords = kwargs.get('zoom_coords',None)
#             # add the boundary on the map axes
#             ax_zoom = plt.gca()
#             ax = self.Axes[ix_axes_to_zoom]
#             ax.add_patch(pRectangle((zoom_coords[0], zoom_coords[1]),
#                                      zoom_coords[2], zoom_coords[3],
#                                      fc ='none', 
#                                      ec ="w",
#                                      lw = 3))
        
#         plt.sca(ax_zoom)
#         self.plot_map(**kwargs)

        
class Colorpleth(Map):
    """
    Show a user-provided vector color coded on whatever geo-units requested (cells, iso-zones, heterozones, neighborhoods) 
    It will guess which level is needed from the size of the use provided values_to_map vector. 
    """
    def __init__(self,values_to_map,name=None,pos = (0,0,1,1),**kwargs):
        super().__init__(name=name,pos = pos,**kwargs)
        self.Data['values_to_map'] = values_to_map
        
    def set_view(self):
        # check that number of items in values to map make sense valu
        lvlarr = np.flatnonzero(np.equal(self.V.TMG.N,len(self.Data['values_to_map'])))
        if len(lvlarr)!=1:
            raise ValueError('number of items in values_to_map doesnt match any level size')
        self.lvl = lvlarr[0]
        
        scalar_mapping = self.V.TMG.map_to_cell_level(self.lvl,VecToMap = self.Data['values_to_map'])
        scalar_mapping = scalar_mapping-scalar_mapping.min()
        scalar_mapping = scalar_mapping/scalar_mapping.max()
        self.Styles['polygon']['scalar'] = scalar_mapping
        self.clrmp = 'hot'
        

class RandomColorpleth(Colorpleth):
    """
    Show geo-units (cells, iso-zones, heterozones, neighborhoods) each colored randomly. 
    """
    def __init__(self,id_vec = 0, name = "Random Colorpleth" ,pos = (0,0,1,1)):
        if len(np.shape(id_vec))==0: # if the id_vec is pointing to a current level, give each unit it's own color 
            name = name + " of level: " + str(id_vec)
            id_vec = np.arange(TMG.N[id_vec])
        super().__init__(name = name, values_to_map = id_vec)
        
    def set_view(self):
        super().set_view()
        self.clrmp = ListedColormap(np.random.rand(len(self.values_to_map),3))        
        
class LegendWithCircles(Panel): 
    def __init__(self,map_panel,name=None,pos = (0,0,1,1),**kwargs):
        super().__init__(name = name,pos = pos,**kwargs)
        self.map_panel = map_panel
        self.scale = 300
        
    def set_view(self):
        
        cell_mapped_types = self.map_panel.Data['values_to_map']
        unq,cnt = np.unique(cell_mapped_types,return_counts = True)
        self.sz = cnt.astype('float')
        self.sz = self.sz/self.sz.mean()*self.scale
        layout = self.V.TMG.Layers[self.map_panel.lvl].FG.layout_fruchterman_reingold()
        xy = np.array(layout.coords)
        xy[:,0] = xy[:,0]-xy[:,0].min()
        xy[:,1] = xy[:,1]-xy[:,1].min()
        xy[:,0] = xy[:,0]/xy[:,0].max()
        xy[:,1] = xy[:,1]/xy[:,1].max()
        
        self.xy = xy
        self.clrmp = self.map_panel.clrmp
    
        
    def plot(self): 
        plt.sca(self.ax)
        plt.scatter(x = self.xy[:,0],y = self.xy[:,1],c=self.map_panel.clr,s = self.sz)
        plt.xticks([], [])
        plt.yticks([], [])
        
class LegendWithCirclesAndWedges(LegendWithCircles):
    
    def __init__(self,map_panel,cell_map_panel,name = None,pos = (0,0,1,1),**kwargs):
        super().__init__(map_panel,name = name,pos = pos,**kwargs)
        self.cell_map_panel = cell_map_panel
                               
    def plot(self): 
        # get fractions and sort by type2
        feature_type_mat = self.V.TMG.Layers[self.map_panel.lvl].feature_type_mat
        ordr = np.argsort(self.cell_map_panel.type2)
        feature_type_mat = feature_type_mat[:,ordr]
        sum_of_rows = feature_type_mat.sum(axis=1)
        feature_type_mat = feature_type_mat / sum_of_rows[:, None]
            
        # scale between radi in points to xy that was normed to 0-1
        scale_factor = self.V.fig.dpi * self.V.figsize[0] 
            
        xy = self.xy*scale_factor
        radi = np.sqrt(self.sz/np.pi) * self.scale/100
        cdf_in_angles = np.cumsum(np.hstack((np.zeros((feature_type_mat.shape[0],1)),feature_type_mat)),axis=1)*360

        wedges = list()
        wedge_width = 0.66
        for i in range(feature_type_mat.shape[0]):
            for j in range(feature_type_mat.shape[1]):
                w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
                           cdf_in_angles[i,j+1],width = wedge_width*radi[i], facecolor = self.cell_map_panel.clr[ordr[j],:])
                c = Circle((xy[i,0],xy[i,1]),wedge_width*radi[i],facecolor = self.map_panel.clr[i,:],fill = True) 
                wedges.append(w)
                wedges.append(c)


        p = PatchCollection(wedges,match_original=True)
        self.ax.add_collection(p)

        margins = 0.05
        self.ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
class TypeMap(Map):
    def __init__(self,lvl = 1,name = "cell map",pos = (0,0,1,1),**kwargs):
        super().__init__(name = name,pos = pos,**kwargs)
        self.clr = None
        self.lvl = lvl
        self.cmap_list = ['Purples','Oranges','Blues','Greens','Reds','cividis']
        
    def set_view(self):
        super().set_view()
        
        cell_mapped_types = self.V.TMG.map_to_cell_level(self.lvl)
        self.Data['values_to_map'] = cell_mapped_types
        
        # reduce number of colormaps if we have very few types
        if self.V.TMG.Ntypes[self.lvl] < len(self.cmap_list):
            self.cmap_list=self.cmap_list[:self.V.TMG.Ntypes[self.lvl]]
        
        T2 = self.V.TMG.Layers[self.lvl].Type2
        Dsqr = self.V.TMG.Layers[self.lvl].Dtype
        clr = np.zeros((len(T2),4))
        for i in range(len(self.cmap_list)):
            cmap = cm.get_cmap(self.cmap_list[i])
            ix = np.flatnonzero(T2==i)
            if len(ix)<3: 
                value = np.linspace(0.2,1,len(ix))
            else:
                d = Dsqr[np.ix_(ix,ix)]
                dvec = squareform(d)
                z = linkage(dvec,method='average')
                ordr = optimal_leaf_ordering(z,dvec)

                n=len(ix)
                shift = np.ceil(0.2*n)
                value = (np.arange(len(ix))+shift+1)/(len(ix)+shift)
            clr[ix,:] = cmap(value)
            
        self.Styles['polygon']['scalar'] = self.Data['values_to_map']
        self.clr = clr
        self.type2 = T2
        self.clrmp = ListedColormap(clr)
        
  

  # commented section was used to plot Cond Entropy Vs resolution, not clear if that is a panel
    
#         # add a panel with conditional entropy
#         if hasattr(self.TMG.Layers[0], 'cond_entropy_df') and self.lvl==1:
#             EntropyCalcsL1 = self.TMG.Layers[0].cond_entropy_df
#             fig = plt.figure()
         
#             ax1 = plt.gca()
#             yopt = self.TMG.cond_entropy[1]
#             xopt = self.TMG.Ntypes[1]
#             ax1.plot(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
#             ylm = ax1.get_ylim()
#             ax1.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
#             ax1.set_xlabel('# of types',fontsize=18)
#             ax1.set_ylabel('H (Map | Type)',fontsize=18)
#             fig = plt.gcf()
#             left, bottom, width, height = [0.6, 0.55, 0.25, 0.25]
#             ax2 = fig.add_axes([left, bottom, width, height])
#             ax2.semilogx(EntropyCalcsL1['Ntypes'],EntropyCalcsL1['Entropy'])
#             ylm = ax2.get_ylim()
#             ax2.plot([xopt,xopt],[ylm[0], yopt],'r--',linewidth=1)
            
#             fig = plt.figure()
#             unq,cnt = np.unique(self.TMG.Layers[0].Type,return_counts=True)
#             plt.hist(cnt,bins=15);
#             plt.title("Cells per type")
#             plt.xlabel("# Cells in a type")
#             plt.ylabel("# of Types")


# class RegionMap(CellMap): 
#     def __init__(self,name = "region map"):
#         super().__init__(TMG,name = name)
#         self.lvl = 2
    
#     def set_view(self):
#         super().set_view()
        
#     def plot(self,V1 = None,**kwargs):
#         super().plot(**kwargs)
#         if V1 is None:
#             return
#         # add another panel with piechart markers
#         # start new figure (to calc size factor)
#         fig = plt.figure(figsize = self.figsize)
#         self.figs.append(fig)
#         ax = plt.gca()
            
#         # get fractions and sort by type2
#         feature_type_mat = self.TMG.Layers[self.lvl].feature_type_mat
#         ordr = np.argsort(V1.type2)
#         feature_type_mat = feature_type_mat[:,ordr]
#         sum_of_rows = feature_type_mat.sum(axis=1)
#         feature_type_mat = feature_type_mat / sum_of_rows[:, None]
            
#         # scale between radi in points to xy that was normed to 0-1
#         scale_factor = fig.dpi * self.figsize[0]
            
#         xy = self.xy*scale_factor
#         radi = np.sqrt(self.sz/np.pi)
#         cdf_in_angles = np.cumsum(np.hstack((np.zeros((feature_type_mat.shape[0],1)),feature_type_mat)),axis=1)*360

#         wedges = list()
#         wedge_width = 0.66
#         for i in range(feature_type_mat.shape[0]):
#             for j in range(feature_type_mat.shape[1]):
#                 w = Wedge((xy[i,0],xy[i,1]), radi[i], cdf_in_angles[i,j], 
#                            cdf_in_angles[i,j+1],width = wedge_width*radi[i], facecolor = V1.clr[ordr[j],:])
#                 c = Circle((xy[i,0],xy[i,1]),wedge_width*radi[i],facecolor = self.clr[i,:],fill = True) 
#                 wedges.append(w)
#                 wedges.append(c)


#         p = PatchCollection(wedges,match_original=True)
#         ax.add_collection(p)

#         margins = 0.05
#         ax.set_xlim(-margins*scale_factor,(1+margins)*scale_factor)
#         ax.set_ylim(-margins*scale_factor,(1+margins)*scale_factor)
#         ax.set_xticks([])
#         ax.set_yticks([])

# next section could be used to show mapping between regions and cell
        
        # fig = plt.figure(figsize=(10,10))
        # self.figs.append(fig)
        # region_cell_types = self.TMG.Layers[2].feature_type_mat
        # row_sums = region_cell_types.sum(axis=1)
        # row_sums = row_sums[:,None]
        # region_cell_types_nrm=region_cell_types/row_sums
        # g = sns.clustermap(region_cell_types_nrm,method="ward", cmap="mako",col_colors=V1.clr,row_colors = self.clr)
        # g.ax_heatmap.set_xticks(list())
        # g.ax_heatmap.set_yticks(list())
        # self.figs.append(fig)


class TypeMapWithLines(TypeMap):
    def __init__(self,lvl=2,pos = (0,0,1,1),name = "neighborhood map"):
        super().__init__(lvl=lvl,pos=pos,name = name)
        self.lvl=lvl
        
    def set_view(self):
        super().set_view()
        region_edge = self.V.TMG.find_regions_edge_level()
        self.Styles['line']['width'] = np.ones(len(region_edge))
        self.Styles['line']['color'] = "#6e736f"
        
    def plot(self,Vcellmap = None):
        super().plot()
        

class IsoZones(Colorpleth):
    """
    simple overloading of colorpleth, only different is that we're getting values from TMG 
    since TMG is only defined after init we pass the Data during set_view instead. 
    """
    def __init__(self,name = "isozones", pos = (0,0,1,1)):
        values_to_map = None
        super().__init__(values_to_map,name = name,pos = pos)
    def set_view(self):
        self.Data['values_to_map'] = np.log10(self.V.TMG.Layers[1].node_size)
        super().set_view()
        
class Histogram(Panel):
    def __init__(self,values_to_count,name = 'hist',pos = (0,0,1,1),n_bins = 50,**kwargs):
        self.n_bins = 50
        self.values_to_count = values_to_count
        self.pos = pos
        self.name = name
        
    def plot(self): 
        self.ax.hist(self.values_to_count,bins = self.n_bins)
    
        
class LogLogPlot(Histogram):
    def __init__(self,values_to_count,name = "loglog", pos = (0,0,1,1),**kwargs):
        super().__init__(values_to_count, name = name, pos = pos,**kwargs)
        
    def set_view(self):
        super().set_view()
        
        mx_sz = self.values_to_count.max()
        bins = np.logspace(0, np.ceil(np.log10(mx_sz)), self.n_bins+1)

        # Calculate histogram
        hist = np.histogram(self.values_to_count, bins=bins)
        # normalize by bin width
        hist_norm = hist[0]/hist[0].sum()

        ix = hist_norm>0
        x=(bins[0:-1]+bins[1:])/2
        self.log_size=np.log10(x[ix])
        self.log_freq = np.log10(hist_norm[ix])

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

        self.p , e = optimize.curve_fit(piecewise_linear, self.log_size, self.log_freq)
        self.exponents = self.p[2:4]
        
    def plot(self):
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
        
        # plot it!
        self.ax.plot(10**self.log_size, 10**self.log_freq,'.')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        self.ax.plot(10**self.log_size, 10**piecewise_linear(self.log_size, *self.p),'r-')

        self.ax.set_xlabel('Value',fontsize=18)
        self.ax.set_ylabel('Freq',fontsize=18)
        self.ax.set_title(f"Exponents: {self.exponents[0]:.2f} {self.exponents[1]:.2f}")
