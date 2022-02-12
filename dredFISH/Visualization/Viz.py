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

def scale_vec(x, low, high):
    """
    Scale vector between bounds

    Input
    -----
    x : 1D vector
    low : minimum value after rescaling
    high : maximum value after rescaling

    Output
    ------
    NumPy array : rescaled values 
    """
    return ((high - low) * (x - x.min()) / (x.max() - x.min())) + low 

def rgb2hex(rgb):
    """
    Convert RGB triples into hex code 

    Input
    -----
    rgb : either a single rgb (tuple or list) or a numpy array of shape Nx3

    Output
    ------
    hex : list of hex codes for the RGB colors
    """
    f_rgb2hex= lambda rgb: '#%02x%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255),int(rgb[3]*255))
    hex = list()
    if type(rgb).__module__ == np.__name__:
        for i in rgb.shape[0]:
            hex.append(f_rgb2hex(rgb[i,:]))
    else: 
        hex = f_rgb2hex(rgb)
        
    return(hex)

def color_from_dist_springs(D, lum = None):
    G = buildgraph(D,metric = 'precomputed',n_neighbors=3)
    if lum is None:
        layout = G.layout_fruchterman_reingold_3d()
        xyz = np.array(layout.coords)
        xyz[:,0] = scale_vec(xyz[:,0], -128, 127)
        xyz[:,1] = scale_vec(xyz[:,1], -128, 127)
        xyz[:,2] = scale_vec(xyz[:,2], 0, 100)
        xyz = np.array(layout.coords)
    else:
        layout = G.layout_fruchterman_reingold()
        xy = np.array(layout.coords)
        xy[:,0] = scale_vec(xy[:,0], -128, 127)
        xy[:,1] = scale_vec(xy[:,1], -128, 127)
        xyz = np.hstack((xy,np.ones((xy.shape[0],1))*lum))
    
     # convert to L*a*b* and from that to RGB
    rgb = np.zeros(xyz.shape)
    for i in range(xyz.shape[0]):
        lab = LabColor(lab_l=xyz[i,2],lab_a=xyz[i,0],lab_b=xyz[i,1])
        rgb_color = convert_color(lab, sRGBColor)
        rgb[i,:] = np.array((rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b))

    return rgb

def color_from_dist(D,lum = None):
    """
    Calculates a list of colors such that their perceptual distance matches the distance matrix D
    Uses MDS scaling to move from D to L*a*b* space. 
    
    Input
    -----
    D : distance matrix (NxN)
    
    Output
    ------
    a Nx3 numpy array with RGB colors, order of D is preserved. 
    """
    
    if lum == None:
        embedding = MDS(n_components=3,dissimilarity = 'precomputed')
        mds_coord = embedding.fit_transform(D)
        mds_coord[:,2] = scale_vec(mds_coord[:,0], 0, 100) 
    else: 
        embedding = MDS(n_components=2,dissimilarity = 'precomputed')
        mds_coord = embedding.fit_transform(D)
        mds_coord = np.stack((np.ones(D.shape[0])*lum,mds_coord))

    # rescale
    mds_coord[:,0] = scale_vec(mds_coord[:,0], -128, 127)
    mds_coord[:,1] = scale_vec(mds_coord[:,1], -128, 127)
    
    # convert to L*a*b* and from that to RGB
    rgb = np.zeros(mds_coord.shape)
    for i in range(mds_coord.shape[0]):
        lab = LabColor(lab_l=mds_coord[i,2],lab_a=mds_coord[i,0],lab_b=mds_coord[i,1])
        rgb_color = convert_color(lab, sRGBColor)
        rgb[i,:] = np.array((rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b))

    return rgb


class MapView:
    def __init__(self,TMG,name=None,**kwargs):
        # each view needs a unique name
        self.name = name
        
        # link to the TMG used to create the view
        self.TMG = TMG
        
        # a view is single layeout, so multiple axes on one figure
        self.axes = list()
        self.plotting_tasks = list()
        self.figs = list()
        self.figsize = (11,11)
        
        # self.zoom_border_colors=["y","w","c"]
        
        # Fundamentally, a view keeps tab of all the type for different geoms
        # and a dictionary that maps these ids to color/shape etc. 
        
        # types, maps each points, line, or polygon to a specific type key. 
        # in these dataframes, the index must match the TMG Geom and different columns store different attributes
        self.line_style = pd.DataFrame()
        self.point_style = pd.DataFrame()
        self.polygon_style = pd.DataFrame()
        self.boundingbox_style = pd.DataFrame()
        
        # colormap is avaliable in case some derived Views needs to use it (for coloring PolyCollection for example)
        self.clrmp = None
        
    def is_empty(self):
        return(self.line_style.empty and self.point_style.empty and self.polygon_style.empty)
    
    def set_view(self): 
        """
        Key abstract method - has to be implemented in the subclass
        signature should always include the TMG (and other stuff if needed)
        """
        return
        # raise NotImplementedError()
        
    
    def plot_boundingbox(self): 
        xy=np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T
        plt.plot(xy[:,0],xy[:,1],color=self.boundingbox_style['color'])
    
    
    def plot_points(self): 
        plt.scatter(x=self.Layers[0].X,
                    y=self.Layers[0].Y,
                    s=self.point_style['size'],
                    c=self.point_style['color'])
        
    def plot_polys(self): 
        p = PolyCollection(self.TMG.Geoms['poly'],cmap=self.clrmp)
        p.set_array(self.polygon_style['scalar'])
        ax = plt.gca()
        ax.add_collection(p)
        
    def plot_lines(self): 
        # get regions lines (subset of line geom)
        region_edge = self.TMG.find_regions_edge_level()
        segs = [self.TMG.Geoms['line'][edges] for edges in region_edge]
        segs = np.array(segs)
        
        line_segments = LineCollection(segs,
                                       linewidths=self.line_style['width'],
                                        colors=self.line_style['color'])
        ax = plt.gca()
        ax.add_collection(line_segments)
            
   
    def plot(self,**kwargs):
        """
        plot the View. 
        """
        
        if self.is_empty():
            raise TypeError('View was not initalized with set_view')
        
        # plotting of different geometries depending on styles that exist in the view
        # current supported geoms are: BoundingBox, lines, poly, points 
        # in each case, there are some assumption about the columns that exists in the view style. 
        # these assumptions are no enforced, so take care! 

        fig = plt.figure(figsize=self.figsize)
        self.figs.append(fig)
        ax = fig.add_subplot(111)
        
        if not self.boundingbox_style.empty:
            self.plot_boundingbox()
        
        if not self.point_style.empty: 
            self.plot_points()
            
        if not self.polygon_style.empty:
            self.plot_polys()
            
        if not self.line_style.empty:
            self.plot_lines()
        
        # set up limits and remove frame
        mx = np.max(np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        mn = np.min(np.array(self.TMG.Geoms['BoundingBox'].exterior.xy).T,axis=0)
        ax.set_xlim(mn[0],mx[0])
        ax.set_ylim(mn[1],mx[1])
        ax.axis('off')
                  
        zoom_coords = kwargs.get('zoom_coords',None)
        if zoom_coords is not None:
            # if len(zoom_coords.shape)==1:
            #     zoom_coords=zoom_coords[None,:]
            # for i in range(zoom_coords.shape[0])
            
            ax.add_patch(pRectangle((zoom_coords[0], zoom_coords[1]),
                                    zoom_coords[2], zoom_coords[3],
                                    fc ='none', 
                                    ec ="w",
                                    lw = 3))
                
            # copy the figure to a new figure and zoom
            buf = io.BytesIO()
            pickle.dump(fig, buf)
            buf.seek(0)
            fig_zoomed = pickle.load(buf) 
            ax_list = fig_zoomed.axes
            ax_list[0].set_xlim(zoom_coords[0],zoom_coords[0]+zoom_coords[2])
            ax_list[0].set_ylim(zoom_coords[1],zoom_coords[1]+zoom_coords[3])
            # # ax_list[0].spines["top"].set_color(self.zoom_border_colors[0])
            # # ax_list[0].spines["bottom"].set_color(self.zoom_border_colors[0])
            # # ax_list[0].spines["left"].set_color(self.zoom_border_colors[0])
            # # ax_list[0].spines["right"].set_color(self.zoom_border_colors[0])
            fig_zoomed
            self.figs.append(fig_zoomed)
                  
        
# for any new view, we derive the class so we have lots of views, each with it's own class so we can keep key attributes and 
# rewrite the different routines for each type of views

      
class Colorpleth(MapView):
    """
    Show a user-provided vector color coded on whatever geo-units requested (cells, iso-zones, heterozones, neighborhoods) 
    It will guess which level is needed from the size of the use provided values_to_map vector. 
    """
    def __init__(self,TMG,name = "Colorpleth",values_to_map = None):
        super().__init__(TMG,name = name)
        self.values_to_map = values_to_map
        lvlarr = np.flatnonzero(np.equal(self.TMG.N,len(self.values_to_map)))
        if len(lvlarr)!=1:
            raise ValueError('number of items in values_to_map doesnt match any level size')
        self.lvl = lvlarr[0]
        
    def set_view(self):
        scalar_mapping = self.TMG.map_to_cell_level(self.lvl,VecToMap = self.values_to_map)
        scalar_mapping = scalar_mapping-scalar_mapping.min()
        scalar_mapping = scalar_mapping/scalar_mapping.max()
        self.polygon_style['scalar'] = scalar_mapping
        self.clrmp = 'hot'
        

class RandomColorpleth(Colorpleth):
    """
    Show geo-units (cells, iso-zones, heterozones, neighborhoods) each colored randomly. 
    """
    def __init__(self,TMG,name = "Random Colorpleth", id_vec = 0):
        if len(np.shape(id_vec))==0: # if the id_vec is pointing to a current level, gove each unit it's own color 
            name = name + " of level: " + str(id_vec)
            id_vec = np.arange(TMG.N[id_vec])
        super().__init__(TMG,name = name, values_to_map = id_vec)
        
    def set_view(self):
        super().set_view()
        # create the colormap
        self.clrmp = ListedColormap(np.random.rand(len(self.values_to_map),3))        
        
        
class CellMap(MapView):
    def __init__(self,TMG,lvl = 1,name = "cell map"):
        super().__init__(TMG,name = name)
        self.lvl = lvl
        self.clr = None
        self.cmap_list = ['Purples','Oranges','Blues','Greens','Reds','cividis']
        
    def set_view(self):
        super().set_view()
        # reduce number of colormaps if we have very few types
        if self.TMG.Ntypes[self.lvl] < len(self.cmap_list):
            self.cmap_list=self.cmap_list[:self.TMG.Ntypes[self.lvl]]
            
        cell_mapped_types = self.TMG.map_to_cell_level(self.lvl)
        T2 = self.TMG.Layers[self.lvl].Type2
        Dsqr = self.TMG.Layers[self.lvl].Dtype
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
            
        self.polygon_style['scalar'] = cell_mapped_types
        self.clr = clr
        self.type2 = T2
        
        unq,cnt = np.unique(cell_mapped_types,return_counts = True)
        self.sz = cnt.astype('float')
        self.sz = self.sz/self.sz.mean()*500
        layout = self.TMG.Layers[self.lvl].FG.layout_fruchterman_reingold()
        xy = np.array(layout.coords)
        xy[:,0] = xy[:,0]-xy[:,0].min()
        xy[:,1] = xy[:,1]-xy[:,1].min()
        xy[:,0] = xy[:,0]/xy[:,0].max()
        xy[:,1] = xy[:,1]/xy[:,1].max()
        
        self.xy = xy
        self.clrmp = ListedColormap(clr)
        
    def plot(self,**kwargs):
        super().plot(**kwargs)
        
        # add a panel showing type as large points
        fig = plt.figure(figsize = self.figsize)
        self.figs.append(fig)
        plt.scatter(x = self.xy[:,0],y = self.xy[:,1],c=self.clr,s = self.sz)
        plt.xticks([], [])
        plt.yticks([], [])
        
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


class NeighborhoodMap(CellMap): 
    def __init__(self,TMG,name = "neighborhood map"):
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


class NeighborhoodMapWithLines(NeighborhoodMap):
    def __init__(self,TMG,name = "neighborhood map"):
        super(NeighborhoodMap,self).__init__(TMG,name = name)
        self.lvl=2
        
    def set_view(self):
        super().set_view()
        region_edge = self.TMG.find_regions_edge_level()
        self.line_style['width'] = np.ones(len(region_edge))
        self.line_style['color'] = "#6e736f"
        
    def plot(self,Vcellmap = None):
        super().plot(Vcellmap)
        

class IsoZones(Colorpleth):
    def __init__(self,TMG):
        super().__init__(TMG,name="isozones",values_to_map = np.log10(TMG.Layers[1].node_size))
        self.n_bins = 50
        
    def set_view(self):
        super().set_view()
        
        mx_sz = self.TMG.Layers[1].node_size.max()
        bins = np.logspace(0, np.ceil(np.log10(mx_sz)), self.n_bins+1)

        # Calculate histogram
        hist = np.histogram(self.TMG.Layers[1].node_size, bins=bins)
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
        super().plot()

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
        
        fig = plt.figure(figsize = self.figsize)
        self.figs.append(fig)

        # plot it!
        plt.plot(10**self.log_size, 10**self.log_freq,'.')
        plt.xscale('log')
        plt.yscale('log')

        plt.plot(10**self.log_size, 10**piecewise_linear(self.log_size, *self.p),'r-')

        plt.xlabel('Zone size',fontsize=18)
        plt.ylabel('Freq',fontsize=18)
        plt.title(f"Exponents: {self.exponents[0]:.2f} {self.exponents[1]:.2f}")

    
    
        
class RasterMap(MapView):

    def __init__(self, TMG, name="Raster", lvl=0, color_map=cc.cm.rainbow):
        super().__init__(TMG, name)

        # label 
        self.labels = self.TMG.map_to_cell_level(lvl) 

        # coordinates of Polygons 
        min_xy = self.TMG.Geoms["point"].min(0)
        self.coords = [x - min_xy if len(x) > 0 else np.nan for x in self.TMG.Geoms["poly"]]

        # init mask 
        self.box_bounds = (self.TMG.Geoms["point"].max(0) - min_xy + 100).astype(int)[::-1]
        self.mask = np.zeros(self.box_bounds)

        # colormap
        self.cm = color_map

    def set_view(self):
        """
        Build Raster matrix from polygons using 256 - label value (assumes # labels < 256)
        """
        polys = [(Polygon(x), 256 - self.labels[idx]) for idx, x in enumerate(self.coords) if type(x)!= float]
        self.mask = rasterize(polys, out_shape=self.box_bounds)
        

    def plot(self):
        """
        Plot Raster
        """
        plt.figure(figsize=self.figsize)
        plt.imshow(self.mask, cmap=self.cm)
        plt.xticks([], [])
        plt.yticks([], [])

