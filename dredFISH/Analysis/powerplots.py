
### A SET OF PLOTTING FUNCTIONS INSPIRED BY EXPLORING VIZGEN MERFISH
### --- START OF VIZGEN MERFISH SECTION
import numpy as np
import datashader as ds
import colorcet
import json

from .__init__plots import *

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
            self.npxlx = int(rangex/pxl_scale)
            self.npxly = int(rangey/pxl_scale)
        if npxlx:
            assert isinstance(npxlx, int)
            self.npxlx = npxlx
            self.pxl_scale = rangex/npxlx 
            self.npxly = int(rangey/pxl_scale)
        if npxly:
            assert isinstance(npxly, int)
            self.npxly = npxly
            self.pxl_scale = rangey/npxly 
            self.npxlx = int(rangex/pxl_scale)

        self.num_pxl = self.npxlx*self.npxly
        self.pxl_scale_ux = rangex/self.npxlx
        self.pxl_scale_uy = rangey/self.npxly 
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

class CategoricalColors:
    """
    Arguments: labels, [colors]
    """
    def __init__(self, labels, colors=[], basis_cmap=colorcet.cm.rainbow):
        """
        """
        self.labels = labels
        self.indices = np.arange(len(labels))
        if not colors:
            self.colors = basis_cmap(np.linspace(0, 1, len(self.indices)))
            # colors = colorcet.cm.glasbey(np.arange(len(indices)))
            # colors = sns.color_palette('husl', len(indices))
        else:
            self.colors = colors
        assert len(self.labels) == len(self.colors)
        
        self.gen_cmap()
        
    def gen_cmap(self):
        """Use a list of colors to generate a categorical cmap
        which maps 
            [0, 1) -> self.colors[0]
            [1, 2) -> self.colors[1]
            [2, 3) -> self.colors[2]
            [3, 4) -> self.colors[3]
            ...
        """
        self.cmap = mpl.colors.ListedColormap(self.colors)
        self.bounds = np.arange(len(self.colors)+1)
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

    def add_colorbar(
        self,
        fig, 
        cax_dim=[0.95, 0.1, 0.05, 0.8],
        shift=0.5,
        fontsize=10,
        **kwargs,
        ):
        """
        """
        cax = fig.add_axes(cax_dim)
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=self.cmap, norm=self.norm),
            cax=cax, 
            boundaries=self.bounds, 
            ticks=self.bounds[:-1]+shift,
            drawedges=True,
            **kwargs,
            )
        cbar.ax.set_yticklabels(self.labels, fontsize=fontsize)
        cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
        return 
    
    def to_dict(self, to_hex=True, output=""):
        """
        """
        if to_hex:
            self.palette = {label: mpl.colors.to_hex(color) 
                        for label, color 
                        in zip(self.labels, self.colors)
                    }
        else:
            self.palette = {label: color 
                        for label, color 
                        in zip(self.labels, self.colors)
                    }

        if output:
            with open(output, 'w') as fh:
                json.dump(self.palette, fh)
                print("saved to file: {}".format(output))

        return self.palette
    
def agg_data(
    data, 
    x, y, 
    npxlx, npxly, 
    agg,
    ):
    """
    """
    aggdata = ds.Canvas(plot_width=npxlx, plot_height=npxly).points(data, x, y, agg=agg)
    return aggdata

def agg_data_count(
    data, 
    x, y, 
    npxlx, npxly,
    ):
    agg = ds.count()
    aggdata = agg_data(data, x, y, npxlx, npxly, agg)
    agg = ds.any()
    aggdata_any = agg_data(data, x, y, npxlx, npxly, agg)
    aggdata = aggdata/aggdata_any
    return aggdata

def agg_data_ps(data, x, y, agg, scale_paras):
    """
    """
    # main

    rangex = data[x].max() - data[x].min()
    rangey = data[y].max() - data[y].min()
    ps = PlotScale(rangex, rangey, **scale_paras)
    aggdata = agg_data(data, x, y, ps.npxlx, ps.npxly, agg,)

    return aggdata, ps

def agg_count_cat(
    data, x, y, z, scale_paras, 
    clip_max=0, 
    reduce=False,
    sharp_boundary=True, 
    ):
    """count categorical data
    """
    # collect aggdata and ps
    agg = ds.count_cat(z)
    aggdata, ps = agg_data_ps(data, x, y, agg, scale_paras)
    zlabels = aggdata[z].values
   
    if clip_max:
        aggdata = aggdata.clip(max=clip_max)
        
    if reduce:
        aggdata = aggdata.argmax(z)
        
    if sharp_boundary:
        # normalize by any (set no cells to nan)
        agg = ds.any()
        aggdata_any = agg_data(data, x, y, ps.npxlx, ps.npxly, agg)
        aggdata_any = aggdata_any.astype(int)
        aggdata = aggdata/aggdata_any
    
    return aggdata, ps, zlabels

def set_vmin_vmax(
    numbers, vmaxp=99
    ):
    """
    """
    vmin, vmax = 0, np.nanpercentile(numbers, vmaxp)
    return vmin, vmax 
    
def add_colorbar_unified_colorbar(
    fig, cax, 
    vmin=0, vmax=0,
    cmap=sns.cubehelix_palette(as_cmap=True),
    **kwargs,
    ):
    """User specified vmin and vmax
    """
      # colorbar
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm,)
    fig.colorbar(sm, cax=cax, 
                 ticks=[vmin, vmax],
                 label='Normalized expression', 
                 **kwargs,
                )
    return 

def add_colorbar(
    fig, cax, 
    vmaxp=99, 
    cmap=sns.cubehelix_palette(as_cmap=True),
    **kwargs,
    ):
    """[log10(normalized_counts+1)] further normed by the 99% highest expression)
    """
      # colorbar
    norm = plt.Normalize(0, vmaxp)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm,)
    fig.colorbar(sm, cax=cax, 
                 ticks=[0, vmaxp],
                 label='Normalized expression\n(normed by 99% highest expression)', 
                 **kwargs,
                )
    return 

def imshow_routine(
    ax, 
    aggdata,
    cmap=sns.cubehelix_palette(as_cmap=True),
    vmin=None, vmax=None,
    origin='lower',
    aspect='equal',
    **kwargs
    ):
    """
    """
    ax.imshow(aggdata, aspect=aspect, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax, **kwargs)
    ax.axis('off')
    return ax 

def spatial_histogram(ax, data, x, y, scale_para, 
    cmap=sns.cubehelix_palette(as_cmap=True),
    vmin=0,
    vmax=0,
    vmaxp=100,
    agg=ds.count(),
    ):
    """
    """
    rangex = data[x].max() - data[x].min()
    rangey = data[y].max() - data[y].min()
    ps = PlotScale(rangex, rangey, **scale_para)

    if isinstance(agg, str) and agg == 'mycount':
        aggdata = agg_data_count(data, x, y, ps.npxlx, ps.npxly)
    else:
        aggdata = agg_data(data, x, y, ps.npxlx, ps.npxly, agg)

    if vmin==0 and vmax==0:
        vmin, vmax = set_vmin_vmax(aggdata.values, vmaxp)

    imshow_routine(ax, aggdata, cmap=cmap, 
                vmin=vmin, vmax=vmax,
                )
    return ax, ps, aggdata, vmin, vmax

def massive_scatterplot(
    ax, 
    data, 
    x, y,  
    npxlx, npxly, 
    agg=ds.count(),
    cmap=sns.cubehelix_palette(as_cmap=True),
    vmin=0,
    vmax=0,
    vmaxp=99,
    ):
    """
    """
    if isinstance(agg, str) and agg == 'mycount':
        aggdata = agg_data_count(data, x, y, npxlx, npxly)
    else:
        aggdata = agg_data(data, x, y, npxlx, npxly, agg)
    
    if vmin==0 and vmax == 0:
        vmin, vmax = set_vmin_vmax(aggdata.values, vmaxp)
        imshow_routine(ax, aggdata, cmap=cmap, 
                    vmin=vmin, vmax=vmax,
                    )
    else:
        imshow_routine(ax, aggdata, cmap=cmap, 
                    vmin=vmin, vmax=vmax,
                    )

    return ax, aggdata

def massive_scatterplot_withticks(
    ax, data, x, y, npxlx, npxly, 
    aspect='auto',
    color_logscale=False,
    ):
    xmin, xmax = data[x].min(), data[x].max()
    ymin, ymax = data[y].min(), data[y].max()
    aggdata = agg_data_count(data, x, y, npxlx, npxly)
    if color_logscale:
        aggdata = np.log10(aggdata)
    ax.imshow(
        aggdata, origin='lower', aspect=aspect, 
        extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return 

def add_arrows(
    ax, label, 
    fontsize=15,
    px=-0.01, 
    py=-0.01,
    ):
    """
    """
    # arrows
    ax.arrow(px, py, 0, 0.1,
             transform=ax.transAxes,
             head_width=0.01, head_length=0.01, 
             fc='k', ec='k', clip_on=False,)
    ax.arrow(px, py, 0.1, 0,
             transform=ax.transAxes,
             head_width=0.01, head_length=0.01, 
             fc='k', ec='k', clip_on=False,)
    ax.text(px, py-0.01, label, 
            transform=ax.transAxes,
            va='top', ha='left', 
            fontsize=fontsize)
    # end arrows
    return ax

def add_scalebar(
    ax, left, right, label, fontsize=15, 
    ax_y=-0.01, 
    ):
    """
    """
    ax.hlines(ax_y, left, right, color='k', linewidth=3, 
              transform=ax.get_xaxis_transform(),
              clip_on=False,
              )
    ax.text(right, ax_y-0.01, label, 
            va='top', ha='right', 
            transform=ax.get_xaxis_transform(),
            fontsize=fontsize)
    # end scale bar
    return ax

def plot_gene_insitu_routine(
    ax, data, x, y, hue, scale_paras, cmap, title, 
    arrows=True, scalebar=True, 
    vmaxp=99,
    vmin=0, vmax=0,
    ):
    """
    """
    # main
    agg = ds.mean(hue) 

    rangex = data[x].max() - data[x].min()
    rangey = data[y].max() - data[y].min()
    ps = PlotScale(rangex, rangey, **scale_paras)
    massive_scatterplot(
        ax, data, x, y, ps.npxlx, ps.npxly, 
        agg=agg, 
        cmap=cmap,
        vmaxp=vmaxp,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    # arrows
    if arrows:
        add_arrows(ax, 'in situ')
    # scale bar
    if scalebar:
        bar_length = 1000 # (micron)
        add_scalebar(ax, ps.npxlx-ps.len2pixel(bar_length), ps.npxlx, '1 mm')
    return ax

def plot_gene_umap_routine(
    ax, data, x, y, hue, scale_paras, cmap, title, 
    arrows=True, 
    vmaxp=99,
    ):
    """
    """
    # main
    agg = ds.mean(hue) 

    rangex = data[x].max() - data[x].min()
    rangey = data[y].max() - data[y].min()
    ps = PlotScale(rangex, rangey, **scale_paras)
    massive_scatterplot(ax, data, x, y, ps.npxlx, ps.npxly, 
                         agg=agg, 
                         cmap=cmap,
                         vmaxp=vmaxp,
                        )
    ax.set_title(title)
    # arrows
    if arrows:
        add_arrows(ax, 'UMAP', px=-0.03, py=-0.03)

    return ax

def plot_cluster_insitu_routine(
    ax, 
    ps,
    aggdata,
    hue, 
    zlabel,
    title,
    cmap, 
    arrows=True, scalebar=True, 
    ):
    """
    ps - an instance of PlotScale
    """
    zlabels = aggdata.coords[hue].values
    i = np.where(zlabels==zlabel)[0][0]
    imshow_routine(
        ax, 
        aggdata[:,:,i],
        cmap=cmap,
    )
    ax.set_title(title)
    # arrows
    if arrows:
        add_arrows(ax, 'in situ')
    # scale bar
    if scalebar:
        bar_length = 1000 # (micron)
        add_scalebar(ax, ps.npxlx-ps.len2pixel(bar_length), ps.npxlx, '1 mm')

    return ax

def plot_cluster_umap_routine(
    ax, 
    ps,
    aggdata,
    hue, 
    zlabel,
    title,
    cmap, 
    arrows=True, scalebar=True, 
    ):
    """
    ps - an instance of PlotScale
    """
    zlabels = aggdata.coords[hue].values
    i = np.where(zlabels==zlabel)[0][0]
    imshow_routine(
        ax, 
        aggdata[:,:,i],
        cmap=cmap,
    )
    ax.set_title(title)
    # arrows
    if arrows:
        add_arrows(ax, 'UMAP')

    return ax
### END OF VIZGEN MERFISH SECTION

def gen_cdf(array, ax, x_range=[], n_points=1000, show=True, flip=False, **kwargs):
    """ returns x and y values
    """
    x = np.sort(array)
    y = np.arange(len(array))/len(array)
    if flip:
        # x = x[::-1]
        y = 1 - y

    if not x_range:
        if show:
            ax.plot(x, y, **kwargs)
        return x, y 
    else:
        start, end = x_range
        xbins = np.linspace(start, end, n_points)
        ybins = np.interp(xbins, x, y)
        if show:
            ax.plot(xbins, ybins, **kwargs)
        return xbins, ybins 

def savefig(fig, path):
    """
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
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