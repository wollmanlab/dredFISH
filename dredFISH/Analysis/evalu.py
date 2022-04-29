import pandas as pd
import numpy as np
import umap # not used much

import matplotlib.pyplot as plt
import seaborn as sns

from .__init__plots import *
from . import basicu
from . import powerplots

class Level3Res:
    def __init__(self, meta, levels, 
                 refdata_anndata, data_anndata,  
                 refdata_name='scRNA-seq', data_name='dredFISH',
                 data_layer='norm_cell',
                 embed=False, df_embed='',
                ):        
        """
        """
        self.meta = meta
        self.data = data_anndata
        self.refdata = refdata_anndata
        self.levels = levels
        
        self.X = self.refdata.X
        if data_layer == 'norm_cell':
            self.Y = self.data.layers['norm_cell']
        else:
            self.Y = self.data.X
        
        self.Xobs = self.refdata.obs
        self.Yobs = self.data.obs
        
        self.Xname = refdata_name
        self.Yname = data_name
        if embed:
            self.df_embed = df_embed
    
    def get_cluster_level(self):
        """
        """
        # cluster mean
        level = self.levels[2]
        row_order = self.meta['l3_clsts']
        col_order = self.meta['l3_bits']

        # cluster mean reference
        # refdata (X)
        Xclst, _ = basicu.group_mean(
            self.X, 
            self.Xobs[level].values, 
            row_order)
        Xclst = basicu.zscore(Xclst, allow_nan=True, axis=0)
        Xclst = Xclst[:,col_order]
        Xclst = pd.DataFrame(Xclst, index=row_order, columns=col_order)

        # data (Y)
        Yclst, _ = basicu.group_mean(
            self.Y,
            self.Yobs[level].values,
            row_order)
        Yclst = basicu.zscore(Yclst, allow_nan=True, axis=0)
        Yclst = Yclst[:,col_order]
        Yclst = pd.DataFrame(Yclst, index=row_order, columns=col_order)
        
        return Xclst, Yclst
    
    def get_cell_level(self, nsample=0, mode='Y', norm_bits=True, reorder_bits=True):
        """
        """
        # random samples of cells
        col_order = self.meta['l3_bits']
        mat = getattr(self, mode)
        obs = getattr(self, mode+'obs')
        
        if nsample > 0:
            # sample
            level = self.levels[2]
            row_order = self.meta['l3_clsts']
            
            dfsub = basicu.stratified_sample(obs, level, nsample, group_keys=True).reindex(row_order, level=0) # .reset_index()
            cellids = dfsub.index.get_level_values(1)
            cellintids = basicu.get_index_from_array(obs.index.values, cellids)
            clst_sizes = dfsub[level].value_counts().reindex(row_order)

            if norm_bits:
                mat_cell = basicu.zscore(mat, allow_nan=True, axis=0)
            else:
                mat_cell = mat

            # sample
            mat_cell = mat_cell[cellintids]

            if reorder_bits:
                mat_cell = mat_cell[:,col_order]

            return mat_cell, clst_sizes, dfsub
        
        else:
            if norm_bits:
                mat_cell = basicu.zscore(mat, allow_nan=True, axis=0)
            else:
                mat_cell = mat

            if reorder_bits:
                mat_cell = mat_cell[:,col_order]
            return mat_cell, '', ''
            
    def get_umap(self, **kwargs):
        """Working but slow
        """
        Xcell, _, _ = self.get_cell_level(mode='X')
        Ycell, _, _ = self.get_cell_level(mode='Y')
        
        # run UMAP
        embed = umap.UMAP(**kwargs).fit_transform(
            np.vstack([Xcell, Ycell])
            )
        
        dfembed = pd.concat([self.Xobs, self.Yobs], axis=0) 
        dfembed['dataset'] = [self.Xname]*len(self.Xobs) + [self.Yname]*len(self.Yobs) 
        dfembed['embed_1'] = embed[:,0]
        dfembed['embed_2'] = embed[:,1]
        self.df = dfembed 
        return dfembed 

    def plot_spatial(self):
        """
        """
        df = self.data.obs
        hue_col = self.levels[2]
        huegroup_col = self.levels[1]
        hue_order = self.meta['l3_clsts']
        huegroup_order = self.meta['l2_clsts']
        palette = self.meta['l3_palette']
        
        plot_validation_spatial_level3(df, hue_col, hue_order, palette)
        plot_validation_spatial_level3_expand(df, hue_col, hue_order, palette)
        plot_validation_spatial_level3_expand_v2(df, hue_col, huegroup_col, huegroup_order)
    

    def plot_genes(self, nsample=10):
        """
        """
        Xclst, Yclst = self.get_cluster_level()
        
        plot_validation_genes_level3(Xclst, Yclst, self.meta['l3_hlines'], self.meta['l3_vlines'], 
                                    xtitle=self.Xname, ytitle=self.Yname,
                                    )
        
        Ycell, clst_sizes, dfsub = self.get_cell_level(nsample=nsample, mode='Y')
        plot_validation_genes_level3_expand(Ycell, clst_sizes, self.meta['l3_vlines'], ylabel=self.Yname)
        
    def plot_embeds(self, hues=[]):
        """
        """
        df = self.df_embed
        if len(hues) == 0:
            hues = ['dataset'] + self.levels.tolist()
        n = len(hues)
        
        fig, axs = plt.subplots(1, n, figsize=(6*n,8), sharex=True, sharey=True)
        for ax, hue in zip(axs, hues):
            plot_embed(ax, df, hue)
        fig.subplots_adjust(wspace=0)
        plt.show()

def variance_explained(Xmat, Ymat):
    """
    """
    # var explained
    cond = ~np.any(Ymat.isnull(), axis=1) #.sum(axis=1) == 0 # not null
    Xmatval = Xmat[cond] 
    Ymatval = Ymat[cond]
    if (np.abs(np.power(Xmatval, 2).mean().mean() - 1) < 1e-2
    ):
        varexp = 1-np.power(Ymatval-Xmatval, 2).mean().mean()
    else:
        varexp = 1-np.power(Ymatval-Xmatval, 2).mean().mean()/np.power(Xmatval, 2).mean().mean() # not calculate
    return varexp 

def compare_cluster_pearsonr(Xmat, Ymat):
    """matched rows
    """
    # var explained
    cond = ~np.any(Ymat.isnull(), axis=1) #.sum(axis=1) == 0 # not null
    Xmatval = Xmat[cond].values
    Ymatval = Ymat[cond].values
    celltypes = Xmat[cond].index.values

    rs = basicu.corr_paired_rows_fast(Xmatval, Ymatval, offset=0, mode='pearsonr')

    return rs, celltypes

def plot_validation_genes_cluster_r(ax, Xmat, Ymat, xlabel='scRNA-seq', ylabel='dredFIS'):
    """
    """
    rs, types = compare_cluster_pearsonr(Xmat, Ymat)
    dfr = pd.DataFrame()
    dfr['r'] = rs
    dfr['type'] = types
    dfr['condition'] = f'{xlabel} vs {ylabel}'

    # rs, types
    sns.boxplot(data=dfr, x='condition', y='r', color='white', showfliers=False, 
                ax=ax,
            )
    sns.swarmplot(data=dfr, x='condition', y='r', color='black', s=5, ax=ax)
    ax.set_ylim([0,1])
    sns.despine(ax=ax)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('')

    meanr = np.mean(rs)
    ax.set_title(f'Mean r = {meanr:.2g}')
    return ax

def plot_validation_genes_level3(Xmat, Ymat, splitat, splitat_v, xtitle='scRNA-seq', ytitle='dredFISH',
    show_rows_in_both_only=False,
    figsize=(12,10),
    outpath="",
    ):
    """
    """
    # var explained
    varexp = variance_explained(Xmat, Ymat) 

    # plot 
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    cbar_ax = fig.add_axes([.97, .4, .01, .2])
    boxplot_ax = fig.add_axes([1.15, .3, .1, .4])
    plot_validation_genes_cluster_r(boxplot_ax, Xmat, Ymat, xlabel=xtitle, ylabel=ytitle)
    
    if show_rows_in_both_only:
        rows_cond = ~np.any(np.isnan(Ymat), axis=1)
        print(f"excluded: {Ymat[~rows_cond].index.values}")
        Xmat = Xmat[rows_cond]
        Ymat = Ymat[rows_cond]
    else:
        # Xmat = Xmat.fillna(0)
        Ymat = Ymat.fillna(0)

    ax = axs[0]
    sns.heatmap(Xmat, 
                yticklabels=True, xticklabels=True, 
                vmax=3, vmin=-3, cmap='coolwarm', 
                cbar_ax=cbar_ax,
                cbar_kws=dict(label='normed features'),
                ax=ax)
    ax.set_title(f'{xtitle}')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
    # lines
    ax.hlines(1+np.array(splitat), 0, Ymat.shape[1], linewidth=0.5, color='gray')
    ax.vlines(1+np.array(splitat_v), 0, Ymat.shape[0], linewidth=0.5, color='gray')

    ax = axs[1]
    ax.yaxis.tick_right()
    sns.heatmap(Ymat, 
                # yticklabels=True, #[yl if yl == '**NA**' else "" for yl in Ymat.index], 
                xticklabels=True, 
                vmax=3, vmin=-3, cmap='coolwarm', 
                cbar=False,
                ax=ax)
    ax.set_title(f'{ytitle} (var explained = {varexp:.2f})')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    # ylabels
    ylabels = Ymat.index.values
    ax.set_yticks(0.5+np.arange(len(ylabels))[ylabels=='**NA**'])
    ax.set_yticklabels(ylabels[ylabels=='**NA**'], fontsize=10, rotation=0)
    # lines
    ax.hlines(1+np.array(splitat), 0, Ymat.shape[1], linewidth=0.5, color='gray')
    ax.vlines(1+np.array(splitat_v), 0, Ymat.shape[0], linewidth=0.5, color='gray')
    
    fig.subplots_adjust(wspace=0.05)


    if outpath:
        powerplots.savefig_autodate(fig, outpath)
    plt.show()


def plot_validation_genes_level3_expand(Ymat, clst_sizes, splitat_v, ylabel='dredFISH'):
    """
    """
    fig, ax = plt.subplots(figsize=(8,15))
    sns.heatmap(Ymat, 
                yticklabels=[], xticklabels=True, 
                vmax=3, vmin=-3, cmap='coolwarm', 
                # cbar_ax=cbar_ax,
                cbar_kws=dict(label='normed features', shrink=0.2),
                ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    
    for (clst, y) in np.cumsum(clst_sizes).iteritems():
        ax.text(0, y, clst, ha='right', fontsize=10)
        ax.axhline(y, color='white', linewidth=1)
    ax.vlines(1+np.array(splitat_v), 0, Ymat.shape[0], linewidth=0.5, color='white')
    ax.set_title(f"{ylabel} cells\n(n={clst_sizes.iloc[0]} cells per cell type)")
    
    plt.show()

def plot_validation_spatial_level3(df, hue_col, hue_order, palette):
    """
    """
    # hue_order = np.sort(np.unique(df[hue]))
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(data=df, #[cond],
                    x='coord_x', y='coord_y', 
                    hue=hue_col,
                    hue_order=hue_order,
                    palette=palette,
                    ax=ax,
                    s=1,
                    edgecolor='none',
                    rasterized=True,
                   )

    ax.legend(bbox_to_anchor=(1,1), 
              loc='upper left', 
              ncol=1+int(len(hue_order)/20), 
              title=hue_col,
             )
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()
    
# individual clusters (using Allen color)
def plot_validation_spatial_level3_expand(df, hue_col, hue_order, palette):
    """
    """
    nx = 6
    ny = 7

    fig, axs = plt.subplots(ny, nx, figsize=(4*nx,2.5*ny))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i, (cluster, ax) in enumerate(zip(hue_order, axs.flat)):
        cond = (df[hue_col] == cluster)
        sns.scatterplot(data=df, 
                        x='coord_x', y='coord_y', 
                        ax=ax,
                        s=.5,
                        edgecolor='none',
                        color='lightgray',
                        rasterized=True,
                       )
        sns.scatterplot(data=df[cond], #[cond],
                        x='coord_x', y='coord_y', 
                        # hue=hue,
                        # hue_order=hue_order,
                        ax=ax,
                        s=3,
                        alpha=0.5,
                        edgecolor='none',
                        color=palette[cluster],
                        rasterized=True,
                       )
        ax.set_title(cluster, loc='center', y=0, fontsize=15, va='bottom')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_aspect('equal')

    for ax in axs.flat[i:]:
        ax.axis('off')
    plt.show()
    
# individual clusters by neighborhoods (using Allen color)
def plot_validation_spatial_level3_expand_v2(df, hue_col, huegroup_col, huegroup_order):
    """
    """
    nx = 2
    ny = 4

    fig, axs = plt.subplots(ny, nx, figsize=(8*nx,5*ny))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i, (group, ax) in enumerate(zip(huegroup_order, axs.flat)):
        cond = (df[huegroup_col] == group)
        dfsub = df[cond]
        clsts = np.sort(np.unique(df[cond][hue_col]))
        
        sns.scatterplot(data=df, 
                        x='coord_x', y='coord_y', 
                        ax=ax,
                        s=1,
                        edgecolor='none',
                        color='lightgray',
                        rasterized=True,
                       )
        sns.scatterplot(data=dfsub, #[cond],
                        x='coord_x', y='coord_y', 
                        hue=hue_col,
                        hue_order=clsts,
                        cmap='husl',
                        ax=ax,
                        s=5,
                        alpha=0.5,
                        edgecolor='none',
                        rasterized=True,
                       )
        # ax.legend()
        # ax.set_title(group, loc='center', y=0, fontsize=15, va='bottom')
        # ax.set_title(group)
        ax.legend(bbox_to_anchor=(0.5,0.1), 
                  loc='lower center', 
                  ncol=1+int(len(clsts)/4), 
                  title=group,
                  fontsize=10,
                  title_fontsize=10,
                 )
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_aspect('equal')

    for ax in axs.flat[i:]:
        ax.axis('off')
    plt.show()
    
def plot_embed(ax, dfembed, hue):
    hue_order = np.sort(np.unique(dfembed[hue]))
    nums = len(hue_order) 
    sns.scatterplot(
        data=dfembed,
        x='embed_1', y='embed_2',
        hue=hue,
        hue_order=hue_order,
        edgecolor='none',
        s=1,
        ax=ax,
    )
    ax.legend(bbox_to_anchor=(0,0), loc='upper left', ncol=1+int(nums/10))
    ax.set_aspect('equal')
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    