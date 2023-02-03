"""Convienient functions to evaluate the neural network probe design 
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dredFISH.Utils import basicu

def get_mse_torch(tnsr_true, tnsr_pred=[]):
    """Mean (over rows) squared error
    if no pred, returns the variance.
    """
    if len(tnsr_pred):
        mse = (tnsr_true - tnsr_pred).square().mean(axis=0)
    else:
        mse = (tnsr_true - tnsr_true.mean(axis=0)).square().mean(axis=0)
    return mse
        
def get_r2_torch(tnsr_true, tnsr_pred, zero_tol=1e-10):
    """
    """
    mse = get_mse_torch(tnsr_true, tnsr_pred)
    var = get_mse_torch(tnsr_true)
    r2 = 1-mse/torch.clamp(var, min=zero_tol)
    return r2

def plot_enc(pmat, title=""):
    """
    """
    pm = sns.color_palette('husl', n_colors=pmat.shape[1])
    
    with sns.axes_style('ticks'):
        fig = plt.figure(figsize=(15,6))
        ax_dict = fig.subplot_mosaic("AACBBB")
        ax = ax_dict['A']
        # fig, ax = plt.subplots(figsize=(6,8))
        fpmat = pmat.divide(pmat.sum(axis=0), axis=1)
        _mat, _row, _col = basicu.diag_matrix_rows(fpmat.values)
        nmat = len(_mat)
        sns.heatmap(pd.DataFrame(_mat, columns=_col), 
                    xticklabels=5, 
                    cmap='rocket_r',
                    ax=ax, 
                    vmax=0.01, 
                    cbar_kws=dict(shrink=0.3, label='Weight prop.', ticks=[0,0.01], aspect=10,),
                   )
        ax.set_xlabel('Basis')
        ax.set_ylabel('Genes')
        ax.set_yticks([nmat])
        ax.set_title(f"Total #: {int(np.sum(pmat.values)):,}")
        ax.text(0, nmat, nmat, ha='right')
        
        ax = ax_dict['B']
        for i, col in enumerate(pmat):
            _x = pmat[col]
            _x = np.flip(np.sort(_x[_x>0])) #[:,:,-1]
            ax.plot(_x, color=pm[i], label=f'{i}')

        ax.set_yscale('log')
        ax.set_xlabel('Genes')
        ax.set_ylabel('Weight')
        sns.despine(ax=ax)
        ax.set_title('Encoding matrix')
        ax.legend(ncol=3, title='Basis')
        
        ax_dict['C'].axis('off')
        fig.suptitle(title)
        plt.show()
    return 

def plot_intn(prjx):
    """
    # intensity across bits
    """
    fig = plt.figure(figsize=(15,4))
    ax_dict = fig.subplot_mosaic("AABB")
    ax = ax_dict['A']
    sns.boxplot(data=prjx, ax=ax, color='gray', fliersize=2, width=0.7)
    ax.set_title('Intensities')
    ax.set_yscale('log')
    sns.despine(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax = ax_dict['B']
    prjx_medbits = np.median(prjx, axis=0)
    prjx_q10bits = np.percentile(prjx, 10, axis=0)
    prjx_q90bits = np.percentile(prjx, 90, axis=0)
    order = np.argsort(prjx_medbits)[::-1]
    
    n = len(prjx_medbits)
    ax.plot(np.arange(n), prjx_medbits[order], '-o', markersize=5, label='median')
    ax.plot(np.arange(n), prjx_q10bits[order], '-o', markersize=5, label='10 perctl.')
    ax.plot(np.arange(n), prjx_q90bits[order], '-o', markersize=5, label='90 perctl.')
    ax.set_xticks(np.arange(n)) 
    ax.set_xticklabels(order, rotation=90)

    ax.set_title('Intensities re-ordered')
    ax.set_yscale('log')
    sns.despine(ax=ax)
    ax.legend(bbox_to_anchor=(1,1))

    fig.subplots_adjust(wspace=0.4)
    plt.show()
    
    return

def plot_embx_clsts(
    prjx_clsts, embx_clsts, embx_clsts_z, 
    title1='log10(mean (Z))',
    title2='mean (Z_norm)',
    title3='bitwise zscored [mean (Z_norm)]',
    rownames=None, colnames=None, 
    _rows=None, _cols=None, 
    title='',
    figsize=(3*4,1*8)):
    """
    Projection matrix (Z)
    Embedded matrix (Z_norm)
    zscored
    
    rows, cols
    
    """
    m, n  = prjx_clsts.shape
    if rownames is None:
        rownames = np.arange(m)
    if colnames is None:
        colnames = np.arange(n)
    if _rows is None:
        _rows = np.arange(m)
    if _cols is None:
        _cols = np.arange(n)

    cbar_kws = dict(shrink=0.7, orientation='horizontal', fraction=0.02, pad=0.15)
    fig, axs = plt.subplots(1,3,figsize=figsize)
    ax = axs[0]
    sns.heatmap(np.log10(prjx_clsts+1)[_rows][:,_cols], 
                yticklabels=rownames[_rows],
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title(title1)
    ax.set_xlabel('Basis')
    ax.set_ylabel('Cell types')

    ax = axs[1]
    sns.heatmap(embx_clsts[_rows][:,_cols], 
                yticklabels=False,
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title(title2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

    ax = axs[2]
    zmin = np.min(embx_clsts_z)
    zmax = np.max(embx_clsts_z)
    vmin = np.clip(zmin, -3, None) 
    vmax = np.clip(zmax, None, 3) 
    sns.heatmap(embx_clsts_z[_rows][:,_cols], 
                yticklabels=False,
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
               )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title(title3)
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.05)
    plt.show()

def plot_heatcorr(embx_clsts_corr, vmin=0.5, vmax=1, title='', ax=None, cbar_ax=None, label=True, 
    cmap='rocket_r', metric_label='Pearson corr.',
    ): 
    """clst-clst corr
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,4))
        cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.4])

    sns.heatmap(embx_clsts_corr, 
                xticklabels=False, 
                yticklabels=False, 
                cbar_ax=cbar_ax,
                cbar_kws=dict(label=metric_label, 
                              # ticks=[0.5, 0.75, 1],
                              aspect=5,
                             ),
                cmap=cmap,
                ax=ax, 
                vmin=vmin,
                vmax=vmax,
               )
    if label:
        ax.set_xlabel(f'Known cell types\n(n={len(embx_clsts_corr)})')
        ax.set_ylabel(f'Known cell types\n(n={len(embx_clsts_corr)})')
    ax.set_aspect('equal')
    ax.set_title(title)

    # powerplots.savefig_autodate(fig, os.path.join(fig_dir, "NN_DPNMF_correlation_matrix.pdf"))

def plot_dcdx(
    prjx_clsts,
    title1='decoder',
    rownames=None, colnames=None, 
    _rows=None, _cols=None, 
    figsize=(1*4,1*8)):
    """
    Projection matrix (Z)
    Embedded matrix (Z_norm)
    zscored
    
    rows, cols
    
    """
    m, n  = prjx_clsts.shape
    if rownames is None:
        rownames = np.arange(m)
    if colnames is None:
        colnames = np.arange(n)
    if _rows is None:
        _rows = np.arange(m)
    if _cols is None:
        _cols = np.arange(n)

    cbar_kws = dict(shrink=0.7, orientation='horizontal', fraction=0.02, pad=0.15)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.heatmap(prjx_clsts[_rows][:,_cols], 
                yticklabels=rownames[_rows],
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title(title1)
    ax.set_xlabel('Basis')
    ax.set_ylabel('Cell types')
    fig.subplots_adjust(wspace=0.05)
    plt.show()

def plot_embx_clsts_v2(
    prjx_clsts, embx_clsts, embx_clsts_z, dcdx, 
    title1='log10(mean (Z))',
    title2='mean (Z_norm)',
    title3='bitwise zscored [mean (Z_norm)]',
    title4='decoder',
    rownames=None, colnames=None, 
    _rows=None, _cols=None, 
    title='',
    figsize=(4*4,1*8)):
    """
    Projection matrix (Z)
    Embedded matrix (Z_norm)
    zscored
    
    rows, cols
    
    """
    m, n  = prjx_clsts.shape
    if rownames is None:
        rownames = np.arange(m)
    if colnames is None:
        colnames = np.arange(n)
    if _rows is None:
        _rows = np.arange(m)
    if _cols is None:
        _cols = np.arange(n)

    cbar_kws = dict(shrink=0.7, orientation='horizontal', fraction=0.02, pad=0.15)
    fig, axs = plt.subplots(1,4,figsize=figsize)
    ax = axs[0]
    sns.heatmap(np.log10(prjx_clsts+1)[_rows][:,_cols], 
                yticklabels=rownames[_rows],
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title(title1)
    ax.set_xlabel('Basis')
    ax.set_ylabel('Cell types')

    ax = axs[1]
    sns.heatmap(embx_clsts[_rows][:,_cols], 
                yticklabels=False,
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title(title2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

    ax = axs[2]
    zmin = np.min(embx_clsts_z)
    zmax = np.max(embx_clsts_z)
    vmin = np.clip(zmin, -3, None) 
    vmax = np.clip(zmax, None, 3) 
    sns.heatmap(embx_clsts_z[_rows][:,_cols], 
                yticklabels=False,
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
               )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_title(title3)

    ax = axs[3]
    sns.heatmap(dcdx[_rows][:,_cols], 
                yticklabels=False,
                xticklabels=colnames[_cols],
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title(title4)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.05)
    plt.show()