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

def plot_enc(pmat, fpmat):
    """
    """
    pm = sns.color_palette('husl', n_colors=pmat.shape[1])
    
    with sns.axes_style('ticks'):
        fig = plt.figure(figsize=(15,6))
        ax_dict = fig.subplot_mosaic("AACBBB")
        ax = ax_dict['A']
        # fig, ax = plt.subplots(figsize=(6,8))
        _mat, _row, _col = basicu.diag_matrix_rows(fpmat.values)
        nmat = len(_mat)
        sns.heatmap(pd.DataFrame(_mat, columns=_col+1), 
                    xticklabels=5, 
                    cmap='rocket_r',
                    ax=ax, 
                    vmax=0.01, 
                    cbar_kws=dict(shrink=0.3, label='Weight prop.', ticks=[0,0.01], aspect=10,),
                   )
        ax.set_xlabel('Basis')
        ax.set_ylabel('Genes')
        ax.set_yticks([nmat])
        ax.set_title(f"Total #: {int(np.sum(pmat.values))}")
        ax.text(0, nmat, nmat, ha='right')
        
        ax = ax_dict['B']
        for i, col in enumerate(pmat):
            _x = pmat[col]
            _x = np.flip(np.sort(_x[_x>0])) #[:,:,-1]
            ax.plot(_x, color=pm[i], label=f'{i+1}')

        ax.set_yscale('log')
        ax.set_xlabel('Genes')
        ax.set_ylabel('Weight')
        sns.despine(ax=ax)
        ax.set_title('Encoding matrix')
        ax.legend(ncol=3, title='Basis')
        
        ax_dict['C'].axis('off')
        plt.show()
    return 

def plot_intn(prjx):
    """
    # intensity across bits
    """
    fig = plt.figure(figsize=(12,4))
    ax_dict = fig.subplot_mosaic("AAABB")
    ax = ax_dict['A']
    sns.boxplot(data=prjx, ax=ax, color='gray', fliersize=2, width=0.7)
    ax.set_title('Intensities')
    ax.set_yscale('log')
    sns.despine(ax=ax)

    ax = ax_dict['B']
    prjx_medbits = np.median(prjx, axis=0)
    prjx_q10bits = np.percentile(prjx, 10, axis=0)
    prjx_q90bits = np.percentile(prjx, 90, axis=0)
    order = np.argsort(prjx_medbits)[::-1]
    
    ax.plot(prjx_medbits[order], '-o', markersize=5, label='median')
    ax.plot(prjx_q10bits[order], '-o', markersize=5, label='10 perctl.')
    ax.plot(prjx_q90bits[order], '-o', markersize=5, label='90 perctl.')
    
    ax.set_xticks(np.arange(len(prjx_medbits)))
    ax.set_title('Median Intensity across bits\n(re-ordered)')
    ax.set_yscale('log')
    sns.despine(ax=ax)
    ax.legend(bbox_to_anchor=(1,1))

    fig.subplots_adjust(wspace=0.4)
    plt.show()
    
    return

def plot_embx_clsts(prjx_clsts, embx_clsts, embx_clsts_z, _rows, _cols):
    """
    Projection matrix (Z)
    Embedded matrix (Z_norm)
    zscored
    
    rows, cols
    
    """
    fig, axs = plt.subplots(1,3,figsize=(3*4,1*8))
    cbar_kws = dict(shrink=0.7, orientation='horizontal', fraction=0.02, pad=0.15)
    ax = axs[0]
    sns.heatmap(np.log10(prjx_clsts)[_rows][:,_cols], 
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title('log10(Z)')
    ax.set_xlabel('Basis')
    ax.set_ylabel('Cell types')

    ax = axs[1]
    sns.heatmap(embx_clsts[_rows][:,_cols], 
                yticklabels=False,
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title('Z_norm')

    ax = axs[2]
    sns.heatmap(embx_clsts_z[_rows][:,_cols], 
                yticklabels=False,
                cmap='coolwarm', 
                cbar_kws=cbar_kws,
                ax=ax,
               )
    ax.set_title('Z_norm - zscored per bit')
    fig.subplots_adjust(wspace=0.05)
    plt.show()

def plot_heatcorr(embx_clsts_corr, vmin=0.5): 
    """clst-clst corr
    """
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.4])
    sns.heatmap(embx_clsts_corr, 
                xticklabels=False, 
                yticklabels=False, 
                cbar_ax=cbar_ax,
                cbar_kws=dict(label='Pearson corr.', 
                              # ticks=[0.5, 0.75, 1],
                              aspect=5,
                             ),
                cmap='rocket_r',
                ax=ax, 
                vmin=vmin,
               )
    ax.set_xlabel('Known cell types')
    ax.set_ylabel('Known cell types')
    ax.set_aspect('equal')
    ax.set_title('')

    # powerplots.savefig_autodate(fig, os.path.join(fig_dir, "NN_DPNMF_correlation_matrix.pdf"))
    plt.show()
