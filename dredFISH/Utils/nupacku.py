import nupack
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from scipy.special import comb


from dredFISH.Utils import sequ
from dredFISH.Utils.__init__plots import *


def get_num_combinations(n):
    """get number of expected complex

    Args:
        n (int): number of ingredients

    Returns:
        float: number of resulting complex (single, self-bind, cross-bind)
    """
    return n+n+comb(n,2)

def get_rcseq(seq):
    """reverse complement

    Args:
        seq (str): _description_

    Returns:
        str: _description_
    """
    return str(Seq(seq).reverse_complement())

def tabulate_results(tube_results, name='t1'):
    """Turn nupack output into a pandas Series

    Args:
        tube_results (nupack.named.Result): nupack result object
        name (str, optional): tube name (input to nupack). Defaults to 't1'.

    Returns:
        pandas.series: concentration for each complex
    """
    conc = pd.Series({key.name.strip("()"): item for key, item in 
            tube_results[name].complex_concentrations.items()
           })
    return conc

def assign_group(index):
    """_summary_

    Args:
        index (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "+" not in index:
        label = "single"
    else:
        a, b = index.split('+')
        if a == b:
            label = "self-bind"
        else:
            label = "cross-bind"
            
    return label

def organize_raw_conc_table(conc, baseconc):
    """_summary_

    Args:
        conc (_type_): _description_

    Returns:
        _type_: _description_
    """
    resfancy = conc.sort_values(ascending=False).to_frame("conc")
    resfancy['order'] = np.arange(len(resfancy))
    resfancy["log10frac"] = np.log10(resfancy['conc']/baseconc)
    resfancy["group"] = [assign_group(idx) for idx in resfancy.index.values]

    return resfancy
    
def summarize(conc, readout_i):
    """summarize the nupack result Series
    it assumes a pair-wise simulation with the following initial elements:
        - e0, e1, e2, ..., ej, ....
        - r{i}

    Args:
        conc (pandas.Series): concentrations for each complex; assumed to have r{i}+e{j} etc...
        readout_i (int): which type of labels to pull statistics from

    Returns:
        tuple: (precision, usage, recall)
    """
    lbl_signal = f'r{readout_i}+e{readout_i}'
    lbl_signal2 = f'e{readout_i}+r{readout_i}'
    
    lbl_floating = [f'r{readout_i}',
                    f'r{readout_i}+r{readout_i}',
                   ]
    
    ### this was flawed
    total   = pd.concat([
                conc.filter(regex=f'^r{readout_i}\+'),
                conc.filter(regex=f'\+r{readout_i}$'),
                conc.filter(regex=f'^r{readout_i}$'),
                ]).sum()  # all terms with r
    
    total_e = pd.concat([
                conc.filter(regex=f'^e{readout_i}\+'),
                conc.filter(regex=f'\+e{readout_i}$'),
                conc.filter(regex=f'^e{readout_i}$'),
                ]).sum()  # all terms with e
    ### this was flawed 
    
    if lbl_signal in conc.index.values:
        signal = conc.loc[lbl_signal]
    elif lbl_signal2 in conc.index.values:
        signal = conc.loc[lbl_signal2]
        
    floating = conc.loc[lbl_floating].sum()
    
    usage = signal/total # fraction of provided r that goes to signal
    precision = signal/(total-floating) # fraction of correct binding
    recall = signal/total_e
    
    return precision, usage, recall

def summarize_heatmap_fast(conc, readout_i):
    """get r{i}+e{j} where j goes from 0 to n

    Args:
        conc (pandas.Series): _description_
        readout_i (int): _description_

    Returns:
        numpy.ndarray: r{i} vs e{j} with all j
    """
    case1 = conc.filter(regex=f'^r{readout_i}\+e')  # r0+e...
    newidx1 = [int(lbl.split('+')[1][1:]) for lbl in case1.index]
    
    case2 = conc.filter(regex=f'^e[0-9]+\+r{readout_i}') # e..+r0
    newidx2 = [int(lbl.split('+')[0][1:]) for lbl in case2.index]
    
    vec = np.zeros(len(case1)+len(case2))
    vec[newidx1] = case1.values
    vec[newidx2] = case2.values
    return vec

def simulate_nupack(seqs_enc, readout_i, conc_r, conc_e, t=37, sodium=0.3, tube_name='t1'):
    """Run 1 simulation
    with many encodings; 1 readout (complement to 1 of the encodings)
    The tube has all the encoding probes (e{j} with all js) and one readout probe r{i}

    Args:
        seqs_enc (numpy.ndarray): a string array -- complementary to the above
        readout_i (int): index in seqs_enc whose reverse complement will be added
        conc_r (float): concentration, in M (molar)
        conc_e (float): concentration, in M (molar)
        sodium (float, optional): concentration, in M. Defaults to 0.3.
        t (int, optional): temperature in Celsius. Defaults to 37.

    Returns:
        _type_: _description_
    """
    # specify strands
    seq_rdt = get_rcseq(seqs_enc[readout_i])

    strands_e = [nupack.Strand(seq_enc, name=f"e{i}") 
                 for i, seq_enc in enumerate(seqs_enc)]
    strand_r = nupack.Strand(seq_rdt, name=f"r{readout_i}")

    strands_tube = {strand: conc_e for strand in 
                    strands_e} # include all
    strands_tube[strand_r] = conc_r

    tube = nupack.Tube(strands=strands_tube,  
                        complexes=nupack.SetSpec(max_size=2), 
                        name=tube_name)
    
    # analyze with different model temperatures
    model = nupack.Model(material='dna', 
                            celsius=t,
                            sodium=sodium,
                            )
    tube_results = nupack.tube_analysis(tubes=[tube], model=model)
    # res = tabulate_results_fancy(tube_results, name=tube_name)

    return tube_results 

def simulate_cross_binding(
                    seqs_enc, 
                    seqs_tag=None, 
                    seqs_token=None, # which entries in seqs_enc to introduce its reverse complement sequence as "readout"
                    conc_e=1e-11,
                    conc_r=1e-9,
                    temps=[37], # [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                    sodium=0.3,
                    material='dna',
                    adaptive=False):
    """Given a list of sequences, we simulate the scenario where
    all those sequences are present in a tube (encoding probes), while introducing the reverse complement (readout)
    one at a time in separate tubes.

    Each tube has all the encoding probes (e{j} for all js) and one readout probe r{i}.

    Args:
        seqs_enc (numpy.ndarray): a list of string
        seqs_tag (numpy.ndarray, optional): names for each seq, defaults to None.
        conc_e (float, optional): concentration, in M (molar), defaults to 1e-11 (0.01 nM)
        conc_r (float, optional): concentration, in M (molar), defaults to 1e-9  (1 nM)
        temps (list, optional): list of temperatures (Celsius) to test. Defaults to [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75].
        sodium (float, optional): sodium concentration, in M (molar).
        adaptive (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: (res, emap, raw_concs, tms) results pandas DataFrame; numpy array for Ri-Ej; dictionary of raw concentrations.
    """

    seqs_enc = np.array(seqs_enc)
    # get RC seqs
    seqs_rdt = np.array([get_rcseq(seq_enc) for seq_enc in seqs_enc])
    # get TMs
    tms = [sequ.get_tm(x, Na=sodium*1e3, dnac1=conc_r*1e9, dnac2=conc_e*1e9, fmd=0) for x in seqs_enc]
    # name for each pair
    if seqs_tag is None: 
        seqs_tag = np.arange(len(seqs_enc))
    assert len(seqs_tag) == len(seqs_enc)

    # tokens to run tubes on
    if seqs_token is None:
        seqs_token = np.arange(len(seqs_enc)) # everything

    # all encoding probes 
    strands_e = [nupack.Strand(seq_enc, name=f"e{i}") 
                 for i, seq_enc in enumerate(seqs_enc)]

    # one readout token per tube 
    tubes = []
    for readout_i in seqs_token:
        strand_r = nupack.Strand(seqs_rdt[readout_i], name=f"r{readout_i}")
        if adaptive:
            strands_tube = {strand: conc_e for strand in 
                            strands_e[readout_i:]} # exclude previous
        else:
            strands_tube = {strand: conc_e for strand in 
                            strands_e} # include all
        strands_tube[strand_r] = conc_r
        tube = nupack.Tube(strands=strands_tube,  
                         complexes=nupack.SetSpec(max_size=2), 
                         name=f'tube{readout_i}')
        tubes.append(tube)
    
    # analyze with different model temperatures
    res = [] 
    emaps = {}
    raw_concs = {}
    for t in temps:
        print('>', end='')
        raw_concs[t] = {}
        emaps[t] = []
        model = nupack.Model(material=material, 
                             celsius=t,
                             sodium=sodium,
                            )
        tube_results = nupack.tube_analysis(tubes=tubes, model=model)
        
        for readout_i in seqs_token:
            print('.', end='')
            conc = tabulate_results(tube_results, name=f'tube{readout_i}')
            emap = summarize_heatmap_fast(conc, readout_i)
            precision, usage, recall = summarize(conc, readout_i)
            conc_fancy = organize_raw_conc_table(conc, conc_e)

            raw_concs[t][readout_i] = conc_fancy
            emaps[t].append(emap)
            res.append({
                        'readout_idx': readout_i,
                        'readout_tag': seqs_tag[readout_i],
                        'tm': tms[readout_i],
                        't': t,
                        'precision': precision,
                        'usage': usage,
                        'recall': recall,
                       })
        print("")

    res = pd.DataFrame(res)
    return res, emaps, raw_concs, tms


def view_raw(res, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    sns.scatterplot(data=res, x='order', y='log10frac', hue='group', s=10)
    ax.legend(bbox_to_anchor=(1,1))
    sns.despine(ax=ax)

def view_emap(emap, baseconc, ax=None, title=None, vmax=0, vmin=-3):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
    mat = np.log10(np.vstack(emap)/baseconc)
    sns.heatmap(mat, 
                xticklabels=5,
                yticklabels=5,
                vmax=vmax, 
                vmin=vmin,
                cbar_kws=dict(shrink=0.5, label=f'log10(conc/baseline)', ticks=[-3, -2,-1,0]), 
                cmap='coolwarm',
                ax=ax,
               )
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)

def view_pr(resplot, ax_row, **kwargs):
    """
    """
    p, r = resplot['precision'], resplot['recall']
    minpr = np.minimum(p, r)
    f1 = 2/(1/p+1/r)
    
    ax = ax_row[0]
    ax.plot(resplot['t'], p, '-o', **kwargs)
    ax.set_xlabel('Celsius')
    ax.set_ylabel('Precision')
    sns.despine(ax=ax)
    
    ax = ax_row[1]
    ax.plot(resplot['t'], r, '-o', **kwargs)
    ax.set_xlabel('Celsius')
    ax.set_ylabel('Recall')
    sns.despine(ax=ax)
    
    ax = ax_row[2]
    ax.plot(resplot['t'], minpr, '-o', **kwargs)
    ax.set_xlabel('Celsius')
    ax.set_ylabel('Min (Prec., Recall)')
    ax.set_ylim([-.05,1.1])
    sns.despine(ax=ax)
