import nupack
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from dredFISH.Utils import sequ

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

def run_n_readouts(seqs_rdt, seqs_enc, seqs_tag, conc_r, conc_e, 
                   ts=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                   adaptive=False):
    """Run n readouts one at a time.
    Each tube has all the encoding probes (e{j} with all js) and one readout probe r{i}

    Args:
        seqs_rdt (numpy.ndarray): a string array
        seqs_enc (numpy.ndarray): a string array -- complementary to the above
        seqs_tag (numpy.ndarray): names for each seq pair
        conc_r (float): concentration, in M (molar)
        conc_e (float): concentration, in M (molar)
        ts (list, optional): list of temperatures (Celsius) to test. Defaults to [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75].
        adaptive (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: (res, emap) -- results pandas DataFrame; numpy array for Ri-Ej
    """
    num_tubes = len(seqs_rdt)
    
    # specify strands
    strands_e = [nupack.Strand(seq_enc, name=f"e{i}") 
                 for i, seq_enc in enumerate(seqs_enc)]
    
    tubes = []
    for tube_idx in np.arange(num_tubes):
        readout_i = tube_idx
        tube_name = f'tube{tube_idx}'
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
                         name=tube_name)
        tubes.append(tube)
    
    # analyze with different model temperatures
    res = [] 
    emaps = {}
    for t in ts:
        print('>', end='')
        emaps[t] = []
        model = nupack.Model(material='dna', 
                              celsius=t,
                              sodium=0.3,
                             )
        tube_results = nupack.tube_analysis(tubes=tubes, model=model)
        
        for tube_idx in np.arange(num_tubes):
            print('.', end='')
            readout_i = tube_idx
            tube_name = f'tube{tube_idx}'
            conc = tabulate_results(tube_results, name=tube_name)
            precision, usage, recall = summarize(conc, readout_i)
            emap = summarize_heatmap_fast(conc, readout_i)
            emaps[t].append(emap)
            res.append({'t': t,
                        'index': tube_idx,
                        'tube': tube_name,
                        'hybe': seqs_tag[readout_i],
                        'precision': precision,
                        'usage': usage,
                        'recall': recall,
                       })
        print("")

    res = pd.DataFrame(res)
    return res, emaps