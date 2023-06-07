"""
HCRdesign is a library of function used to design HCR probes sequences
"""

import more_itertools as mit
import itertools as it
import numpy as np
import random
import Levenshtein 
import re 
import matplotlib.pyplot as plt
import seaborn as sns
import nupack
import time
from IPython.display import HTML
from dredFISH.Utils.nupacku import *
from dredFISH.Utils.sequ import *
from dredFISH.Design.ShortSeqDesignTools import *

def BuildtailedHCR(r1,r2,s1,s2):
    a=s1[0:12]
    c=s1[12:]
    b=s2
    H1=r1+a+b+get_rcseq(c)+get_rcseq(b)
    H2=get_rcseq(b)+get_rcseq(a)+b+c+r2
    I=get_rcseq(b)+get_rcseq(a)

    return(I,H1,H2)

def TestHCRfidelity(I,H1,H2):
    H1strand  = nupack.Strand(H1,name='H1')
    H2strand = nupack.Strand(H2,name='H2')
    Istrand = nupack.Strand(I,name='I')
    strands_tube = {H1strand: 1e-9, H2strand  : 1e-9, Istrand : 1e-10}
    tube = nupack.Tube(strands=strands_tube,  complexes=nupack.SetSpec(max_size=5),
                            name='hcr')
    tube_results = nupack.tube_analysis(tubes=[tube], model=my_model)
    conc = tabulate_results(tube_results,name='hcr')
    ptrn = '^(?=.*I)(?=.*H1.*H1)(?=.*H2.*H2).*\+.*\+.*'
    polymer_frac = conc.filter(regex=ptrn).sum()/conc.filter(regex='I').sum()
    return(polymer_frac)

my_model = nupack.Model(material='dna', 
                        celsius=37,
                        sodium=0.3, 
                        ensemble='nostacking'
                    )

H1_org = "GAAGCGAATATGGTGAGAGTTGGAGGTAGGTTGAGGCACATTTACAGACCTCAACCTACCTCCAACTCTCAC"
H2_org = "CCTCAACCTACCTCCAACTCTCACCATATTCGCTTCGTGAGAGTTGGAGGTAGGTTGAGGTCTGTAAATGTG"
I_org = "CCTCAACCTACCTCCAACTCTCACCATATTCGCTTC"

tail1 = 'ATTGTCAGGACCTTAGGCCA'
tail2 = 'TGGTCTCGTCTCTGTTTCTG'

H1_org_tail = tail1 + H1_org 
H2_org_tail = H2_org + tail2

strnds = [H2_org_tail,H2_org_tail,I_org,H1_org_tail,H1_org_tail]
mfe_structures = nupack.mfe(strands=strnds, model=my_model)
opt_structure_matrix = mfe_structures[0].structure.matrix()

def score_hcr(seqs,tail_length = 20):
    if len(seqs)==4:
        r1,r2,s1,s2 = seqs
        I1,H1,H2 = BuildtailedHCR(r1,r2,s1,s2)
        I2 = H1[-36:]
    elif len(seqs)==3: 
        I1,H1,H2 = seqs
        I2 = H1[-36:]
    else: 
        raise ValueError("seqs must have 4 (r1,r2,s1,s2) or 3 (I,H1,H2) elements")

    # there are two types of "mismatches" for HCR
    # 1. The interactions between the molecuels is not as expected. 
    # 2. There are interactins within the molecules when they are in strcture that shouldn't be there

    # Score interactions between strands: 
    strnd_bindings_I1,I1_matrix = count_strand_interactions([H2,I1,H1],return_matrix=True)
    strnd_bindings_I1[0,0] -= tail_length
    strnd_bindings_I1[2,2] -= tail_length
    strnd_bindings_I2,I2_matrix = count_strand_interactions([I2,H1,H2],return_matrix=True)
    strnd_bindings_I2[1,1] -= tail_length
    strnd_bindings_I2[2,2] -= tail_length
    I1_opt = [[36,0,36],
              [0,0,36],
              [36,36,0]]

    I2_opt = [[0,0,36],
              [0,36,36],
              [36,36,0]]
    mismatch = np.abs(strnd_bindings_I1-I1_opt) + \
               np.abs(strnd_bindings_I2-I2_opt)
    mismatch = mismatch.sum()

    # score off diagonal within strands (there shouldn't be any...)
    non_interacting_nt = np.trace(I1_opt) + 2*tail_length
    mismatch += (non_interacting_nt - np.trace(I1_matrix))
    mismatch += (non_interacting_nt - np.trace(I2_matrix))
    
    return(mismatch)

strnds = [H2_org_tail,I_org,H1_org_tail]
mfe_structures = nupack.mfe(strands=strnds, model=my_model)
opt_3way_matrix = mfe_structures[0].structure.matrix()
opt_two_hcrs = np.zeros((opt_3way_matrix.shape[0]*2,opt_3way_matrix.shape[0]*2))
opt_two_hcrs[:opt_3way_matrix.shape[0],:opt_3way_matrix.shape[0]]=opt_3way_matrix
opt_two_hcrs[opt_3way_matrix.shape[0]:,opt_3way_matrix.shape[0]:]=opt_3way_matrix
opt_two_hcrs_triu = np.triu(opt_two_hcrs,k=-1)

def count_strand_interactions(strnds,nupack_model = my_model,return_matrix = False):
    lengths = [len(s) for s in strnds]
    mfe_structures = nupack.mfe(strands=strnds, model=nupack_model)
    structure_mat = mfe_structures[0].structure.matrix()
    result = np.zeros((len(lengths), len(lengths)), dtype=int)
    row_start = 0
    for i, row_length in enumerate(lengths):
        col_start = 0
        for j, col_length in enumerate(lengths):
            block = structure_mat[row_start:row_start + row_length, col_start:col_start + col_length]
            result[i, j] = np.count_nonzero(block)
            col_start += col_length
        row_start += row_length

    if return_matrix: 
        return result,structure_mat
    return result

def score_two_hcrs(HCR1,HCR2,plot_flag=False):
    HCR1=HCR1 + (HCR1[1][-36:],)
    HCR2=HCR2 + (HCR2[1][-36:],)
    cross_interactions = 0
    # there are four possible cross interactions: 
    poss_strand_int = [[HCR1[3],HCR1[1],HCR1[2],HCR2[3],HCR2[1],HCR2[2]], 
                    [HCR1[2],HCR1[0],HCR1[1],HCR2[2],HCR2[0],HCR2[1]], 
                    [HCR1[3],HCR1[1],HCR1[2],HCR2[2],HCR2[0],HCR2[1]], 
                    [HCR1[2],HCR1[0],HCR1[1],HCR2[3],HCR2[1],HCR2[2]]]

    for inter_strnds in poss_strand_int:
        strnd_cross_bindings = count_strand_interactions(inter_strnds)
        cross_interactions += np.sum(strnd_cross_bindings[0:3,3:])
        if plot_flag: 
            plot_strand_mfe_structure(inter_strnds)
            plt.title(f"cross bind={np.sum(strnd_cross_bindings[0:3,3:])}")
    return cross_interactions
    
def find_good_hcrs(s24mers,s20mers,max_mfe = 1000, verbose = False):

    class CustomSet(set):
        def __contains__(self, item):
            if not isinstance(item, tuple) or len(item) != 4:
                raise ValueError("Item must be a tuple of 4 elements")
            sorted_item = tuple(sorted(item))
            return super().__contains__(sorted_item)
        def add(self, item):
            if not isinstance(item, tuple) or len(item) != 4:
                raise ValueError("Item must be a tuple of 4 elements")
            sorted_item = tuple(sorted(item))
            ln = len(self)
            super().add(sorted_item)
            if len(self) == ln:
                print("item already exists, didn't add anything")
        
    hits = []
    unused_24mers = set(s24mers)
    unused_20mers = set(s20mers)
    combos_tested = CustomSet()
    mfe_cnt = 0
    while len(combos_tested) < max_mfe and len(unused_24mers) > 1 and len(unused_20mers) > 1:
        s24 = random.sample(list(unused_24mers),k=2)
        s20 = random.sample(list(unused_20mers),k=2)
        seq_to_test = (s20[0],s20[1],s24[0],s24[1])
        if seq_to_test in combos_tested: 
            continue
        combos_tested.add(seq_to_test)
        scr=score_hcr(seq_to_test)
        mfe_cnt += 1
        if scr==0:
            unused_24mers.difference_update(s24)
            unused_20mers.difference_update(s20)
            I,H1,H2 = BuildtailedHCR(s20[0],s20[1],s24[0],s24[1])
            hits.append((I,H1,H2))
        if verbose and mfe_cnt % 100 ==0 : 
            print(f"Tested: {len(combos_tested)} goods: {len(hits)} s24 left: {len(unused_24mers)} s20s left: {len(unused_20mers)}")
    if verbose : 
            print(f"Final - Tested: {len(combos_tested)} goods: {len(hits)} s24 left: {len(unused_24mers)} s20s left: {len(unused_20mers)}")
    return hits
    
def find_hcr_sets(s24mers,s20mers,max_mismatch = 20,iter = 1000,save_file = None):
    start = time.time()
    ij_24mers = list(it.combinations(range(len(s24mers)),2))
    ij_20mers = list(it.combinations(range(len(s20mers)),2))
    hits = [(I_org,H1_org_tail,H2_org_tail)]
    used_24mers = set()
    used_20mers = set()

    for i in range(iter):
        s24 = random.choice(ij_24mers)
        s20 = random.choice(ij_20mers)
        if s24[0] in used_24mers or s24[1] in used_24mers: 
            continue
        if s20[0] in used_20mers or s20[1] in used_20mers:
            continue 
        scr=score_hcr((s20mers[s20[0]],s20mers[s20[1]],s24mers[s24[0]],s24mers[s24[1]]))
        if scr==0:
            I,H1,H2 = BuildtailedHCR(s20mers[s20[0]],s20mers[s20[1]],s24mers[s24[0]],s24mers[s24[1]])
            scr_hcr_pair = np.ones(len(hits))
            for j in range(len(hits)):
                scr_hcr_pair[j] = score_two_hcrs((I,H1,H2),hits[j])
            if all(scr_hcr_pair<max_mismatch):
                used_24mers.add(s24[0])
                used_24mers.add(s24[1])
                used_20mers.add(s20[0])
                used_20mers.add(s20[1])
                I,H1,H2 = BuildtailedHCR(s20mers[s20[0]],s20mers[s20[1]],s24mers[s24[0]],s24mers[s24[1]])
                hits.append((I,H1,H2,f"{time.time()-start:.2f}"))
        if save_file is not None: 
            HCR_set = np.array(hits,dtype=str)
            np.savetxt(save_file, HCR_set, delimiter=',',fmt="%s")
    return(hits)
