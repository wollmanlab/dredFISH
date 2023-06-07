"""
EncoderProbePrimerDesignTools is a library of function used to find primers to add to encoder probe library
"""

import numpy as np
import re 
import matplotlib.pyplot as plt
import nupack

from dredFISH.Utils.nupacku import *
from dredFISH.Utils.sequ import *
from dredFISH.Design.ShortSeqDesignTools import *

def combine_readouts_to_encoder_based_on_weights(Encoding,Readouts,Weights,Spacer):
    OligoPool = []
    for i in range(Weights.shape[0]):
        w=np.array(Weights.iloc[i,1:])
        if not any(w):
            continue
        g=Weights.iloc[i,0]
        enc_seq = np.array(Encoding[Encoding['gname']==g]['seq'])
        for j in range(len(enc_seq)):
            ix=np.flatnonzero(w)
            r = get_rcseq(Readouts['seq'][ix[0]])
            e = enc_seq[j]
            oligo = ''.join([r, Spacer, e, Spacer, r])
            OligoPool.append(oligo) 
            w[ix[0]]-=1  

    return OligoPool

def  score_primer_pair_non_specific_binding(pp,OligoSet,qntl,nupack_model):
    fwd,rev = pp
    bps_fwd = np.zeros((len(OligoSet),2))
    dGs_fwd = np.zeros_like(bps_fwd)
    bps_rev = np.zeros((len(OligoSet),2))
    dGs_rev = np.zeros_like(bps_fwd)
    for i,oligo in enumerate(OligoSet): 
        (blah,bps_fwd[i,0],dGs_fwd[i,0]) =  all_pairwise_pairing([fwd,oligo+get_rcseq(rev)],my_model,avoid_self = True)
        (blah,bps_fwd[i,1],dGs_fwd[i,1]) =  all_pairwise_pairing([fwd,get_rcseq(oligo+get_rcseq(rev))],my_model,avoid_self = True)
        (blah,bps_rev[i,0],dGs_rev[i,0]) =  all_pairwise_pairing([fwd+oligo,rev],my_model,avoid_self = True)
        (blah,bps_rev[i,1],dGs_rev[i,1]) =  all_pairwise_pairing([get_rcseq(fwd+oligo),rev],my_model,avoid_self = True)
    avg_dG_fwd = np.quantile(dGs_fwd.min(axis=1),qntl)
    avg_dG_rev = np.quantile(dGs_rev.min(axis=1),qntl)
    avg_bps_fwd = np.quantile(bps_fwd.max(axis=1),1-qntl)
    avg_bps_rev = np.quantile(bps_rev.max(axis=1),1-qntl)
    return avg_bps_fwd,avg_bps_rev,avg_dG_fwd,avg_dG_rev

def score_primers_recall(pp,primer_overhangs,OligoSet,nupack_model,OligoConc = 10**-11,PrimerConc = 10**-8):
    fwd,rev = pp
    fwd_no_overhang = re.sub(primer_overhangs[0],'',fwd)
    rev_no_overhang = re.sub(primer_overhangs[1],'',rev)
    # Show initial PCR results
    recalls = np.zeros((len(OligoSet),2))
    for i,oligo in enumerate(OligoSet):
        oligo = fwd_no_overhang + oligo + get_rcseq(rev_no_overhang) 
        FwdStrand = nupack.Strand(fwd,name="Fwd")
        RevStrand = nupack.Strand(rev,name="Rev")
        OligoStrand = nupack.Strand(oligo,name=f"Oligo")
        RevCompOligoStrand = nupack.Strand(get_rcseq(oligo),name=f"rcOligo")
        strands_tube = {FwdStrand : PrimerConc, RevStrand : PrimerConc, 
                    OligoStrand : OligoConc, RevCompOligoStrand : OligoConc}
        tube1 = nupack.Tube(strands={FwdStrand : PrimerConc, RevStrand : PrimerConc, 
                                    RevCompOligoStrand : OligoConc},  
                                complexes=nupack.SetSpec(max_size=2), 
                                name="PCR_5to3")
        tube2 = nupack.Tube(strands={FwdStrand : PrimerConc, RevStrand : PrimerConc, 
                                    OligoStrand : OligoConc},  
                                complexes=nupack.SetSpec(max_size=2), 
                                name="PCR_3to5")
        tube_results = nupack.tube_analysis(tubes=[tube1,tube2], model=nupack_model) 
        # score recall for fwd primer
        conc = tabulate_results(tube_results,'PCR_5to3')
        conc = conc.drop(['Fwd','Rev','Fwd+Rev','Rev+Fwd','Fwd+Fwd','Rev+Rev'], errors='ignore')
        if 'rcOligo+Fwd' in conc.keys(): 
            recalls[i,0] = conc.loc['rcOligo+Fwd']/conc.sum()
        else: 
            recalls[i,0]  = conc.loc['Fwd+rcOligo']/conc.sum()
        # now for rev: 
        conc = tabulate_results(tube_results,'PCR_3to5')
        conc = conc.drop(['Fwd','Rev','Fwd+Rev','Rev+Fwd','Fwd+Fwd','Rev+Rev'], errors='ignore')
        if 'Oligo+Rev' in conc.keys(): 
            recalls[i,1] = conc.loc['Oligo+Rev']/conc.sum()
        else: 
            recalls[i,1] = conc.loc['Rev+Oligo']/conc.sum()

    return recalls.mean(axis=0)


def evolve_primer_pairs(primer_pair,masks,cand_attempts,mfe_attempts,my_nupack_model,exclude_patterns = None,Tm_min = 0):

    # find which index the last consecutive True starts at for both masks - will be used to calc Tm
    five_p_overhang = np.zeros(2,dtype=int)
    five_p_overhang[0] = np.where(masks[0])[0][0]
    five_p_overhang[1] = np.where(masks[1])[0][0]
    
    # calculate initial starting point:
    (oligo_pairs,bps,dGs) = all_pairwise_pairing(primer_pair,my_nupack_model)

    min_bps = bps.max()
    max_dG = dGs.min()
    
    test_strand = primer_pair.copy()
    best_strand = test_strand.copy()
    mfe_cnt = 0
    cand_cnt = 0
    stop_cond = False
    while not stop_cond:
        # generate mutations in seqs
        candidate_strands = mutate_seqs(test_strand,masks)
        cand_cnt = cand_cnt + 1
        if cand_cnt > cand_attempts: 
            stop_cond = True

        # calc Tm and regex
        Tms = np.zeros(len(candidate_strands))
        bad_seq = np.zeros(len(Tms),dtype=bool)
        for i in range(len(candidate_strands)):
            Tms[i] = get_tm(candidate_strands[i][five_p_overhang[i]:],Na=20,dnac1=500)
            if re.search(exclude_patterns,candidate_strands[i]):
                bad_seq[i] = True 
        if np.any(Tms<Tm_min): 
            continue
        if np.any(bad_seq): 
            continue

        # if we are here, that means that mutated strands pass Tm and exlclusions. Next we score for primer dimers. 
        # we need to account for all possible primer-dimers (Fwd vs Rev, Fwd vs Fwd, and Rev vs Rev)
        mfe_cnt = mfe_cnt+1 
        if mfe_cnt>mfe_attempts: 
            stop_cond = True
        test_strand = candidate_strands.copy()
        (oligo_pairs,bps,dGs) = all_pairwise_pairing(test_strand,my_nupack_model)

        
        if bps.max() <= min_bps and dGs.min() >= max_dG:
            min_bps = bps.max()
            max_dG = dGs.min()
            best_strand = test_strand.copy()
    return (best_strand,min_bps,max_dG)