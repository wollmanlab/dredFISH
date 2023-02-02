# Utilities for sequences

from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

def get_gc(seq):
    """
    """
    return (seq.count('G') + seq.count('C'))/len(seq)

def get_tm(seq, fmd=0, Na=1e-5, dnac1=0, dnac2=0):
    """seq is a Bio.Seq.Seq object
    myseq = Seq(mystring)
    """
    if isinstance(seq, str):
        _seq = Seq(seq)
    else:
        _seq = seq
    res = mt.Tm_NN(_seq, Na=Na, dnac1=dnac1, dnac2=dnac2)
    res = mt.chem_correction(res, fmd=fmd)
    return res
    