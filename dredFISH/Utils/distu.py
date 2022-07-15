"""Distances
"""

import itertools
from pyemd import emd
import numpy as np
from scipy.spatial.distance import squareform 

def dist_emd(E1, E2, Dsim, out_format='compact'):
    """Earth mover distance
    outformat: compact or squareform
    """
    if E2 is None:
        E1 = E1.astype('float64')
        Dsim = Dsim.astype('float64')
        D = []
        for i, j in itertools.combinations(np.arange(E1.shape[0]), r=2):
            d = emd(E1[i,:], E1[j,:], Dsim)
            D.append(d)

    else:
        sum_of_rows = E1.sum(axis=1)
        E1 = E1 / sum_of_rows[:, None]
        sum_of_rows = E2.sum(axis=1)
        E2 = E2 / sum_of_rows[:, None]

        E1 = E1.astype('float64')
        E2 = E2.astype('float64')
        Dsim = Dsim.astype('float64')
        D = []
        for i in range(E1.shape[0]):
            e1=E1[i,:]
            e2=E2[i,:]
            d = emd(e1,e2,Dsim)
            D.append(d)

    if out_format == 'compact':
        return D
    elif out_format == 'squareform':
        return squareform(D)
