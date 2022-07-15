import pandas as pd
import numpy as np 

from cell_colors import colors_from_list
from utils import get_aggregate_counts, cut_dend, ct2subct, avg_dend

from copy import copy

# metadata
mpath="/home/jperrie/Documents/max_biocart/HippocampusCellTypeCalls.csv"
meta=pd.read_csv(mpath, index_col=0)

# dendrogram
dpath = "/home/jperrie/Documents/max_biocart/AllenDendrogramAsTable.csv" 
dend = pd.read_csv(dpath, index_col=0)
dend=dend.loc[dend["H"]>0]
dend.sort_values(by="H",inplace=True)
dend.set_index(np.arange(len(dend)))
dend["A"]=dend["A"].astype(str)
dend["B"]=dend["B"].astype(str)
dend["C"]=dend["C"].astype(str)

# list colors
fpath = "/home/rlittman/JSTA_classified_celltypes/data/hippocampus.merfish.jsta.segmented.counts.csv.gz"
counts = get_aggregate_counts(fpath) 
color_1 = colors_from_list(counts.index, copy(counts), lum=50)

sub_labels = cut_dend(dend, copy(meta["cell_type_low"].values), 0.125)
cell_map = ct2subct(meta["cell_type_low"].values, sub_labels)
sub_labels = list(map(lambda x: cell_map[x], counts.index))
color_2 = colors_from_list(sub_labels, copy(counts), lum=50)

# dendrogram colors 
dend_counts = avg_dend(dend, counts)
color_3 = colors_from_list(dend_counts.index, copy(dend_counts), lum=50)