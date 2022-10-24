from Analysis.Graph import TissueGraph8 as tg
from Viz.utils import *
from Viz.cell_colors import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from copy import copy

# run from Python dir 

mpath="/home/jperrie/Documents/max_biocart/HippocampusCellTypeCalls.csv"
meta=pd.read_csv(mpath, index_col=0)
XY = meta[["x_coordinate","y_coordinate"]].values

fpath = "/home/rlittman/JSTA_classified_celltypes/data/hippocampus.merfish.jsta.segmented.counts.csv.gz"
counts = get_aggregate_counts(fpath) 

dpath = "/home/jperrie/Documents/max_biocart/AllenDendrogramAsTable.csv" 
dend = pd.read_csv(dpath, index_col=0)
dend=dend.loc[dend["H"]>0]
dend.sort_values(by="H",inplace=True)
dend.set_index(np.arange(len(dend)))
dend["A"]=dend["A"].astype(str)
dend["B"]=dend["B"].astype(str)
dend["C"]=dend["C"].astype(str)

# dendrogram colors 
dend_counts = avg_dend(dend, counts)
color_dict = colors_from_list(dend_counts.index, copy(dend_counts), lum=50)

TG = tg.TissueGraph()
TG = TG.BuildSpatialGraph(XY)
TG.Type=meta["cell_type_low"].values
TG.plot(color_dict=color_dict)
plt.savefig("tmp1.png")
plt.clf()

cell_type_cut=cut_dend(dend,np.array(TG.Type),0.125)
CG=TG.ContractGraph(cell_type_cut)
CG.plot(XY=TG.XY,cell_type=TG.Type,color_dict=color_dict)
plt.savefig("tmp2.png")
plt.clf()

CG.plot(XY=TG.XY,cell_type=TG.Type,color_dict=color_dict,inner=True)
plt.savefig("tmp3.png")
plt.clf()
