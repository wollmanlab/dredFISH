#!/usr/bin/env python
# coding: utf-8
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt 
from dredFISH.Analysis import TissueGraph
from dredFISH.Analysis import Classification
from dredFISH.Utils import tmgu


# ### Import and create an empty TMG
basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t4'
TMG = TissueGraph.TissueMultiGraph(basepath=basepth, 
                                   redo=True, # create an empty one
                                  ) 
# ### Create a `cell` layer
# Creating a cell layer, load data from file, normalizes and creates an unclassified tissue graph
TMG.create_cell_layer(metric='cosine')
logging.info(f"TMG has {len(TMG.Layers)} Layers")

# ### Save to files
# TMG is saved as a config json file, one AnnData file per layer, and one dataframe per taxonomy. 
TMG.save()