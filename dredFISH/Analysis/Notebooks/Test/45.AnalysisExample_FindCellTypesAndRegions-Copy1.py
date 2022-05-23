#!/usr/bin/env python
# coding: utf-8

# #### Imports
from dredFISH.Analysis.TissueGraph import *
from dredFISH.Visualization.Viz import *
import ipyparallel as ipp
import matplotlib.pyplot as plt 
import logging
logging.basicConfig(level=logging.INFO)
ti = time.time()
logging.info(f"Time spent: {time.time()-ti}")

# #### Load data
base_path = '/bigstore/Images2021/gaby/dredFISH/DPNMF_PolyA_2021Nov19/'
dataset = 'DPNMF_PolyA_2021Nov19'
TMG = TissueMultiGraph()
XY,PNMF = TMG.load_and_normalize_data(base_path,dataset,norm_bit = 'robustZ-iqr',norm_cell = 'l1')
logging.info(f"Time spent: {time.time()-ti}")

# #### Build layers 1-2: cells and zones
TMG.create_cell_and_zone_layers(XY,PNMF)
logging.info(f"Time spent: {time.time()-ti}")

TMG.add_geoms()
logging.info(f"Time spent: {time.time()-ti}")

TMG.Layers[0].calc_entropy_at_different_Leiden_resolutions()
logging.info(f"Time spent: {time.time()-ti}")

# this takes long....
#  ipcluster start -n 4  # to start client
# topics = TMG.find_topics(use_parallel=True) # need to first start the engine
topics = TMG.find_topics(use_parallel=False) # this take long? 
logging.info(f"Time spent: {time.time()-ti}")

TMG.create_region_layer(topics)
logging.info(f"Time spent: {time.time()-ti}")

# ### Save
TMG.save('/bigstore/GeneralStorage/fangming/projects/dredfish/data_dump/TMG_example_3.pkl')
logging.info(f"Time spent: {time.time()-ti}")