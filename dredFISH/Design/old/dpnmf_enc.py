import json
import numpy as np
import pandas as pd

reference_path = '/bigstore/binfo/mouse/Brain/DRedFISH/Allen_V3_Reference/'
model_path = '/home/jperrie/Documents/neural_network_probe_set/'

dpnmf_embmat = pd.read_csv(reference_path+"10X_dpnmf/weights.csv", index_col=0)
tmp=np.array(json.load(open(model_path+'results/embmat=max-half_nrml-90000.0-1.35E+05-24-0-1.75E-09-0.01-1.0.json')))