#!/usr/bin/env python
import logging
import numpy as np
# import torch
import os
import importlib
# from tqdm import tqdm
from datetime import datetime
# from metadata import Metadata
# import sys
# import pandas as pd
from dredFISH.Utils import fileu
# import anndata

import argparse
import shutil
import time
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
import time
from metadata import Metadata

import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from collections import Counter
import torch

from dredFISH.Registration.Registration import *

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config_tree', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='all', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-w","--well", type=str,dest="well",default='', action='store',help="keyword in well to identify which section to process")
    parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='infer', action='store',help="fishdata name for save directory")
    
    args = parser.parse_args()


if __name__ == '__main__':
    """
     Main Executable to process raw Images to dredFISH data
    """
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)
    config.parameters['nucstain_acq']
    if args.fishdata=='infer':
        processing_paths = [i for i in os.listdir(metadata_path) if 'Processing_' in i]
        processing_date = [os.path.getctime(os.path.join(metadata_path,processing)) for processing in processing_paths]
        sorted_processing_paths = [x for _, x in reversed(sorted(zip(processing_date, processing_paths)))]
        fishdata = sorted_processing_paths[0]
    elif not '_' in args.fishdata:
        fishdata = args.fishdata+str(datetime.today().strftime("_%Y%b%d"))
    else:
        fishdata = args.fishdata

    fishdata_path = os.path.join(metadata_path,fishdata)
    print(args)
    if args.section=='all':
        image_metadata = Metadata(metadata_path)
        hybe = [i for i in image_metadata.acqnames if config.parameters['nucstain_acq']+'_' in i.lower()]
        posnames = np.unique(image_metadata.image_table[np.isin(image_metadata.image_table.acq,hybe)].Position)
        sections = np.unique([i.split('-Pos')[0] for i in posnames if '-Pos' in i])
    else:
        sections = np.array([args.section])
    if sections.shape[0]==0:
        sections = np.array(['Section1'])
    if args.well!='':
        sections = np.array([i for i in sections if args.well in i])
    # np.random.shuffle(sections)
    print(sections)

    """ Determine best order to register """
    
    completion_array = np.array([False for i in sections])
    reference_data = None
    while np.sum(completion_array==False)>0:
        for idx,section in enumerate(sorted(sections)):
            processing_path = os.path.join(metadata_path,fishdata)
            if not os.path.exists(processing_path):
                processing_path = os.path.join(metadata_path,fishdata,'Processing')
            registration_path = os.path.join(metadata_path,'Registration'+str(datetime.today().strftime("_%Y%b%d")))
            self = Registration_Class(processing_path,registration_path,section)
            self.overwrite = True
            self.ref_XYZC = reference_data
            self.update_user(str(np.sum(completion_array==False))+ ' Unfinished Sections')
            self.update_user('Processing Section '+section)
            out = str(robust_input("Continue? (Y/N): ",dtype='str'))
            if 'y' in out.lower():
                self.run()
                reference_data = self.ref_XYZC
            completion_array[idx] = True
        if np.sum(completion_array==False)>0:
            time.sleep(60)
    out = ''
    while not 'y' in out:
        out = str(robust_input("Satisfied? (Y/N): ",dtype='str'))