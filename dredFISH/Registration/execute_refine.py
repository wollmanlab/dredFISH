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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import Counter
import torch

from dredFISH.Registration.Registration import *

import matplotlib as mpl

import os
from dredFISH.Utils import fileu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import socket
mpl.rcParams['figure.dpi'] = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("user", type=str,default='Aihui', help="Aihui, Gaby, or Haley")
    args = parser.parse_args()

if __name__ == '__main__':
    user = args.user
    Total = 0
    not_completed_datasets = []
    completed_datasets = []
    processing_key = 'Processing_2024May28'
    project_path = '/scratchdata1/Images2024/Zach/MouseBrainAtlas/'
    datasets = [i for i in os.listdir(project_path) if '_' in i]
    status = {}
    for dataset in sorted(datasets):
        processing_path = os.path.join(project_path,dataset,processing_key)
        if not os.path.exists(processing_path):
            continue
        status[dataset] = {}
        sections = [i for i in os.listdir(processing_path) if '-' in i]
        for section in sections:
            """ Check if processing is finished """
            data_path = fileu.generate_filename(path=os.path.join(processing_path,section),hybe='',channel='',file_type='anndata')
            if os.path.exists(data_path):
                """ Check if Registration is finished """
                registration_paths = [i for i in os.listdir(os.path.join(project_path,dataset)) if 'Registration' in i]
                registered = False
                for registration_path in registration_paths:
                    if not section in os.listdir(os.path.join(project_path,dataset,registration_path)):
                        continue
                    if os.path.exists(fileu.generate_filename(path=os.path.join(project_path,dataset,registration_path,section),hybe='',channel='Y',file_type='Model')):
                        registered = True
                        break
                if registered:
                    status[dataset][section] = 'Registered'
                else:
                    status[dataset][section] = 'Processed'
            else:
                status[dataset][section] = 'Not Processed'
            dataset_status = status[dataset]
        processed = np.sum([value=='Processed' for section,value in dataset_status.items()])
        registered = np.sum([value=='Registered' for section,value in dataset_status.items()])
        not_processed = np.sum([value=='Not Processed' for section,value in dataset_status.items()])
        total = 0
        total+=processed
        total+=registered
        Total+=total
        if not_processed > 0:
            not_completed_datasets.append(dataset)
            print(f"{total} Completed Sections for \033[91m{dataset}\033[0m {not_processed}")
        elif registered>0:
            completed_datasets.append(dataset)
            print(f"{total} Completed Sections for \033[92m{dataset}\033[0m ")
        else:
            completed_datasets.append(dataset)
            print(f"{total} Completed Sections for \033[93m{dataset}\033[0m ")
    print(f"{Total} Completed Sections for All")



    animals = np.unique([i.split('_')[0] for i in completed_datasets if not 'GapFill' in i])


    master_section_df_path = '/scratchdata1/MouseBrainAtlas_Sections.csv'
    master_section_df = pd.concat([fileu.create_input_df(project_path, animal) for animal in animals],ignore_index=True)
    ticker = 0
    converter = {}
    for well in ['A','B','C','D','E','F']:
        for section in [1,2,3,4]:
            if ticker == 0:
                converter[f"Well{well}-Section{section}"] = 'Aihui'
            elif ticker == 1:
                converter[f"Well{well}-Section{section}"] = 'Gaby'
            else:
                ticker = -1
                converter[f"Well{well}-Section{section}"] = 'Haley'
            ticker+=1
    master_section_df['user'] = master_section_df['section_acq_name'].map(converter)


    # master_section_df['unique_name'] = master_section_df['dataset'] + '_' + master_section_df['section_acq_name']
    # converter = {0:'orange',1:'blue',2:'purple'}
    # master_section_df['server'] = np.random.randint(0,3,len(master_section_df))
    # master_section_df['server'] = master_section_df['server'].map(converter)
    # if False:#os.path.exists(master_section_df_path):
    #     old_master_section_df = pd.read_csv(master_section_df_path,index_col=0)
    #     # update servers to match old
    #     master_section_df['server'] = old_master_section_df['unique_name'].map(master_section_df.set_index('unique_name')['server'])
    master_section_df.to_csv(master_section_df_path)
    master_section_df

    # host = socket.gethostname()
    reference_data = None
    master_section_df = master_section_df[master_section_df['user']==user]
    print(master_section_df.shape)

    if user=='Zach':
        project_path = '/scratchdata1/Images2024/Haley/dredfish/'
        master_section_df = fileu.create_input_df(project_path, 'RNA') 
        master_section_df = master_section_df[master_section_df['dataset']=='RNA_2024Jul23']
    ticker = 0
    for idx,row in master_section_df.iterrows():
        ticker+=1
        print(row['dataset'],row['section_acq_name'])
        print(row['registration_path'])
        print(row['processing_path'])
        print(f"{idx} : {ticker} out of {master_section_df.shape[0]}")
        self = Registration_Class(row['processing_path'],row['registration_path'],row['section_acq_name'])
        self.ref_XYZC = reference_data
        self.update_user(f"Processing Section {row['dataset']} : {row['section_acq_name']}")
        out = str(robust_input("Refine? (Y/N[skip]): ",dtype='str'))
        if 'y' in out.lower():
            self.refine()
            reference_data = self.ref_XYZC
