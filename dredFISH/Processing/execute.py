#!/usr/bin/env python
import argparse
from dredFISH.Processing.Dataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='', action='store',help="keyword in posnames to identify which section to process")
    
    args = parser.parse_args()
    
if __name__ == '__main__':
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    print(args)
    dataset = [i for i in metadata_path.split('/') if not i==''][-1]
    self = Dataset_Class(metadata_path,dataset,cword_config,verbose=True)
    self.config.parameters['brain'] = args.section
    self.run()
