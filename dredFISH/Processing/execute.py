#!/usr/bin/env python
import argparse
import shutil
from datetime import datetime
import dredFISH.Processing as Processing
from dredFISH.Processing.Section import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='fishdata', action='store',help="fishdata name for save directory")
    
    args = parser.parse_args()
    
if __name__ == '__main__':
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    if args.fishdata == 'fishdata':
        fishdata = args.fishdata+str(datetime.today().strftime("_%Y%b%d"))
    else:
        fishdata = args.fishdata
    print(args)
    dataset = [i for i in metadata_path.split('/') if not i==''][-1]
    if args.section=='all':
        image_metadata = Metadata(metadata_path)
        hybe1s = [i for i in image_metadata.acqnames if 'hybe1_' in i]
        posnames = np.unique(image_metadata.image_table[np.isin(image_metadata.image_table.acq,hybe1s)].Position)
        sections = np.unique([i.split('-')[0] for i in posnames])
    else:
        sections = [args.section]
    np.random.shuffle(sections)
    for section in sections:
        print('Processing Section ',section)
        self = Section_Class(metadata_path,dataset,section,cword_config,verbose=True)
        self.config.parameters['fishdata'] = fishdata
        self.out_path = os.path.join(self.metadata_path,self.config.parameters['fishdata'])
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        src = os.path.join(Processing.__file__.split('dredFISH/P')[0],args.cword_config+'.py')
        dst = os.path.join(self.out_path,args.cword_config+'.py')
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copyfile(src, dst)
        self.run()
