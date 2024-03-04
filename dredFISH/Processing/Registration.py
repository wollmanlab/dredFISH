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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import Counter



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config_tree', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='all', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-w","--well", type=str,dest="well",default='', action='store',help="keyword in well to identify which section to process")
    parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='fishdata', action='store',help="fishdata name for save directory")
    
    args = parser.parse_args()



class Registration_Class(object):
    def __init__(self,
                 fishdata_path,
                 metadata_path,
                 section,
                 cword_config,
                 verbose=True):
        self.completed = False
        self.metadata_path = metadata_path
        self.dataset = [i for i in self.metadata_path.split('/') if i!= ''][-1]
        self.section = str(section)
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.image_metadata = ''
        self.verbose=verbose
        self.side = ''
        self.epochs = 10000

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=self.config.parameters['registration_log_level'])
        self.log = logging.getLogger("Registration")
        """ Convert to Checks not making directories These things should already exist """

        # """ maybe make a registrationdata path within fishdata"""
        # if self.config.parameters['fishdata'] == 'fishdata':
        #     fishdata = self.config.parameters['fishdata']+str(datetime.today().strftime("_%Y%b%d"))
        # else:
        #     fishdata = self.config.parameters['fishdata']
        self.path = fishdata_path
        if not os.path.exists(self.path):
            self.update_user(self.path,level=50)
            self.update_user('No fishdata Path Found',level=50)
        self.path = os.path.join(self.path,self.dataset)
        if not os.path.exists(self.path):
            self.update_user(self.path,level=50)
            self.update_user('No Dataset Path Found',level=50)
        self.path = os.path.join(self.path,self.section)
        if not os.path.exists(self.path):
            self.update_user(self.path,level=50)
            self.update_user('No Section Path Found',level=50)

        logging.basicConfig(
                    filename=os.path.join(self.path,'registration_log.txt'),filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',level=self.config.parameters['registration_log_level'], force=True)
        self.log = logging.getLogger("Registration")

    def run(self):
        """ Load MERFISH Data """
        self.load_reference_data()
        self.model_type = 'total' #FIX
        self.load_data()
        self.preview()
        self.filter_cells()
        self.center_rotate()
        self.split()
        self.df_points_list = []
        self.reference_data_list = []
        self.window = 0.25
        self.all_data = self.data.copy()
        for side in ['left','right']:
            self.update_user(f"Starting {side} Side")
            plt.close('all')
            self.side = side
            self.m = self.all_data.obs['side'] == self.side
            self.data = self.all_data[self.m].copy()

            self.preview(prefix='ref_')
            out = str(robust_input("Skip This Side (Y/N): ",dtype='str'))
            if 'y' in out.lower():
                self.update_user(f"Manually Skipping {side} Side")
                continue
            self.set_section_index()
            self.set_registration_points()
            self.calculate_transformation()
            self.apply_transformation()
            self.view_transformation()
            completed = False
            while not completed:
                message = str(robust_input(" Ready To Save?",dtype='str')).lower()
                if message=='y':
                    print('saving')
                    self.save_data()
                    completed = True
                elif message =='n':
                    completed = True

        plt.close('all')
        self.side = ''
        self.data = self.all_data.copy()
        del self.all_data
        self.data.obs['ref_z'] = np.abs(self.data.obs['ref_z'])
        self.data.uns['registration_points'] = pd.concat(self.df_points_list)
        self.data.uns['reference_data'] = pd.concat(self.reference_data_list)
        completed = False
        while not completed:
            message = str(robust_input(" Ready To Save All Data ?",dtype='str')).lower()
            if message=='y':
                print('saving')
                self.save_data()
                completed = True
            elif message =='n':
                completed = True
        self.completed = True
    
    def update_user(self,message,level=20):
        """
        update_user Wrapper to fileu.update_user

        :param message: Message to be logged
        :type message: str
        :param level: priority 0-50, defaults to 20
        :type level: int, optional
        """
        if self.verbose:
            print(message)
        fileu.update_user(message,level=level,logger=self.log)
        if level==50:
            raise ValueError(message)

    def check_existance(self,hybe='',channel='',file_type='',model_type=''):
        """
        check_existance wrapper to fileu.check_existance

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        :return : True of file exists, False otherwise
        :rtype : bool
        """
        return fileu.check_existance(self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_filename(self,hybe,channel,file_type,model_type=''):
        """
        fname Wrapper to fileu.generate_filename Generate Filename

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str
        :return: File Path
        :rtype: str
        """
        return fileu.generate_filename(self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def save(self,data,hybe='',channel='',file_type='',model_type=''):
        """
        save Wrapper to fileu.save Sae Files

        :param data: data object to be saved
        :type path: object
        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        """
        fileu.save(data,path=self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,dataset=self.dataset,section=self.section+self.side,logger=self.log)

    def load(self,hybe='',channel='',file_type='anndata',model_type=''):
        """
        load Wrapper to fileu.load Load Files

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        :return : Desired Data Object or None if not found
        :rtype : object
        """
        return fileu.load(path=self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_iterable(self,iterable,message,length=0):
        """
        generate_iterable Generate Iterable, Can be loading bar if self.verbose is True

        :param iterable: object to be iterated over 
        :type iterable: list
        :param message: short description of what is being iterated over
        :type message: str
        :param length: length of iterable if complex data type, defaults to ''
        :type length: int, optional
        :return: either initial iterable or iterable with loading bar
        :rtype: list,ProgressDialog?
        """
        if self.verbose:
            self.update_user(message,level=10)
            if length==0:
                return tqdm(iterable,desc=str(datetime.now().strftime("%H:%M:%S"))+' '+message)
            else:
                return tqdm(iterable,total=length,desc=str(datetime.now().strftime("%H:%M:%S"))+' '+message)
        else:
            return iterable
        
    def load_reference_data(self):
        """ Load Reference Coordinates"""
        self.update_user('Loading Reference Data',level=20)

        """ Update to use config file """
        """ /orangedata/Images2024/Zach/dredFISH/Tree20um37C500M1H3A2SDS24H100ug47C50F18H50F1H2SDS37C1H-FF.A_FF.B_FF.C_PFA.D_PFA.E_PFA.F_2024Feb15/fishdata_2024Feb01/Notebooks/reference_sections.ipynb"""
        reference_data = pd.read_csv('/orangedata/ExternalData/WMB_Spatial_Data_ZH2024Feb22.csv',index_col=0,low_memory=False)
        reference_data = reference_data.astype(str)
        reference_data['ccf_x'] = reference_data['ccf_x'].astype(float)
        reference_data['ccf_y'] = reference_data['ccf_y'].astype(float)
        reference_data['ccf_z'] = reference_data['ccf_z'].astype(float)
        # reference_data['x'] = reference_data['x'].astype(float)
        # reference_data['y'] = reference_data['y'].astype(float)
        # reference_data['z'] = reference_data['z'].astype(float)
        self.reference_data = reference_data

    def load_data(self):
        """ Load processed anndata object from Section_Class """
        self.update_user('Loading Data',level=20)

        try:
            self.data = self.load(file_type='anndata',model_type=self.model_type)

            self.data.obs['x'] = self.data.obs['stage_x']
            self.data.obs['y'] = self.data.obs['stage_y']
            self.data.obs['z'] = np.ones_like(self.data.obs['stage_x'])
            self.data.obs['c'] = np.ones_like(self.data.obs['stage_x'])
        except Exception as e:
           self.update_user(str(e),level=40) 
           self.update_user('Unable to Load Data',level=50) 
        if isinstance(self.data,type(None)):
            self.update_user('Data Not Found',level=50) 

    def filter_cells(self):
        """ Remove Non Cells Manually """
        self.update_user('Filtering Cells',level=20)
        """ Filter Out Non Cells"""
        x_column = 'dapi'
        y_column = 'size'
        x = self.data.obs[x_column]
        y = self.data.obs[y_column]
        c = self.data.X.sum(1)
        c = np.log10(np.clip(c,1,None))
        x = np.log10(np.clip(x,1,None))
        y = np.sqrt(np.clip(y,2,None))*self.config.parameters['pixel_size']
        # y = np.log10(np.clip(y,1,None))
        vmin,vmax = np.percentile(c,[5,95])
        c = np.clip(c,vmin,vmax)
        plt.figure(figsize=[5,5])
        plt.title('Filter Cells')
        idx = np.random.choice(np.array(range(x.shape[0])),100000)
        plt.scatter(x[idx],y[idx],s=0.1,c=c[idx],cmap='jet')
        plt.xlabel('Log '+x_column)
        plt.ylabel(y_column)
        plt.grid()
        cbar = plt.colorbar()
        cbar.ax.set_title('Log Sum')
        plt.show(block=False)
        time.sleep(2)
        print(' ')
        print(' Set Filter Gates ')
        x_min = float(robust_input("Enter min dapi value: ",dtype='float'))
        x_max = float(robust_input("Enter max dapi value: ",dtype='float'))

        y_min = float(robust_input("Enter min size value: ",dtype='float'))
        y_max = float(robust_input("Enter max size value: ",dtype='float'))

        plt.close('all')
        m = (x>x_min)&(x<x_max)&(y>y_min)&(y<y_max)

        plt.figure(figsize=[5,5])
        plt.title('Filter Gates')
        plt.plot([x_min,x_max],[y_min,y_min],c='k')
        plt.plot([x_min,x_max],[y_max,y_max],c='k')
        plt.plot([x_min,x_min],[y_min,y_max],c='k')
        plt.plot([x_max,x_max],[y_min,y_max],c='k')
        idx = np.random.choice(np.array(range(x.shape[0])),100000)
        plt.scatter(x[idx],y[idx],s=0.1,c=c[idx],cmap='jet')
        plt.grid()
        plt.xlabel('Log '+x_column)
        plt.ylabel('Log '+y_column)
        cbar = plt.colorbar()
        cbar.ax.set_title('Log Sum')
        path = self.generate_filename('Registration','FilterGates','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        self.data = self.data[m,:]

        X = np.array(self.data.obs['x'].values).ravel()
        Y = np.array(self.data.obs['y'].values).ravel()
        C = self.data.X.sum(1)
        C = np.log10(np.clip(C,1,None))
        vmin,vmax = np.percentile(C,[5,95])
        C = np.clip(C,vmin,vmax)
        x = X
        y = Y
        c = C.copy()
        plt.figure(figsize=[5,5])
        plt.title('Filtered Cells Preview')
        idx = np.random.choice(np.array(range(x.shape[0])),100000)
        plt.scatter(x[idx],y[idx],s=0.1,c=c[idx],cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.set_title('Log Sum')
        path = self.generate_filename('Registration','FilteredCells','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)



    def fix_tears(self):
        """ Sew together tears in sections """
        self.update_user('Fixing Tears',level=20)


    def center_rotate(self):
        """ Center Cells and rotate using PCA Perform Rough Scaling"""
        self.update_user('Centering & Rotating Data',level=20)

        """ Center and Rotate """
        completed = False
        angle = 0 
        while not completed:
            """ Center and Rotate then scale"""
            X = np.array(self.data.obs['x'].values).ravel()
            Y = np.array(self.data.obs['y'].values).ravel()
            Z = np.array(self.data.obs['z'].values).ravel().astype(str)

            """ Center """
            X = X-np.median(X)
            Y = Y-np.median(Y)

            """ Rotate"""
            X,Y = rotate_points(X, Y, angle)

            """ Rough Scale """
            # Move to mm
            X = X*0.490/1000
            Y = Y*0.490/1000
            Y = Y+np.mean(self.reference_data.loc[:,'ccf_y'])

            C = self.data.X.sum(1)
            C = np.log10(np.clip(C,1,None))
            vmin,vmax = np.percentile(C,[5,95])
            C = np.clip(C,vmin,vmax)
            x = X.copy()
            y = Y.copy()
            c = C.copy()
            idx = np.random.choice(self.reference_data.index,100000)
            
            fig = plt.figure(figsize=[5,5])
            fig.suptitle('Centered and Rotated?')
            plt.scatter(self.reference_data.loc[idx,'ccf_z'],self.reference_data.loc[idx,'ccf_y'],s=0.1,c='k')
            plt.scatter(x,y,s=0.1,c=c,cmap='jet')
            cbar = plt.colorbar()
            cbar.ax.set_title('Log Sum')
            path = self.generate_filename('Registration','RoughRegistration','Figure',model_type=self.model_type)
            plt.savefig(path)
            plt.show(block=False)
            print(' ')
            out = str(robust_input("Satisfied? (Y/ Set Angle): ",dtype='str'))
            if 'y' in out.lower():
                completed = True
            else:
                angle = float(out)
                plt.close(fig)

        """ Wierd mapping """
        self.data.obs['ref_x'] = 0
        self.data.obs['ref_y'] = Y
        self.data.obs['ref_z'] = X
        self.data.obs['side'] = 'center'

    def split(self):
        """
        Draw 1 or more boxes to seperate into zones 
        """
        self.update_user('Splitting Data',level=20)
        """ Split Left Right"""
        completed = False
        center = 0
        while not completed:
            plt.figure(figsize=[5,5])
            plt.title('Set Center : '+str(center))
            m = self.data.obs['ref_z']>center
            plt.scatter(self.data.obs['ref_z'][m],self.data.obs['ref_y'][m],s=0.1,c='m')
            plt.scatter(self.data.obs['ref_z'][m==False],self.data.obs['ref_y'][m==False],s=0.1,c='c')
            path = self.generate_filename('Registration','LeftRightSplit','Figure',model_type=self.model_type)
            plt.savefig(path)
            plt.show(block=False)

            print(' ')
            out = str(robust_input("Satisfied? (Y/ Next Center): ",dtype='str'))
            if 'y' in out.lower():
                completed = True
            else:
                center= float(out)
                plt.close('all')

        self.data.obs['ref_z'] = self.data.obs['ref_z']-center
        for side in ['left','right']:
            if side =='right':
                m = self.data.obs['ref_z']<=0
            else:
                m = self.data.obs['ref_z']>0
            self.data.obs.loc[m,'side'] = side
        self.data.obs['ref_z'] = np.abs(self.data.obs['ref_z'])

        for side,cc in Counter(self.data.obs.loc[:,'side'].values).items():
            self.update_user(f"{str(cc)} cells found in {side} side")


    def set_section_index(self):
        """ Choose with of the reference sections matches zone best"""
        self.update_user('Setting Section Index',level=20)
        """ Set Reference Section """
        self.ref_z  = np.array(self.reference_data['ccf_z'])
        self.ref_y = np.array(self.reference_data['ccf_y'])
        self.ref_c = np.array(self.reference_data['c'])
        x = np.array(self.reference_data['ccf_x'])
        # unique_x = np.array(sorted(np.unique(x)))
        completed = False
        self.ref_x = None
        while not completed:
            if isinstance(self.ref_x,type(None)):
                self.ref_x = float(robust_input("Enter Reference Section: ",dtype='float'))
            self.ref_m = (x>self.ref_x-self.window)&(x<self.ref_x+self.window)
            self.ref_c = np.array(pd.Categorical(self.ref_c).codes)
            C = self.data.X.sum(1)
            C = np.log10(np.clip(C,1,None))
            vmin,vmax = np.percentile(C,[5,95])
            C = np.clip(C,vmin,vmax)

            fig,axs = plt.subplots(1,2,figsize=[10,5])
            fig.suptitle(self.side+': Set Reference Section: '+str(self.ref_x))
            axs = axs.ravel()
            axs[0].scatter(self.ref_z[self.ref_m],self.ref_y[self.ref_m],s=0.1,c=self.ref_c[self.ref_m],cmap='jet')
            axs[1].scatter(self.data.obs['ref_z'],self.data.obs['ref_y'],s=0.1,c=C,cmap='jet')
            path = self.generate_filename('Registration','ReferenceMatching','Figure',model_type=self.model_type)
            plt.savefig(path)
            plt.show(block=False)

            print(' ')
            out = str(robust_input("Satisfied? (Y/ Next Reference Z): ",dtype='str'))
            if 'y' in out.lower():
                completed = True
            else:
                self.ref_x = float(out)
                plt.close('all')
        # SAVE REFERENCE Z AND XYZ
        print('Selected Reference Section '+str(self.ref_x))
        self.all_data.obs.loc[self.m,'ref_x'] = self.ref_x


    def set_registration_points(self):
        """ Pair Points between Reference and Data"""
        self.update_user('Setting Registation Points',level=20)
        plt.close('all')
        def set_registration_points(side,path,X,Y,C,ref_X,ref_Y,ref_C):
            def onpick(event):
                xmouse = event.mouseevent.xdata
                ymouse = event.mouseevent.ydata
                points_cor_x.append(xmouse)
                points_cor_y.append(ymouse)
                if len(points_cor_x)%2!=0:
                    # print('Fixed',round(xmouse,3),round(ymouse,3))
                    axs[0].scatter(xmouse, ymouse , marker = 'x', color = 'k', s = 100, linewidth = 1)
                    axs[0].annotate(str(math.ceil(len(points_cor_x)/2)), 
                                    xy = [xmouse, ymouse], color = 'k', fontsize = 12)
                else:
                    # print('Moving',round(xmouse,3),round(ymouse,3))
                    axs[1].scatter(xmouse, ymouse , marker = 'x', color = 'k', s = 100, linewidth = 1)
                    axs[1].annotate(str(math.ceil(len(points_cor_x)/2)), 
                            xy = [xmouse, ymouse], color = 'k', fontsize = 12)
                plt.draw()

            points_cor_x = []
            points_cor_y = []

            fig,axs = plt.subplots(1,2,figsize=[10,5])
            fig.suptitle(side+': Pick Registration Points')
            axs = axs.ravel()
            # img = np.histogram2d(ref_X,ref_Y,bins=1000)[0]
            # vmin,vmax = np.percentile(img.ravel(),[5,95])
            # img = np.clip(img,vmin,vmax)
            # axs[0].imshow(img,cmap='Greys')
            x = np.linspace(ref_X.min(),ref_X.max(),100000)
            y = np.linspace(ref_Y.min(),ref_Y.max(),100000)
            np.random.shuffle(x)
            np.random.shuffle(y)
            axs[0].scatter(x,y,s=5,c='w',picker=True)
            idxs = np.random.choice(np.array(range(ref_X.shape[0])),100000)
            axs[0].scatter(ref_X[idxs],ref_Y[idxs],s=0.1,c=ref_C[idxs],cmap='jet')#,picker=True)

            # img = np.histogram2d(X,Y,bins=1000)[0]
            # vmin,vmax = np.percentile(img.ravel(),[5,95])
            # img = np.clip(img,vmin,vmax)
            # axs[1].imshow(img,cmap='Greys')
            x = np.linspace(X.min(),X.max(),100000)
            y = np.linspace(Y.min(),Y.max(),100000)
            np.random.shuffle(x)
            np.random.shuffle(y)
            axs[1].scatter(x,y,s=5,c='w',picker=True)
            idxs = np.random.choice(np.array(range(X.shape[0])),100000)
            axs[1].scatter(X[idxs],Y[idxs],s=0.1,c=C[idxs],cmap='jet')#,picker=True)
            fig.canvas.mpl_connect('pick_event', onpick)
            # plt.savefig(path)
            plt.show()


            fig,axs = plt.subplots(1,2,figsize=[10,5])
            fig.suptitle(side+': Pick Registration Points')
            axs = axs.ravel()
            idxs = np.random.choice(np.array(range(ref_X.shape[0])),100000)
            axs[0].scatter(ref_X[idxs],ref_Y[idxs],s=0.1,c=ref_C[idxs],cmap='jet')
            idxs = np.random.choice(np.array(range(X.shape[0])),100000)
            axs[1].scatter(X[idxs],Y[idxs],s=0.1,c=C[idxs],cmap='jet')
            x = np.array(points_cor_x)
            y = np.array(points_cor_y)
            for i in range(x.shape[0]):
                if len(i)%2!=0:
                    axs[0].scatter(x[i], y[i] , marker = 'x', color = 'k', s = 100, linewidth = 1)
                    axs[0].annotate(str(math.ceil(i/2)), 
                                    xy = [x[i], y[i]], color = 'k', fontsize = 12)
                else:
                    axs[1].scatter(x[i], y[i] , marker = 'x', color = 'k', s = 100, linewidth = 1)
                    axs[1].annotate(str(math.ceil(i/2)), 
                            xy = [x[i], y[i]], color = 'k', fontsize = 12)
            plt.savefig(path)
            plt.show()

            return points_cor_x,points_cor_y

        X = np.array(self.data.obs['ref_z'].values).ravel()
        Y = np.array(self.data.obs['ref_y'].values).ravel()
        Z = np.array(self.data.obs['ref_x'].values).ravel().astype(str)
        C = self.data.X.sum(1)
        C = np.log10(np.clip(C,1,None))
        vmin,vmax = np.percentile(C,[5,95])
        C = np.clip(C,vmin,vmax)
        path = self.generate_filename('Registration','RegistrationPointPicking','Figure',model_type=self.model_type)
        points_cor_x,points_cor_y = set_registration_points(self.side,path,X,Y,C,self.ref_z[self.ref_m],self.ref_y[self.ref_m],self.ref_c[self.ref_m])
        df_points = pd.DataFrame()
        df_points['fix_z'] = np.array(points_cor_x)[::2]
        df_points['fix_y'] = np.array(points_cor_y)[::2]
        df_points['mov_z'] = np.array(points_cor_x)[1::2]
        df_points['mov_y'] = np.array(points_cor_y)[1::2]
        df_points['side'] = self.side
        self.df_points_list.append(df_points)
        temp = self.reference_data[self.ref_m].copy()
        temp['side'] = self.side
        self.reference_data_list.append(temp)
        self.data.uns['registration_points'] = df_points
        self.data.uns['reference_data'] = temp

    def calculate_transformation(self):
        """ Use Points to calculate transformation"""
        self.update_user('Calculating Transformation',level=20)

        # Extract control points from data
        fix_z = np.array(self.data.uns['registration_points']['fix_z'])
        fix_y = np.array(self.data.uns['registration_points']['fix_y'])
        mov_z = np.array(self.data.uns['registration_points']['mov_z'])
        mov_y = np.array(self.data.uns['registration_points']['mov_y'])

        # Combine coordinates for model input
        inputs = np.hstack((mov_z[:, np.newaxis], mov_y[:, np.newaxis]))
        outputs = np.hstack((fix_z[:, np.newaxis], fix_y[:, np.newaxis]))

        # Create the neural network model
        model = Sequential([
            Dense(16, activation='relu', input_shape=(2,)),  # Hidden layer with ReLU activation
            Dense(16, activation='relu'),  # Hidden layer with ReLU activation
            Dense(2)  # Output layer with 2 neurons for predicted coordinates
        ])

        # Compile the model with appropriate optimizer, loss, and metrics
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Train the model on the control points
        model.fit(inputs, outputs, epochs=self.epochs, batch_size=150,verbose=0)  # Adjust epochs and batch size as needed

        self.model = model

    def apply_transformation(self):
        self.update_user('Applying Tranformation',level=20)

        predicted = self.model.predict(np.array(self.data.obs[['ref_z','ref_y']]))
        self.all_data.obs.loc[self.m,'ccf_y'] = predicted[:,1]
        self.all_data.obs.loc[self.m,'ccf_z'] = predicted[:,0]
        self.data.obs.loc[:,'ccf_y'] = predicted[:,1]
        self.data.obs.loc[:,'ccf_z'] = predicted[:,0]

    def view_transformation(self):
        self.update_user('Viewing Transformation',level=20)
        data = self.data.copy()
        model = self.model

        plt.figure(figsize=[5,5])
        plt.scatter(data.uns['registration_points']['fix_z'],data.uns['registration_points']['fix_y'],c='k',s=np.array(data.uns['registration_points'].index))
        plt.scatter(data.uns['registration_points']['mov_z'],data.uns['registration_points']['mov_y'],c='r',s=np.array(data.uns['registration_points'].index))
        for i in range(len(data.uns['registration_points'])):
            plt.plot([data.uns['registration_points']['fix_z'][i], data.uns['registration_points']['mov_z'][i]], [data.uns['registration_points']['fix_y'][i], data.uns['registration_points']['mov_y'][i]], c='b')
        path = self.generate_filename('Registration','RegPointsRaw','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        predicted = model.predict(np.array(data.uns['registration_points'][['mov_z','mov_y']]))
        data.uns['registration_points']['pred_x'] = predicted[:,0]
        data.uns['registration_points']['pred_y'] = predicted[:,1]
        plt.figure(figsize=[5,5])
        plt.scatter(data.uns['registration_points']['fix_z'],data.uns['registration_points']['fix_y'],c='k',s=np.array(data.uns['registration_points'].index))
        plt.scatter(data.uns['registration_points']['pred_x'],data.uns['registration_points']['pred_y'],c='r',s=np.array(data.uns['registration_points'].index))
        for i in range(len(data.uns['registration_points'])):
            plt.plot([data.uns['registration_points']['fix_z'][i], data.uns['registration_points']['pred_x'][i]], [data.uns['registration_points']['fix_y'][i], data.uns['registration_points']['pred_y'][i]], c='b')
        path = self.generate_filename('Registration','RegPointsAligned','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        predicted = model.predict(np.array(data.uns['registration_points'][['mov_z','mov_y']]))
        data.uns['registration_points']['pred_x'] = predicted[:,0]
        data.uns['registration_points']['pred_y'] = predicted[:,1]
        plt.figure(figsize=[5,5])
        plt.scatter(data.uns['registration_points']['mov_z'],data.uns['registration_points']['mov_y'],c='k',s=np.array(data.uns['registration_points'].index))
        plt.scatter(data.uns['registration_points']['pred_x'],data.uns['registration_points']['pred_y'],c='r',s=np.array(data.uns['registration_points'].index))
        for i in range(len(data.uns['registration_points'])):
            plt.plot([data.uns['registration_points']['mov_z'][i], data.uns['registration_points']['pred_x'][i]], [data.uns['registration_points']['mov_y'][i], data.uns['registration_points']['pred_y'][i]], c='b')
        path = self.generate_filename('Registration','RegPointsDistanceMoved','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        # predicted = np.array(data.obs[['ref_z','ref_y']])
        # x = predicted[:,0]
        # y = predicted[:,1]
        fig = plt.figure(figsize=[10,10])
        fig.suptitle('Raw vs Ref')
        plt.scatter(data.uns['reference_data'].loc[:,'ccf_z'],data.uns['reference_data'].loc[:,'ccf_y'],s=0.1,c='k')
        plt.scatter(data.obs['ref_z'],data.obs['ref_y'],s=0.1,c='r')
        path = self.generate_filename('Registration','RoughAlignment','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        # predicted = model.predict(np.array(data.obs[['ref_z','ref_y']]))
        # x = predicted[:,0]
        # y = predicted[:,1]
        fig = plt.figure(figsize=[10,10])
        fig.suptitle('Pred vs Ref')
        plt.scatter(data.uns['reference_data'].loc[:,'ccf_z'],data.uns['reference_data'].loc[:,'ccf_y'],s=0.1,c='k')
        plt.scatter(data.obs['ccf_z'],data.obs['ccf_y'],s=0.1,c='r')
        path = self.generate_filename('Registration','Aligned','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

        fig = plt.figure(figsize=[10,10])
        fig.suptitle('Distance Moved')
        # predicted = model.predict(np.array(data.obs[['ref_z','ref_y']]))
        # x = predicted[:,0]
        # y = predicted[:,1]
        c = np.sqrt(np.sum((np.array(data.obs[['ccf_z','ccf_y']])-np.array(data.obs[['ref_z','ref_y']]))**2,1))
        vmin,vmax = np.percentile(c,[1,99])
        c = np.clip(c,vmin,vmax)
        plt.scatter(data.obs['ccf_z'],data.obs['ccf_y'],s=0.1,c=c,cmap='jet')
        plt.colorbar()
        path = self.generate_filename('Registration','DistanceMoved','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

    def save_data(self):
        self.update_user('Saving Data',level=20)
        data = self.data.copy()
        # """ Cleaning"""
        # for column in ['x','y','z','ref_x','ref_y','ref_z']:
        #     try:
        #         data.obs.drop(column,axis=1,inplace=True)
        #     except Exception as e:
        #         self.update_user(str(e))
        self.save(
            data,
            file_type='anndata',
            model_type=self.model_type)

    def preview(self,prefix=''):
        """ Preview """
        if ('ref' in prefix)|('ccf' in prefix):
            X = np.array(self.data.obs[prefix+'z'].values).ravel()
        else:
            X = np.array(self.data.obs[prefix+'x'].values).ravel()
        Y = np.array(self.data.obs[prefix+'y'].values).ravel()
        C = self.data.X.sum(1)
        C = np.log10(np.clip(C,1,None))
        vmin,vmax = np.percentile(C,[5,95])
        C = np.clip(C,vmin,vmax)
        x = X
        y = Y
        c = C.copy()
        plt.figure(figsize=[5,5])
        plt.title('Preview')
        idx = np.random.choice(np.array(range(x.shape[0])),100000)
        plt.scatter(x[idx],y[idx],s=0.1,c=c[idx],cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.set_title('Log Sum')
        path = self.generate_filename('Registration','Preview','Figure',model_type=self.model_type)
        plt.savefig(path)
        plt.show(block=False)

# for interactively selecting corresponding points

import math

def rotate_points(X, Y, n_degrees):
    """
    Rotates a set of coordinates (X, Y) around the origin (0, 0) by n degrees.

    Args:
        X: A list of x-coordinates.
        Y: A list of y-coordinates.
        n_degrees: The rotation angle in degrees.

    Returns:
        A tuple of lists containing the rotated x and y coordinates.

    Raises:
        ValueError: If the lengths of X and Y are not equal.
    """

    if len(X) != len(Y):
        raise ValueError("X and Y coordinates must have the same length.")

    n_radians = math.radians(n_degrees)  # Convert degrees to radians
    cos_theta = math.cos(n_radians)
    sin_theta = math.sin(n_radians)

    rotated_x = []
    rotated_y = []
    for x, y in zip(X, Y):
        new_x = x * cos_theta - y * sin_theta
        new_y = x * sin_theta + y * cos_theta
        rotated_x.append(new_x)
        rotated_y.append(new_y)

    return np.array(rotated_x), np.array(rotated_y)

def robust_input(message,options=[],dtype=None):
    completed = False
    while not completed:
        try:
            out = str(input(message))
            if not isinstance(dtype,type(None)):
                if dtype=='str':
                    out = str(out)
                elif dtype=='float':
                    out = float(out)
                elif dtype=='int':
                    out = int(out)
                else:
                    raise ValueError(str(dtype)+' is not valid dtype')
            if len(options)==0:
                completed = True
            else:
                if out in options:
                    completed = True
                else:
                    print('Input not in options')
                    print(options)

        except Exception as e:
            print(e)
    return out


if __name__ == '__main__':
    """
     Main Executable to process raw Images to dredFISH data
    """
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)
    config.parameters['nucstain_acq']
    if args.fishdata == 'fishdata':
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
    completion_array = np.array([False for i in sections])
    while np.sum(completion_array==False)>0:
        for idx,section in enumerate(sections):
            self = Registration_Class(fishdata_path,metadata_path,section,cword_config)
            self.update_user(str(np.sum(completion_array==False))+ ' Unfinished Sections')
            self.update_user('Processing Section '+section)
            self.run()
            completion_array[idx] = True
        if np.sum(completion_array==False)>0:
            time.sleep(60)
