#!/usr/bin/env python
import logging
import numpy as np
import os
from datetime import datetime
from dredFISH.Utils import fileu
import time
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
import time

import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from scipy.interpolate import Rbf
import math
import time
import anndata

class Registration_Class(object):
    def __init__(self, XYZC, 
                 registration_path,
                 section,
                 verbose=True):
        self.completed = False
        self.section = str(section)
        self.verbose=verbose
        self.window = 0.1
        self.overwrite = False

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=20)
        self.log = logging.getLogger("Registration")
        """ Convert to Checks not making directories These things should already exist """
        self.ref_XYZC = None
        self.XYZC = XYZC
        self.path = registration_path
        if not os.path.exists(self.path):
            self.update_user(self.path,level=30)
            self.update_user('No Registration Path Found',level=30)
            os.mkdir(self.path)
        self.path = os.path.join(self.path,self.section)
        if not os.path.exists(self.path):
            self.update_user(self.path,level=30)
            self.update_user('No Section Path Found',level=30)
            os.mkdir(self.path)

        fileu_config = fileu.interact_config(self.path,return_config=True)
        if not 'version' in fileu_config.keys():
            fileu_config = fileu.interact_config(self.path,key='version',data=2,return_config=True)

        logging.basicConfig(
                    filename=os.path.join(self.path,'registration_log.txt'),filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',level=20, force=True)
        self.log = logging.getLogger("Registration")

    def run(self):
        plt.close('all')
        start = time.time()
        self.load_data()
        """ check Config & Model"""
        fit = True
        self.config = self.load(channel='Registration_Parameters',file_type='Config')
        self.X_model = self.load(channel='X',file_type='Model')
        self.Y_model = self.load(channel='Y',file_type='Model')
        self.Z_model = self.load(channel='Z',file_type='Model')
        if self.overwrite:
            fit = False
        if isinstance(self.config,type(None)):
            fit = False
        if isinstance(self.X_model,type(None)):
            fit = False
        if isinstance(self.Y_model,type(None)):
            fit = False
        if isinstance(self.Z_model,type(None)):
            fit = False
        if not fit:
            """ Fit Model"""
            self.fit()
        self.update_user(f"Total Run Time : {str(time.time()-start)} seconds")
        return self.non_rigid_transformation()
    
    def rigid_transformation(self):
        rigid_transformed_XYZC = self.XYZC.copy()
        """ Center Around Chosen Coordinate"""
        rigid_transformed_XYZC['ccf_y'] = rigid_transformed_XYZC['ccf_y']-self.config['center'][1]
        rigid_transformed_XYZC['ccf_z'] = rigid_transformed_XYZC['ccf_z']-self.config['center'][0]
        """ Rotate By Chosen Angle"""
        rotated_XY = rotate_points(rigid_transformed_XYZC['ccf_z'],rigid_transformed_XYZC['ccf_y'],self.config['angle'])
        rigid_transformed_XYZC['ccf_y'] = rotated_XY[:,1]
        rigid_transformed_XYZC['ccf_z'] = rotated_XY[:,0]
        """ Scale By Choosen Scale"""
        rigid_transformed_XYZC['ccf_y'] = rigid_transformed_XYZC['ccf_y'] / self.config['scale']
        rigid_transformed_XYZC['ccf_z'] = rigid_transformed_XYZC['ccf_z'] / self.config['scale']
        """ Center Around Reference"""
        rigid_transformed_XYZC['ccf_y'] = rigid_transformed_XYZC['ccf_y']+self.config['reference_center'][1]
        rigid_transformed_XYZC['ccf_z'] = rigid_transformed_XYZC['ccf_z']+self.config['reference_center'][0]
        return rigid_transformed_XYZC
    
    def non_rigid_transformation(self):
        self.config = self.load(channel='Registration_Parameters',file_type='Config')
        self.X_model = self.load(channel='X',file_type='Model')
        self.Y_model = self.load(channel='Y',file_type='Model')
        self.Z_model = self.load(channel='Z',file_type='Model')
        """ Center Rotate and Scale"""
        rigid_transformed_XYZC = self.rigid_transformation()
        non_rigid_transformed_XYZC = rigid_transformed_XYZC.copy()
        """ Transform Y """
        non_rigid_transformed_XYZC['ccf_z'] = self.Z_model(rigid_transformed_XYZC['ccf_y'],rigid_transformed_XYZC['ccf_z'])
        """ Transform Z """
        non_rigid_transformed_XYZC['ccf_y'] = self.Y_model(rigid_transformed_XYZC['ccf_y'],rigid_transformed_XYZC['ccf_z'])
        """ Transform X """
        Y = non_rigid_transformed_XYZC['ccf_y']
        Z = non_rigid_transformed_XYZC['ccf_z']
        design_matrix = np.c_[Y,Z]
        non_rigid_transformed_XYZC['ccf_x'] = self.X_model.predict(design_matrix)#[0]
        
        return non_rigid_transformed_XYZC

    def fit(self):
        self.load_reference_data()
        self.set_rigid()
        self.fit_X_model()
        self.fit_YZ_models()
        self.view_transformation()

    def set_rigid(self):
        self.update_user(f"Setting Rigid Parameters")
        """ Use these to decide center and angle for rotation"""
        if isinstance(self.config,type(None))|self.overwrite:
            plt.close('all')
            def set_registration_points(X,Y,C):
                def onpick(event):
                    xmouse = event.mouseevent.xdata
                    ymouse = event.mouseevent.ydata
                    labels = ['Bottom','Top','Finished']
                    if len(points_cor_x)<len(labels)-1:
                        points_cor_x.append(xmouse)
                        points_cor_y.append(ymouse)
                        axs[0].set_title(f" Set {labels[len(points_cor_x)]}")
                        axs[0].scatter(xmouse, ymouse , marker = '.', color = 'k', s = 500)
                        plt.draw()
                    else:
                        plt.close('all')

                points_cor_x = []
                points_cor_y = []

                fig,axs = plt.subplots(1,1,figsize=[5,5])
                fig.suptitle(f"Set Top and Bottom")
                # axs = axs.ravel()
                axs = [axs]
                x = np.linspace(X.min(),X.max(),100000)
                y = np.linspace(Y.min(),Y.max(),100000)
                np.random.shuffle(x)
                np.random.shuffle(y)
                axs[0].set_title('Pick Bottom First')
                axs[0].scatter(x,y,s=25,c='w',picker=True)
                idxs = np.random.choice(np.array(range(X.shape[0])),100000)
                axs[0].scatter(X[idxs],Y[idxs],s=5,c=C[idxs],cmap='jet')
                axs[0].grid()
                fig.canvas.mpl_connect('pick_event', onpick)
                plt.show()
                bottom = np.array([points_cor_x[0],points_cor_y[0]])
                top = np.array([points_cor_x[1],points_cor_y[1]])
                center = (bottom+top)/2

                bottom_vector = bottom - center
                angle = np.arctan2(bottom_vector[0], bottom_vector[1]) * 180 / np.pi
                scale = 1000
                self.update_user(f" Center:{center} Angle {angle} Scale {scale}")
                return center,angle,scale


            self.update_user(f"The purpose of this action is to rotate and center the section")
            self.update_user(f"First you will click on the bottom of the tissue at the midline")
            self.update_user(f"Then you will click of the top of the tissue at the midline")
            self.update_user(f"Lastly you can click anywhere else to finish the action")

            X = self.XYZC['ccf_x'].copy()
            Y = self.XYZC['ccf_y'].copy()
            Z = self.XYZC['ccf_z'].copy()
            C = self.XYZC['color'].copy()
            path = self.generate_filename(channel='Rigid Registration',file_type='Figure')
            center,angle,scale = set_registration_points(Z,Y,C)
            self.config = {}
            self.config['center'] = list(center)
            self.config['reference_center'] = [self.ref_XYZC['ccf_z'].mean(),self.ref_XYZC['ccf_y'].mean()]
            self.config['angle'] = angle
            self.config['scale'] = scale
            self.save(self.config,channel='Registration_Parameters',file_type='Config')

            rigid_transformed_XYZC = self.rigid_transformation()
            fig,axs = plt.subplots(1,1,figsize=[5,5])
            plt.suptitle('Rigid Registration')
            # axs = axs.ravel()
            axs = [axs]
            # idxs = np.random.choice(np.array(range(X.shape[0])),100000)
            # axs[0].scatter(Z[idxs],Y[idxs],s=5,c=C[idxs],cmap='jet')
            # axs[0].grid()

            idxs = np.random.choice(rigid_transformed_XYZC.index,100000)
            axs[0].scatter(rigid_transformed_XYZC.loc[idxs,'ccf_z'],rigid_transformed_XYZC.loc[idxs,'ccf_y'],s=5,c=rigid_transformed_XYZC.loc[idxs,'color'],cmap='jet')
            axs[0].grid()

            plt.savefig(path,dpi=200)
            plt.show(block=False)
    
    def fit_X_model(self):
        if isinstance(self.X_model,type(None))|self.overwrite:
            self.update_user(f"Setting Reference Z")
            """ Set Reference Section """

            rigid_transformed_XYZC = self.rigid_transformation()
            completed = False
            while not completed:
                X = []
                Y = []
                Z = []
                self.update_user(f"The purpose of this action is to roughly pick which plane of the reference does our section align with")
                self.update_user(f"To do this you should first have ordered the sections for this animal using ordering.ipynb")
                self.update_user(f"Once you have the order and the rough ccf_x that the section lays")
                self.update_user(f"You can then correct for any uneven sectioning")
                self.update_user(f"This is done by setting the ccf_x for 4 points[Top:Left,Bottom:Left,Top:Right,Bottom:Right]")
                self.update_user(f"If the section is not uneven all 4 points can be given the same ccf_x")
                for z in [-2,2]:
                    for y in [6,3]:
                        x = float(robust_input(f" Enter Desired Section for [{str(z)},{str(y)}]",dtype='float'))
                        X.append(x)
                        Y.append(y)
                        Z.append(z)

                plt.close('all')
                xyz_coordinates = pd.DataFrame()
                xyz_coordinates['x'] = X
                xyz_coordinates['y'] = Y
                xyz_coordinates['z'] = Z
                
                design_matrix = np.c_[Y,Z]
                model = LinearRegression()
                model.fit(design_matrix, X)
                Y = self.ref_XYZC['ccf_y']
                Z = self.ref_XYZC['ccf_z']
                design_matrix = np.c_[Y,Z]
                distance = self.ref_XYZC['ccf_x'] - model.predict(design_matrix)#[0]

                self.ref_XYZC_sample = np.random.choice(self.ref_XYZC[np.abs(distance)<self.window].index,100000)
                fig,axs = plt.subplots(1,2,figsize=[20,10])
                fig.suptitle(': Set Reference Section: ')
                axs = axs.ravel()
                axs[0].scatter(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_z'],self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_y'],c=self.ref_XYZC.loc[self.ref_XYZC_sample,'color'],s=1,cmap='jet')
                sample = np.random.choice(rigid_transformed_XYZC.index,100000)
                axs[1].scatter(rigid_transformed_XYZC.loc[sample,'ccf_z'],rigid_transformed_XYZC.loc[sample,'ccf_y'],s=1,c=rigid_transformed_XYZC.loc[sample,'color'],cmap='jet')
                path = self.generate_filename(channel='Reference_Section',file_type='Figure')
                plt.savefig(path,dpi=200)
                plt.show(block=False)
                print(' ')
                out = str(robust_input("Satisfied? (Y/N): ",dtype='str'))
                if 'y' in out.lower():
                    completed = True
                else:
                    plt.close('all')
            self.X_model = model
            self.save(self.X_model,channel='X',file_type='Model')

    def fit_YZ_models(self):
        plt.close('all')
        if isinstance(self.Y_model,type(None))|self.overwrite|isinstance(self.Z_model,type(None)):
            """ Pair Points between Reference and Data"""
            self.update_user('Setting Registation Points',level=20)

            self.update_user(f"The purpose of this action is to perform a non rigid transformation")
            self.update_user(f"To do this select points on the reference then the measured that you are confident are the same point")
            self.update_user(f"The algorithm will interpolate between points so the more points you have the more accurate it will be")
            self.update_user(f"One Strategy is to allign the outer border then work your way in")
            self.update_user(f"Select atleast 20 points but you may need more if there are any sectioning artifacts")
            plt.close('all')
            def set_registration_points(path,X,Y,C,ref_X,ref_Y,ref_C):
                def onpick(event):
                    xmouse = event.mouseevent.xdata
                    ymouse = event.mouseevent.ydata
                    points_cor_x.append(xmouse)
                    points_cor_y.append(ymouse)
                    if len(points_cor_x)%2!=0:
                        axs[0].set_title('Pick Right Side: '+str(math.ceil(len(points_cor_x)/2)))
                        # axs[0].scatter(xmouse, ymouse , marker = 'x', color = 'k', s = 100, linewidth = 3, edgecolor='w')
                        axs[0].scatter(xmouse, ymouse , marker = '.', color = 'k', s = 500)
                        axs[0].scatter(xmouse, ymouse , marker = '.', color = 'r', s = 300)
                        # axs[0].annotate(str(math.ceil(len(points_cor_x)/2)), xy = [xmouse, ymouse], color = 'k', fontsize = 14, linewidth = 2)
                    else:
                        axs[0].set_title('Pick Left Side: '+str(math.ceil(len(points_cor_x)/2)))
                        axs[0].scatter(points_cor_x[-2], points_cor_y[-2] , marker = '.', color = 'k', s = 500)
                        axs[1].scatter(xmouse, ymouse , marker = '.', color = 'k', s = 300)
                        # axs[1].annotate(str(math.ceil(len(points_cor_x)/2)), xy = [xmouse, ymouse], color = 'k', fontsize = 14, linewidth = 2)
                    plt.draw()

                points_cor_x = []
                points_cor_y = []

                fig,axs = plt.subplots(1,2,figsize=[20,10])
                fig.suptitle('Pick Registration Points')
                axs = axs.ravel()
                x = np.linspace(ref_X.min(),ref_X.max(),100000)
                y = np.linspace(ref_Y.min(),ref_Y.max(),100000)
                np.random.shuffle(x)
                np.random.shuffle(y)
                axs[0].set_title('Pick Left Side First')
                axs[0].scatter(x,y,s=25,c='w',picker=True)
                idxs = np.random.choice(np.array(range(ref_X.shape[0])),100000)
                axs[0].scatter(ref_X[idxs],ref_Y[idxs],s=5,c=ref_C[idxs],cmap='jet')
                axs[0].grid()
                x = np.linspace(X.min(),X.max(),100000)
                y = np.linspace(Y.min(),Y.max(),100000)
                np.random.shuffle(x)
                np.random.shuffle(y)
                axs[1].scatter(x,y,s=25,c='w',picker=True)
                idxs = np.random.choice(np.array(range(X.shape[0])),100000)
                axs[1].scatter(X[idxs],Y[idxs],s=5,c=C[idxs],cmap='jet')
                axs[1].grid()
                fig.canvas.mpl_connect('pick_event', onpick)
                plt.show()


                fig,axs = plt.subplots(1,2,figsize=[20,10])
                fig.suptitle('Registration Points')
                axs = axs.ravel()
                idxs = np.random.choice(np.array(range(ref_X.shape[0])),100000)
                axs[0].scatter(ref_X[idxs],ref_Y[idxs],s=1,c=ref_C[idxs],cmap='jet')
                idxs = np.random.choice(np.array(range(X.shape[0])),100000)
                axs[1].scatter(X[idxs],Y[idxs],s=1,c=C[idxs],cmap='jet')
                x = np.array(points_cor_x)
                y = np.array(points_cor_y)
                for i in range(x.shape[0]):
                    if i%2!=0:
                        axs[1].scatter(x[i], y[i] , marker = 'x', color = 'k', s = 100, linewidth = 2,edgecolor='w')
                        axs[1].annotate(str(math.ceil(i/2)), xy = [x[i], y[i]], color = 'k', fontsize = 14)
                    else:
                        axs[0].scatter(x[i], y[i] , marker = 'x', color = 'k', s = 100, linewidth = 2,edgecolor='w')
                        axs[0].annotate(str(math.ceil(i/2)+1), xy = [x[i], y[i]], color = 'k', fontsize = 14)
                plt.savefig(path,dpi=200)
                plt.show(block=False)

                return points_cor_x,points_cor_y
            Y = self.ref_XYZC['ccf_y']
            Z = self.ref_XYZC['ccf_z']
            design_matrix = np.c_[Y,Z]
            distance = self.ref_XYZC['ccf_x'] - self.X_model.predict(design_matrix)#[0]
            self.ref_XYZC_sample = np.random.choice(self.ref_XYZC[np.abs(distance)<self.window].index,100000)

            ref_X = np.array(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_x'].copy())
            ref_Y = np.array(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_y'].copy())
            ref_Z = np.array(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_z'].copy())
            ref_C = np.array(self.ref_XYZC.loc[self.ref_XYZC_sample,'color'].copy())

            rigid_transformed_XYZC = self.rigid_transformation()

            X = np.array(rigid_transformed_XYZC['ccf_x'].copy())
            Y = np.array(rigid_transformed_XYZC['ccf_y'].copy())
            Z = np.array(rigid_transformed_XYZC['ccf_z'].copy())
            C = np.array(rigid_transformed_XYZC['color'].copy())
            path = self.generate_filename(channel='RegistrationPointPicking',file_type='Figure')
            df_points = self.load(channel='df_points',file_type='matrix')
            if isinstance(df_points,type(None))|self.overwrite:
                points_cor_x,points_cor_y = set_registration_points(path,Z,Y,C,ref_Z,ref_Y,ref_C)

                df_points = pd.DataFrame()
                df_points['fix_z'] = np.array(points_cor_x)[::2]
                df_points['fix_y'] = np.array(points_cor_y)[::2]
                df_points['mov_z'] = np.array(points_cor_x)[1::2]
                df_points['mov_y'] = np.array(points_cor_y)[1::2]
                self.save(df_points,channel='df_points',file_type='matrix')
            self.update_user(df_points)
            """ Fit Model """
            self.Z_model = Rbf(np.array(df_points['mov_y']),np.array(df_points['mov_z']), np.array(df_points['fix_z']))
            self.save(self.Z_model,channel='Z',file_type='Model')

            self.Y_model = Rbf(np.array(df_points['mov_y']),np.array(df_points['mov_z']), df_points['fix_y'])
            self.save(self.Y_model,channel='Y',file_type='Model')


    def view_transformation(self):

        Y = self.ref_XYZC['ccf_y']
        Z = self.ref_XYZC['ccf_z']
        design_matrix = np.c_[Y,Z]
        distance = self.ref_XYZC['ccf_x'] - self.X_model.predict(design_matrix)#[0]
        self.ref_XYZC_sample = np.random.choice(self.ref_XYZC[np.abs(distance)<self.window].index,100000)
        
        self.update_user('Viewing Transformation',level=20)
        df_points = self.load(channel='df_points',file_type='matrix')
        
        plt.figure(figsize=[5,5])
        plt.scatter(df_points['fix_z'],df_points['fix_y'],c='k',s=np.array(df_points.index))
        plt.scatter(df_points['mov_z'],df_points['mov_y'],c='r',s=np.array(df_points.index))
        for i in range(len(df_points)):
            plt.plot([df_points['fix_z'][i], df_points['mov_z'][i]], [df_points['fix_y'][i], df_points['mov_y'][i]], c='b')
        path = self.generate_filename(channel='RegPointsRaw',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

        df_points['pred_z'] = self.Z_model(np.array(df_points['mov_y']),np.array(df_points['mov_z']))
        df_points['pred_y'] = self.Y_model(np.array(df_points['mov_y']),np.array(df_points['mov_z']))
        plt.figure(figsize=[5,5])
        plt.scatter(df_points['fix_z'],df_points['fix_y'],c='k',s=np.array(df_points.index))
        plt.scatter(df_points['pred_z'],df_points['pred_y'],c='r',s=np.array(df_points.index))
        for i in range(len(df_points)):
            plt.plot([df_points['fix_z'][i], df_points['pred_z'][i]], [df_points['fix_y'][i], df_points['pred_y'][i]], c='b')
        path = self.generate_filename(channel='RegPointsAligned',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

        plt.figure(figsize=[5,5])
        plt.scatter(df_points['mov_z'],df_points['mov_y'],c='k',s=np.array(df_points.index))
        plt.scatter(df_points['pred_z'],df_points['pred_y'],c='r',s=np.array(df_points.index))
        for i in range(len(df_points)):
            plt.plot([df_points['mov_z'][i], df_points['pred_z'][i]], [df_points['mov_y'][i], df_points['pred_y'][i]], c='b')
        path = self.generate_filename(channel='RegPointsDistanceMoved',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

        XYZC = self.rigid_transformation()
        fig = plt.figure(figsize=[5,5])
        fig.suptitle('Raw vs Ref')
        plt.scatter(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_z'],self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_y'],s=0.1,c='k',alpha=0.5)
        plt.scatter(XYZC['ccf_z'],XYZC['ccf_y'],s=0.1,c='r',alpha=0.5)
        path = self.generate_filename(channel='RoughAlignment',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

        XYZC = self.non_rigid_transformation()
        fig = plt.figure(figsize=[5,5])
        fig.suptitle('Raw vs Ref')
        plt.scatter(self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_z'],self.ref_XYZC.loc[self.ref_XYZC_sample,'ccf_y'],s=0.1,c='k',alpha=0.5)
        plt.scatter(XYZC['ccf_z'],XYZC['ccf_y'],s=0.1,c='r',alpha=0.5)
        path = self.generate_filename(channel='Aligned',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

        fig = plt.figure(figsize=[5,5])
        fig.suptitle('Distance Moved')

        c = np.sqrt(np.sum((np.array(self.rigid_transformation()[['ccf_z','ccf_y']])-np.array(self.non_rigid_transformation()[['ccf_z','ccf_y']]))**2,1))
        vmin,vmax = np.percentile(c,[1,99])
        c = np.clip(c,vmin,vmax)
        plt.scatter(XYZC['ccf_z'],XYZC['ccf_y'],s=0.1,c=c,cmap='jet')
        plt.colorbar()
        path = self.generate_filename(channel='DistanceMoved',file_type='Figure')
        plt.savefig(path,dpi=200)
        plt.show(block=False)

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

    def check_existance(self,hybe='',channel='',file_type='',model_type='',path=''):
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
        if path=='':
            path=self.path
        return fileu.check_existance(path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_filename(self,hybe='',channel='',file_type='',model_type='',path=''):
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
        if path=='':
            path=self.path
        return fileu.generate_filename(path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def save(self,data,hybe='',channel='',file_type='',model_type='',path=''):
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
        if path=='':
            path=self.path
        fileu.save(data,path=path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,dataset='',section='',logger=self.log)

    def load(self,hybe='',channel='',file_type='anndata',model_type='',path=''):
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
        if path=='':
            path=self.path
        return fileu.load(path=path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

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

        # """ Update to use config file """
        # """ /orangedata/Images2024/Zach/dredFISH/Tree20um37C500M1H3A2SDS24H100ug47C50F18H50F1H2SDS37C1H-FF.A_FF.B_FF.C_PFA.D_PFA.E_PFA.F_2024Feb15/fishdata_2024Feb01/Notebooks/reference_sections.ipynb"""
        # reference_data = pd.read_csv('/orangedata/ExternalData/WMB_Spatial_Data_ZH2024Feb22.csv',index_col=0,low_memory=False)
        # reference_data = reference_data.astype(str)
        # reference_data['ccf_x'] = reference_data['ccf_x'].astype(float)
        # reference_data['ccf_y'] = reference_data['ccf_y'].astype(float)
        # reference_data['ccf_z'] = reference_data['ccf_z'].astype(float)
        if isinstance(self.ref_XYZC,type(None)):
            """ /orangedata/Images2024/Zach/dredFISH/Notebooks/reference_sections_2024Mar06.ipynb"""
            download_base = '/orangedata/ExternalData/Allen_WMB_2024Mar06'
            reference_data = torch.load(os.path.join(download_base,'minimal_spatial_data.pt'))
            reference_data = pd.DataFrame(reference_data.numpy(),columns=['ccf_x','ccf_y','ccf_z','cluster_alias'])
            reference_data['cluster_alias'] = reference_data['cluster_alias'].astype(int)
            negative_reference_data = reference_data.copy()
            negative_reference_data['ccf_z'] = -negative_reference_data['ccf_z']
            reference_data = pd.concat([reference_data,negative_reference_data],ignore_index=True)

            pivot_table = pd.read_csv(os.path.join(download_base,'pivot_table.csv'))

            colormap = dict(zip(pivot_table['cluster_alias'],pivot_table['subclass_color']))
            reference_data['color'] = reference_data['cluster_alias'].map(colormap)
            self.ref_XYZC = reference_data[['ccf_x','ccf_y','ccf_z','color']].copy()
            del reference_data
            self.update_user(self.ref_XYZC.head())

    def load_data(self):
        """ Load processed anndata object from Section_Class """
        self.update_user('Loading Data',level=20)
        if isinstance(self.XYZC,str):
            try:
                self.XYZC = self.load(path=os.path.join(self.XYZC,self.section),file_type='anndata')
            except Exception as e:
                self.update_user(str(e),level=40) 
                self.update_user('Unable to Load Data',level=50) 
        if isinstance(self.XYZC,type(anndata.AnnData())):
            data = self.XYZC.copy()
            self.update_user(data)
            self.XYZC = pd.DataFrame(index=data.obs.index)
            self.XYZC['ccf_x'] = np.ones_like(data.obs['stage_x'])
            self.XYZC['ccf_y'] = np.array(data.obs['stage_y'])
            self.XYZC['ccf_z'] = np.array(data.obs['stage_x'])
            bit = 'RS458122_cy5'
            from dredFISH.Utils import basicu
            X = data.layers['processed_vectors'].copy()
            X = np.log10(np.clip(X,1,None))
            X = basicu.normalize_fishdata_robust_regression(X)
            c = X[:,data.var.index==bit]
            vmin,vmax = np.percentile(c,[5,95])
            c = np.clip(c,vmin,vmax)
            
            self.XYZC['color'] = c#np.array(data.obs['louvain_colors'])
            del data
        if isinstance(self.XYZC,type(None)):
            self.update_user('Data Not Found',level=50) 


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
    out_XY = np.zeros([len(X),2])
    rotated_x = []
    rotated_y = []
    for i,(x, y) in enumerate(zip(X, Y)):
        new_x = x * cos_theta - y * sin_theta
        new_y = x * sin_theta + y * cos_theta
        out_XY[i,0] = new_x
        out_XY[i,1] = new_y
        # rotated_x.append(new_x)
        # rotated_y.append(new_y)

    return out_XY#np.dstack([np.array(rotated_x), np.array(rotated_y)])

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