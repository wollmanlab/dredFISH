import numpy as np
import logging
import torch
# Basic parameters of imaging
bitmap = [('RS810.0_cy5', 'hybe24', 'FarRed'),
            ('B_amyloid', 'hybe2', 'FarRed'),
            ('B_amyloid_HCR', 'hybe22', 'FarRed')] # R20


nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""

parameters['fishdata']='Aggregates_2024Nov12' # use as version control
parameters['hard_overwrite'] = False
parameters['use_scratch'] = False 


# Orange
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['jitter_channel'] = ''
parameters['jitter_correction'] = True

# Purple
# parameters['pixel_size'] =0.409# 0.490#0.327#0.490 # um 490 or 330
# parameters['jitter_channel'] = 'DeepBlue'

parameters['segment_thresh'] = 500

parameters['bin'] = 2 # how many pixels to bin to save computational time 2x2 
# parameters['process_pixel_size'] = parameters['pixel_size']*parameters['bin']
parameters['stitch_raw'] = False # should you stitch an unprocessed image #debugging
parameters['stitch_rotate'] = 0# NEW0 # NEW 0 #scope specific rotation but all scopes synced
parameters['stitch_flipud'] = False# NEW False #scope specific flip up down but all scopes synced
parameters['stitch_fliplr'] = True# NEW True #scope specific flip left right but all scopes synced
parameters['register_stitch_reference'] = True # should you register the stitched image to the reference hybe
parameters['segment_gpu'] = False #Broken unless you have cuda tensor set up
parameters['max_registration_shift'] = 200 # binned pixels

parameters['QC_pixel_size'] = 2 # um # size of pixel for saved tifs
parameters['diameter'] = 15 #15 # um #used for segmenting cells in um
# parameters['segment_diameter'] = parameters['diameter']/parameters['process_pixel_size']
parameters['nucstain_channel'] = 'DeepBlue' #Reference Channel
parameters['nucstain_acq'] = 'hybe24' # reference hybe
parameters['total_channel'] = 'FarRed' # Signal channel for segmentation
parameters['total_acq'] = 'hybe2' #'hybe25' # Which acq to segment on 
parameters['outpath'] = '/greendata/GeneralStorage/Data/dredFISH/' #"Path to save data" #Unused 
# parameters['nuclei_size_threshold'] = parameters['segment_diameter']*2
# parameters['ratio'] = parameters['process_pixel_size']/parameters['QC_pixel_size']
parameters['n_pixels']=[2448, 2048] #size of image in pixels
# ratio = parameters['pixel_size']/parameters['process_pixel_size']
# parameters['n_pixels']=[int(float(i)*ratio) for i in parameters['n_pixels']]
# parameters['border'] = int(np.min(parameters['n_pixels']))

# Background subtraction parameters
parameters['highpass_function'] = 'rolling_ball'#'gaussian_robust[1,60]'#'spline_min_robust_smooth'#'spline_min_smooth'#'spline_robust_min'#'downsample_quantile_0.1' _smooth
parameters['highpass_sigma'] = 100 #binned pixels
parameters['highpass_smooth_function'] = 'median'
parameters['highpass_smooth'] = 3 #binned pixels

parameters['strip'] = True # use strip image as additional backround 
parameters['model_types'] = ['threshold'] # total nuclei or cytoplasm segmentation options 
parameters['dapi_thresh'] = 0 #minimum dapi signal for segmentation
parameters['background_estimate_iters'] = 0  #stimate backgound after stitching iteratively
parameters['stain_correction'] = False # Unused
parameters['stain_correction_downsample'] = 10# Unused
parameters['stain_correction_kernel'] = 1000# Unused
parameters['overlap'] = 0.02 # 2% overlap
# parameters['segment_min_size'] = parameters['segment_diameter']*10
parameters['fileu_version'] = 2 # Version of fileu to use
parameters['overlap_correction'] = False #Problematic # use overlap to correct for constants in the image

parameters['ncpu'] = 5 # number of threads to use
parameters['set_min_zero'] = False#True # Post Strip bkg subtract ****
parameters['metric'] = 'median' # what value to pull from the stiched image for each cell 

parameters['microscope_parameters'] = 'microscope_parameters' # directory name for image parameters FF and constant
parameters['imaging_batch'] = 'hybe' # which FF and constants to average 'acq' 'dataset' 'hybe' #maybe brightness depentant

parameters['post_strip_process'] = False # process after subtracting strip

parameters['acq_FF'] = False # Unused
parameters['acq_constant'] = False #unused

parameters['use_FF'] = True # should you correct for Flat FIeld artifacts
parameters['use_constant'] = True#False # should you correct for constant artifacts

# unused since FF was moved to imageu
parameters['fit_FF'] = False # fit a polynomial to the FF to smooth
parameters['fit_constant'] = False # fit a polynomial to the constant to smooth

parameters['clip_FF'] = False # clip the FF to remove outliers
parameters['clip_constant'] = False # clip the constant to remove outliers

parameters['FF_n_cpu'] = 1  # number of threads to use for FF correction
parameters['constant_poly_degrees'] = 5 # degree of polynomial to fit to constant
parameters['FF_poly_degrees'] = 5 # degree of polynomial to fit to FF
parameters['smooth_FF'] = True # smooth the FF with gaussian filter
parameters['smooth_constant'] = True # smooth the constant with gaussian filter

parameters['constant_smooth_function'] = 'spline_min' 
parameters['constant_smooth_sigma'] = '32|34'
parameters['FF_smooth_function'] = 'spline'#'spline_robust'
parameters['FF_smooth_sigma'] = '32|34' # '64|72' # '32|36' '16|18' '16|17' 
# how many pixels to bin for spline 'value1|value2'
# For binsize=2 [i  for i in range(1,1000) if ((2048/binsize)/i).is_integer()]
# #'[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]|[1, 2, 3, 4, 6, 8, 9, 12, 17, 18, 24, 34, 36, 51, 68, 72, 102, 136, 153, 204, 306, 408, 612]'


###

parameters['post_processing_constant'] = False # calculate and remove residual constant in each image after processing

parameters['process_img_before_FF'] = False # process image before FF correction # dont use
parameters['debug'] = False # more figures printed 

parameters['config_overwrite'] = True # should you overwrite your config file saved in the fishdata path
parameters['overwrite'] = False #False # should you overwrite stitching images
parameters['segment_overwrite'] = False #False # should you overwrite segmentation Masks
parameters['vector_overwrite'] = False #False # should you overwrite pulled vectors
parameters['delete_temp_files'] = True #False # should you delete temporary files to reduce space on server
parameters['overwrite_report']= False # report figures overwrite
parameters['overwrite_louvain'] = False # overwrite louvain unsupervised clustering

parameters['scratch_path_base'] = '/scratchdata2/Processing_tmp' # where to save temporary files

parameters['bitmap'] = bitmap # dont change this its for saving and record keeping