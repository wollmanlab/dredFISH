import numpy as np
import logging
import torch
# Basic parameters of imaging
bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
            ('RS0175_cy5', 'hybe3', 'FarRed'),
            ('RS0332_cy5', 'hybe6', 'FarRed'),
            ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
            ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
            ('RS0384_cy5', 'hybe10', 'FarRed'),
            ('RS0406_cy5', 'hybe11', 'FarRed'),
            ('RS0451_cy5', 'hybe12', 'FarRed'),
            ('RS0548_cy5', 'hybe14', 'FarRed'),
            ('RS64.0_cy5', 'hybe15', 'FarRed'),
            ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
            ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
            ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
            ('RS740.0_cy5', 'hybe23', 'FarRed'),
            ('RS810.0_cy5', 'hybe24', 'FarRed'),
            ('RS458122_cy5', 'hybe21', 'FarRed'), # (’R25’,’RS458122_cy5’, 'hybe25', 'FarRed',’AACTCCTTATCACCCTACTC’)
            ('RS0083_SS_Cy5', 'hybe5', 'FarRed'), # (’R26’,’RS0083_SS_Cy5’, ’hybe5/bDNA’, ‘FarRed’,’ACACTACCACCATTTCCTAT’)
            ('RS0255_SS_Cy5', 'hybe20', 'FarRed')] # R20
# All 24
# bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
#             ('RS0109_cy5', 'hybe2', 'FarRed'),
#             ('RS0175_cy5', 'hybe3', 'FarRed'),
#             ('RS0237_cy5', 'hybe4', 'FarRed'),
#             ('RS0332_cy5', 'hybe6', 'FarRed'),
#             ('RSN9927.0_cy5', 'hybe7', 'FarRed'),
#             ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
#             ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
#             ('RS0384_cy5', 'hybe10', 'FarRed'),
#             ('RS0406_cy5', 'hybe11', 'FarRed'),
#             ('RS0451_cy5', 'hybe12', 'FarRed'),
#             ('RS0468_cy5', 'hybe13', 'FarRed'),
#             ('RS0548_cy5', 'hybe14', 'FarRed'),
#             ('RS64.0_cy5', 'hybe15', 'FarRed'),
#             ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
#             ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
#             ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
#             ('RS156.0_cy5', 'hybe19', 'FarRed'),
#             ('RS643.0_cy5', 'hybe22', 'FarRed'),
#             ('RS740.0_cy5', 'hybe23', 'FarRed'),
#             ('RS810.0_cy5', 'hybe24', 'FarRed'),
#             ('RS458122_cy5', 'hybe21', 'FarRed'), # (’R25’,’RS458122_cy5’, 'hybe25', 'FarRed',’AACTCCTTATCACCCTACTC’)
#             ('RS0083_SS_Cy5', 'hybe5', 'FarRed'), # (’R26’,’RS0083_SS_Cy5’, ’hybe5/bDNA’, ‘FarRed’,’ACACTACCACCATTTCCTAT’)
#             ('RS0255_SS_Cy5', 'hybe20', 'FarRed')] # R20

# bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
#             ('RS0109_cy5', 'hybe2', 'FarRed'),
#             ('RS0175_cy5', 'hybe3', 'FarRed'),
#             ('RS0237_cy5', 'hybe4', 'FarRed'),
#             ('RS0332_cy5', 'hybe6', 'FarRed'),
#             ('RSN9927.0_cy5', 'hybe7', 'FarRed'),
#             ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
#             ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
#             ('RS0384_cy5', 'hybe10', 'FarRed'),
#             ('RS0406_cy5', 'hybe11', 'FarRed'),
#             ('RS0451_cy5', 'hybe12', 'FarRed'),
#             ('RS0468_cy5', 'hybe13', 'FarRed'),
#             ('RS0548_cy5', 'hybe14', 'FarRed'),
#             ('RS64.0_cy5', 'hybe15', 'FarRed'),
#             ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
#             ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
#             ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
#             ('RS156.0_cy5', 'hybe19', 'FarRed'),
#             ('RS643.0_cy5', 'hybe22', 'FarRed'),
#             ('RS740.0_cy5', 'hybe23', 'FarRed'),
#             ('RS810.0_cy5', 'hybe24', 'FarRed'),
#             ('RS458122_cy5', 'hybe21', 'FarRed'), # (’R25’,’RS458122_cy5’, 'hybe25', 'FarRed',’AACTCCTTATCACCCTACTC’)
#             ('RS0083_SS_Cy5', 'hybe5', 'FarRed'), # (’R26’,’RS0083_SS_Cy5’, ’hybe5/bDNA’, ‘FarRed’,’ACACTACCACCATTTCCTAT’)
#             ('RS0255_SS_Cy5', 'hybe20', 'FarRed'), # (’R27’,’RS0255_SS_Cy5’, ’hybe20/DNA’, ‘FarRed’,’TCCTATTCTCAACCTAACCT’)
#             ('Nonspecific_Readout','hybe27','FarRed'), # (’R5’,'RS0307_cy5', 'hybe5', 'FarRed',’TATCCTTCAATCCCTCCACA’)
#             ('Nonspecific_Encoding','hybe26','FarRed'), #(’R28’,’RS1261_SS_Cy5’, ’hybe21/bDNA’, ‘FarRed’,’ACACCATTTATCCACTCCTC’) [Non Specific Encoding Probe ] 
#             ('Housekeeping', 'hybe25', 'FarRed'), #R29
#             ('IEG', 'hybe28', 'FarRed')] # R20
# """2023Dec08"""
# bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
#             ('RS0109_cy5', 'hybe2', 'FarRed'),
#             ('RS0175_cy5', 'hybe3', 'FarRed'),
#             ('RS0237_cy5', 'hybe4', 'FarRed'),
#             ('RS0332_cy5', 'hybe6', 'FarRed'),
#             ('RSN9927.0_cy5', 'hybe7', 'FarRed'),
#             ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
#             ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
#             ('RS0384_cy5', 'hybe10', 'FarRed'),
#             ('RS0406_cy5', 'hybe11', 'FarRed'),
#             ('RS0451_cy5', 'hybe12', 'FarRed'),
#             ('RS0468_cy5', 'hybe13', 'FarRed'),
#             ('RS0548_cy5', 'hybe14', 'FarRed'),
#             ('RS64.0_cy5', 'hybe15', 'FarRed'),
#             ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
#             ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
#             ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
#             ('RS156.0_cy5', 'hybe19', 'FarRed'),
#             ('RS643.0_cy5', 'hybe22', 'FarRed'),
#             ('RS740.0_cy5', 'hybe23', 'FarRed'),
#             ('RS810.0_cy5', 'hybe24', 'FarRed'),
#             ('RS458122_cy5', 'hybe25', 'FarRed'),
#             ('RS0083_SS_Cy5', 'hybe5', 'FarRed'),
#             ('RS0255_SS_Cy5', 'hybe20', 'FarRed'),
#             ('Nonspecific_Readout','hybe27','FarRed'),
#             ('Nonspecific_Encoding','hybe26','FarRed'),
#             ('PolyT', 'hybe21', 'FarRed')]

weights_path = '/greendata/binfo/Probe_Sets/WeightMatrices/DPNMF_tree_clipped_weights_Nov7_2022.csv'

encoding_weight_bias = { 'RS0095_cy5': 1061,
                         'RS0109_cy5': 468,
                         'RS0175_cy5': 356,
                         'RS0237_cy5': 374,
                         'RS0332_cy5': 752,
                         'RSN9927.0_cy5': 920,
                         'RSN2336.0_cy5': 2296,
                         'RSN1807.0_cy5': 1112,
                         'RS0384_cy5': 636,
                         'RS0406_cy5': 555,
                         'RS0451_cy5': 402,
                         'RS0468_cy5': 183,
                         'RS0548_cy5': 2023,
                         'RS64.0_cy5': 458,
                         'RSN4287.0_cy5': 3560,
                         'RSN1252.0_cy5': 475,
                         'RSN9535.0_cy5': 4576,
                         'RS156.0_cy5': 2148,
                         'RS643.0_cy5': 262,
                         'RS740.0_cy5': 196,
                         'RS810.0_cy5': 3555,
                         'RS458122_cy5': 1239,
                         'RS0083_SS_Cy5': 4947,
                         'RS0255_SS_Cy5': 3365}
"""
                            {'hybe1': 1061,
                             'hybe2': 468,
                             'hybe3': 356,
                             'hybe4': 374,
                             'hybe6': 752,
                             'hybe7': 920,
                             'hybe8': 2296,
                             'hybe9': 1112,
                             'hybe10': 636,
                             'hybe11': 555,
                             'hybe12': 402,
                             'hybe13': 183,
                             'hybe14': 2023,
                             'hybe15': 458,
                             'hybe16': 3560,
                             'hybe17': 475,
                             'hybe18': 4576,
                             'hybe19': 2148,
                             'hybe22': 262,
                             'hybe23': 196,
                             'hybe24': 3555,
                             'hybe21': 1239,
                             'hybe5': 4947,
                             'hybe20': 3365}

"""
smartseq_avg_signal = {'RS0095_cy5': 83897.30418786337,
 'RS0109_cy5': 178720.71219794694,
 'RS0175_cy5': 96528.39166969477,
 'RS0237_cy5': 72256.77433457486,
 'RS0332_cy5': 425101.6022892442,
 'RSN9927.0_cy5': 42991.88600336119,
 'RSN2336.0_cy5': 1493763.1729651163,
 'RSN1807.0_cy5': 790618.4193313953,
 'RS0384_cy5': 78371.6824582122,
 'RS0406_cy5': 54350.66128043241,
 'RS0451_cy5': 10816385.494186046,
 'RS0468_cy5': 91950.32360555959,
 'RS0548_cy5': 293490.35646802327,
 'RS64.0_cy5': 255177.45943859013,
 'RSN4287.0_cy5': 2020901.6235465116,
 'RSN1252.0_cy5': 144122.21402616278,
 'RSN9535.0_cy5': 1161627.5065406978,
 'RS156.0_cy5': 229653.3105014535,
 'RS643.0_cy5': 184689.00767623546,
 'RS740.0_cy5': 295275.06025163515,
 'RS810.0_cy5': 950810.4367732558,
 'RS458122_cy5': 303700.3316678779,
 'RS0083_SS_Cy5': 1258167.4207848837,
 'RS0255_SS_Cy5': 2193261.4854651163}

nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""

parameters['fishdata']='Processing_2024May28'
parameters['hard_overwrite'] = False
parameters['use_scratch'] = False 


# Orange
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['jitter_channel'] = ''
parameters['jitter_correction'] = True

# Purple
# parameters['pixel_size'] =0.409# 0.490#0.327#0.490 # um 490 or 330
# parameters['jitter_channel'] = 'DeepBlue'


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
parameters['diameter'] = 8 #15 # um #used for segmenting cells in um
# parameters['segment_diameter'] = parameters['diameter']/parameters['process_pixel_size']
parameters['nucstain_channel'] = 'DeepBlue' #Reference Channel
parameters['nucstain_acq'] = 'hybe24' # reference hybe
parameters['total_channel'] = 'FarRed' # Signal channel for segmentation
parameters['total_acq'] = 'all_max' #'hybe25' # Which acq to segment on 
parameters['outpath'] = '/greendata/GeneralStorage/Data/dredFISH/' #"Path to save data" #Unused 
# parameters['nuclei_size_threshold'] = parameters['segment_diameter']*2
# parameters['ratio'] = parameters['process_pixel_size']/parameters['QC_pixel_size']
parameters['n_pixels']=[2448, 2048] #size of image in pixels
# ratio = parameters['pixel_size']/parameters['process_pixel_size']
# parameters['n_pixels']=[int(float(i)*ratio) for i in parameters['n_pixels']]
# parameters['border'] = int(np.min(parameters['n_pixels']))

# Background subtraction parameters
parameters['highpass_function'] = 'rolling_ball'#'gaussian_robust[1,60]'#'spline_min_robust_smooth'#'spline_min_smooth'#'spline_robust_min'#'downsample_quantile_0.1' _smooth
parameters['highpass_sigma'] = 25 #binned pixels
parameters['highpass_smooth_function'] = 'median'
parameters['highpass_smooth'] = 3 #binned pixels

parameters['strip'] = True # use strip image as additional backround 
parameters['model_types'] = ['total'] # total nuclei or cytoplasm segmentation options 
parameters['dapi_thresh'] = 200 #minimum dapi signal for segmentation
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