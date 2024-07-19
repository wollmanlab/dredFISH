import numpy as np
import logging
import torch
# Basic parameters of imaging
bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
            ('RS0175_cy5', 'hybe2', 'FarRed'), #readout3
            ('RS0332_cy5', 'hybe4', 'FarRed'), #readout6
            ('RSN2336.0_cy5', 'hybe5', 'FarRed'), #readout8
            ('RSN1807.0_cy5', 'hybe6', 'FarRed'), #readout9
            ('RS0384_cy5', 'hybe7', 'FarRed'), #readout10
            ('RS0406_cy5', 'hybe8', 'FarRed'), #readout11
            ('RS0451_cy5', 'hybe9', 'FarRed'), #readout12
            ('RS0548_cy5', 'hybe10', 'FarRed'), #readout14
            ('RS64.0_cy5', 'hybe11', 'FarRed'), #readout15
            ('RSN4287.0_cy5', 'hybe12', 'FarRed'), #readout16
            ('RSN1252.0_cy5', 'hybe13', 'FarRed'), #readout17
            ('RSN9535.0_cy5', 'hybe14', 'FarRed'), #readout18
            ('RS740.0_cy5', 'hybe17', 'FarRed'), #readout23
            ('RS810.0_cy5', 'hybe18', 'FarRed'), #readout24
            ('RS458122_cy5', 'hybe16', 'FarRed'), #readout25
            ('RS0083_SS_Cy5', 'hybe3', 'FarRed'), # readout26
            ('RS0255_SS_Cy5', 'hybe15', 'FarRed')] #readout27

weights_path = '/greendata/binfo/Probe_Sets/WeightMatrices/DPNMF_tree_clipped_weights_Nov7_2022.csv'

nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""

parameters['fishdata']='Processing_2024May28'
parameters['hard_overwrite'] = True
parameters['use_scratch'] = False 


# Orange
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['jitter_channel'] = ''
parameters['jitter_correction'] = True

# Purple
# parameters['pixel_size'] =0.409# 0.490#0.327#0.490 # um 490 or 330
# parameters['jitter_channel'] = 'DeepBlue'


parameters['bin'] = 2
# parameters['process_pixel_size'] = parameters['pixel_size']*parameters['bin']
parameters['stitch_raw'] = False
parameters['stitch_rotate'] = 0# NEW0 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['register_stitch_reference'] = True
parameters['segment_gpu'] = False
parameters['max_registration_shift'] = 200

parameters['QC_pixel_size'] = 2 # um
parameters['diameter'] = 8 #15 # um
# parameters['segment_diameter'] = parameters['diameter']/parameters['process_pixel_size']
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'hybe18'
parameters['total_channel'] = 'FarRed'
parameters['total_acq'] = 'all_max' #'hybe25'
parameters['outpath'] = '/greendata/GeneralStorage/Data/dredFISH/' #"Path to save data"
# parameters['nuclei_size_threshold'] = parameters['segment_diameter']*2
# parameters['ratio'] = parameters['process_pixel_size']/parameters['QC_pixel_size']
parameters['n_pixels']=[2448, 2048]
# ratio = parameters['pixel_size']/parameters['process_pixel_size']
# parameters['n_pixels']=[int(float(i)*ratio) for i in parameters['n_pixels']]
# parameters['border'] = int(np.min(parameters['n_pixels']))

parameters['highpass_function'] = 'rolling_ball'#'gaussian_robust[1,60]'#'spline_min_robust_smooth'#'spline_min_smooth'#'spline_robust_min'#'downsample_quantile_0.1' _smooth
parameters['highpass_sigma'] = 25
parameters['highpass_smooth_function'] = 'median'
parameters['highpass_smooth'] = 3

parameters['strip'] = True
parameters['model_types'] = ['total']
parameters['dapi_thresh'] = 10
parameters['background_estimate_iters'] = 0
parameters['stain_correction'] = False
parameters['stain_correction_downsample'] = 10
parameters['stain_correction_kernel'] = 1000
parameters['overlap'] = 0.02 # 2% overlap
# parameters['segment_min_size'] = parameters['segment_diameter']*10
parameters['fileu_version'] = 2
parameters['overlap_correction'] = False #Problematic

parameters['ncpu'] = 5
parameters['set_min_zero'] = False#True # Post Strip bkg subtract ****
parameters['metric'] = 'median'

parameters['microscope_parameters'] = 'microscope_parameters'
parameters['imaging_batch'] = 'hybe'

parameters['post_strip_process'] = False

parameters['acq_FF'] = False
parameters['acq_constant'] = False

parameters['use_FF'] = True
parameters['use_constant'] = True#False

parameters['fit_FF'] = False
parameters['fit_constant'] = False

parameters['clip_FF'] = False
parameters['clip_constant'] = False

parameters['FF_n_cpu'] = 1
parameters['constant_poly_degrees'] = 5
parameters['FF_poly_degrees'] = 5
parameters['smooth_FF'] = True
parameters['smooth_constant'] = True

parameters['constant_smooth_function'] = 'spline_min'
parameters['constant_smooth_sigma'] = '32|34'
parameters['FF_smooth_function'] = 'spline'#'spline_robust'
parameters['FF_smooth_sigma'] = '32|34' # '64|72' # '32|36' '16|18' '16|17' 
# how many pixels to bin for spline 'value1|value2'
# For binsize=2 [i  for i in range(1,1000) if ((2048/binsize)/i).is_integer()]
# #'[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]|[1, 2, 3, 4, 6, 8, 9, 12, 17, 18, 24, 34, 36, 51, 68, 72, 102, 136, 153, 204, 306, 408, 612]'


parameters['post_processing_constant'] = False

parameters['process_img_before_FF'] = False
parameters['debug'] = False

parameters['config_overwrite'] = True
parameters['overwrite'] = False #False
parameters['segment_overwrite'] = False #False
parameters['vector_overwrite'] = False #False

parameters['overwrite_report']= False
parameters['overwrite_louvain'] = False

parameters['scratch_path_base'] = '/scratchdata2/Processing_tmp'

parameters['bitmap'] = bitmap