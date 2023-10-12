import numpy as np
import logging
# Basic parameters of imaging
bitmap = [('RS0332_cy5', 'hybe6', 'FarRed'),
            ('RS0451_cy5', 'hybe12', 'FarRed'),
            ('RS740.0_cy5', 'hybe23', 'FarRed'),
            ('RS458122_cy5', 'hybe21', 'FarRed'),
            ('Nonspecific_Readout','hybe27','FarRed'),
            ('Nonspecific_Encoding','hybe26','FarRed'),
            ('PolyT', 'hybe25', 'FarRed')]

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

nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['stitch_rotate'] = 0# NEW0 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['register_stitch_reference'] = True

parameters['segment_gpu'] = False
parameters['fishdata']='fishdata_2023Oct12'
parameters['QC_pixel_size'] = 2 # um
parameters['diameter'] = 8 #15 # um
parameters['segment_diameter'] = parameters['diameter']/parameters['pixel_size']
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'hybe25'
parameters['total_channel'] = 'FarRed'
parameters['total_acq'] = 'hybe25' #'hybe25'
parameters['overwrite'] = False #False
parameters['segment_overwrite'] = False #False
parameters['vector_overwrite'] = False #False
parameters['outpath'] = '/greendata/GeneralStorage/Data/dredFISH/' #"Path to save data"
parameters['nuclei_size_threshold'] = parameters['segment_diameter']*2
parameters['ratio'] = parameters['pixel_size']/parameters['QC_pixel_size']
parameters['n_pixels']=np.array([2448, 2048])
parameters['border'] = 1000
parameters['highpass_sigma'] = 25
parameters['highpass_function'] = 'rolling_ball'
parameters['highpass_smooth_function'] = 'rolling_ball'
parameters['strip'] = True
parameters['highpass_smooth'] = 1
parameters['model_types'] = ['total']
parameters['dapi_thresh'] = 100
parameters['processing_log_level'] = logging.DEBUG
parameters['background_estimate_iters'] = 0
parameters['stain_correction'] = False
parameters['stain_correction_downsample'] = 10
parameters['stain_correction_kernel'] = 1000