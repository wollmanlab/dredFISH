import numpy as np
import logging
# Basic parameters of imaging
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
#             ('RS1261_SS_Cy5', 'hybe21', 'FarRed'),
#             ('PolyT', 'hybe21', 'FarRed')]

bitmap = [('RS0095_cy5', 'hybe1', 'FarRed'),
            ('RS0109_cy5', 'hybe2', 'FarRed'),
            ('RS0175_cy5', 'hybe3', 'FarRed'),
            ('RS0237_cy5', 'hybe4', 'FarRed'),
            ('RS0332_cy5', 'hybe6', 'FarRed'),
            ('RSN9927.0_cy5', 'hybe7', 'FarRed'),
            ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
            ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
            ('RS0384_cy5', 'hybe10', 'FarRed'),
            ('RS0406_cy5', 'hybe11', 'FarRed'),
            ('RS0451_cy5', 'hybe12', 'FarRed'),
            ('RS0468_cy5', 'hybe13', 'FarRed'),
            ('RS0548_cy5', 'hybe14', 'FarRed'),
            ('RS64.0_cy5', 'hybe15', 'FarRed'),
            ('RSN4287.0_cy5', 'hybe16', 'FarRed'),
            ('RSN1252.0_cy5', 'hybe17', 'FarRed'),
            ('RSN9535.0_cy5', 'hybe18', 'FarRed'),
            ('RS156.0_cy5', 'hybe19', 'FarRed'),
            ('RS643.0_cy5', 'hybe22', 'FarRed'),
            ('RS740.0_cy5', 'hybe23', 'FarRed'),
            ('RS810.0_cy5', 'hybe24', 'FarRed'),
            ('RS458122_cy5', 'hybe21', 'FarRed'),
            ('RS0083_SS_Cy5', 'hybe5', 'FarRed'),
            ('RS0255_SS_Cy5', 'hybe20', 'FarRed'),
            ('Nonspecific_Readout','hybe27','FarRed'),
            ('Nonspecific_Encoding','hybe26','FarRed'),
            ('PolyT', 'hybe25', 'FarRed')]


nbits = len(bitmap)
parameters = {}

""" New Microscope Setup"""
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['stitch_rotate'] = 0# NEW0 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['register_stitch_reference'] = True

parameters['segment_gpu'] = False
parameters['fishdata']='fishdata_2023Sep06'
parameters['QC_pixel_size'] = 2 # um
parameters['diameter'] = 8 #15 # um
parameters['segment_diameter'] = parameters['diameter']/parameters['pixel_size']
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'hybe25'
parameters['total_channel'] = 'FarRed'
parameters['total_acq'] = 'hybe25' #'hybe25'
parameters['overwrite'] = False #False
parameters['segment_overwrite'] = False #False
parameters['vector_overwrite'] = True #False
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
parameters['background_estimate_iters'] = 2
parameters['stain_correction'] = False
parameters['stain_correction_downsample'] = 10
parameters['stain_correction_kernel'] = 1000