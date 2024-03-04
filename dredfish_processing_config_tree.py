import numpy as np
import logging
# Basic parameters of imaging
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
            ('RS458122_cy5', 'hybe21', 'FarRed'), # (’R25’,’RS458122_cy5’, 'hybe25', 'FarRed',’AACTCCTTATCACCCTACTC’)
            ('RS0083_SS_Cy5', 'hybe5', 'FarRed'), # (’R26’,’RS0083_SS_Cy5’, ’hybe5/bDNA’, ‘FarRed’,’ACACTACCACCATTTCCTAT’)
            ('RS0255_SS_Cy5', 'hybe20', 'FarRed'), # (’R27’,’RS0255_SS_Cy5’, ’hybe20/DNA’, ‘FarRed’,’TCCTATTCTCAACCTAACCT’)
            ('Nonspecific_Readout','hybe27','FarRed'), # (’R5’,'RS0307_cy5', 'hybe5', 'FarRed',’TATCCTTCAATCCCTCCACA’)
            ('Nonspecific_Encoding','hybe26','FarRed'), #(’R28’,’RS1261_SS_Cy5’, ’hybe21/bDNA’, ‘FarRed’,’ACACCATTTATCCACTCCTC’) [Non Specific Encoding Probe ] 
            ('PolyT', 'hybe25', 'FarRed')] #
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
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['stitch_rotate'] = 0# NEW0 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['register_stitch_reference'] = True
parameters['use_FF'] = True
parameters['use_constant'] = True
parameters['segment_gpu'] = False
parameters['fishdata']='fishdata'
parameters['QC_pixel_size'] = 2 # umj
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
parameters['highpass_sigma'] = 25#int(50/parameters['pixel_size']) #ums -> pixels
parameters['highpass_function'] = 'rolling_ball'
parameters['highpass_smooth_function'] = 'none'
parameters['strip'] = True
parameters['highpass_smooth'] = 0
parameters['model_types'] = ['total']
parameters['dapi_thresh'] = 1000
parameters['processing_log_level'] = logging.DEBUG
parameters['registration_log_level'] = logging.DEBUG
parameters['background_estimate_iters'] = 0
parameters['stain_correction'] = False
parameters['stain_correction_downsample'] = 10
parameters['stain_correction_kernel'] = 1000
parameters['overlap'] = 0.02 # 2% overlap
parameters['segment_min_size'] = parameters['segment_diameter']*10