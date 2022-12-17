#!/usr/bin/env python
"""Example run
"""
import os
import glob
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                datefmt='%m-%d %H:%M:%S', 
                level=logging.INFO,
                )

from dredFISH.Visualization import viz_cell_layer
# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
# mode = 'plot-only' #
# mode = 'compile-only' # 

mode = 'view'
# mode = 'plot-only'
redo = False
# basepth_s = [
#     "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-FR_7C_PFA+Methanol_PFA_Methanol_2022Nov07/DPNMF-FR_7C_PFA+Methanol_PFA_Methanol_2022Nov07_Section15",
# ]
basepth_s = np.sort(glob.glob(
    "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-FR_Z1_Z2_9A_Z3_Z4_6C_2022Nov15/DPNMF*Section*"
))
logging.info(f"{len(basepth_s)} samples:")
logging.info(f"{basepth_s}")

for basepth in basepth_s:
    try: 
        samp = basepth.split('/')[-1]
        viz_cell_layer.main(mode, basepth, 
                            compile_pdf=True,
                            pdf_kwargs={'title': samp,
                                        'author': "Fangming",
                                        },
                            tmg_kwargs=dict(
                                        norm_cell=True,
                                        norm_basis=True,
                                        ),
                            redo=redo,
                            )
    except:
        print(f"{samp} failed during processing")

    # break
