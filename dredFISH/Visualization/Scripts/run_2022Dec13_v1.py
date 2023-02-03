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
redo = True 

basepth_s = np.sort(glob.glob(
    "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-FR_R1_4A_UC_R2_5C_2022Nov27_Dec12_strip_tol/DPNMF*Section*"
))
logging.info(f"{len(basepth_s)} samples:")
logging.info(f"{basepth_s}")

for basepth in basepth_s:
    try: 
        samp = basepth.split('/')[-1]
        viz_cell_layer.main(mode, basepth, 
                            title=samp,
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
