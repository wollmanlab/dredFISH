#!/usr/bin/env python
"""Example run
"""
import os

from dredFISH.Visualization import viz_cell_layer
# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
# mode = 'plot-only' #
# mode = 'compile-only' # 

mode = 'view'
# mode = 'plot-only'
redo = False
basepth_s = [
    "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-FR_7C,PFA+Methanol,PFA,Methanol_2022Nov07/DPNMF-FR_7C,PFA+Methanol,PFA,Methanol_2022Nov07_Section15",
]

for basepth in basepth_s:
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
    # break