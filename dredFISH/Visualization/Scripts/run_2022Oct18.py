#!/usr/bin/env python
"""Example run
"""
import os

from dredFISH.Visualization import viz_cell_layer
# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
# mode = 'plot-only' # analysis only; no plots
# mode = 'compile-only' # analysis only; no plots

mode = 'view'
redo = False
basepth_s = [
    "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-R_8C_2022Oct18-S1", # TMG
    # "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-R_8C_2022Oct18-S2", # TMG
    # "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-R_8C_2022Oct18-S3", # TMG
    # "/bigstore/GeneralStorage/Data/dredFISH/DPNMF-R_8C_2022Oct18-S4", # TMG
]
split_lines_s = [
         [[(10500, 2500),(13500,16000)]],
] # hemi-sphere line

for basepth, split_lines in zip(basepth_s, split_lines_s):
    name = os.path.basename(basepth)
    viz_cell_layer.main(mode, basepth, split_lines, 
                        compile_pdf=True,
                        pdf_kwargs={'title': name,
                                    'author': "Fangming",
                                    },
                        tmg_kwargs=dict(
                                    norm_cell=True,
                                    norm_basis=True,
                                    ),
                        redo=redo,
                        )
