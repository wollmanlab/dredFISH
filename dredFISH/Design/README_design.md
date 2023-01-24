# folder structure

## Fangming, Jan 2023

### organization

Design/

- train_model_v3p0.py: design demo - single command; self contained
- train....py
- ...
- eval_model_v3p0.ipynb: interpreting design results - this follows train_model
- eval.....ipynb
- ...
- environments_xxxxx.yml: the conda env used to run and test these design code
- data_loaders/
- models/
  - PNMF.py: a python reimplementation of PNMF including its variations DPNMF and DPNMF-tree
  - model_xxx.py: Neural network based designs with many variations
- Misc/ (all kinds of pre- and post-processing)
  - nupack/
  - paintshop/
  - ...
  - Test/
  - Archive/
