import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from plotters.flavtag_plotter import make_maxpK_efficiency_plot, make_nK_plot, make_maxpKS_efficiency_plot, make_nKS_plot


plt.style.use('fcc.mplstyle')

# load Ks and Kpm data
data_savepath = 'outputs/prelim_cuts_full_data/flavtag_dataframes/selected_kaons'

with open(os.path.join(data_savepath,'flavtag_prompt_charged_K.pkl'), 'rb') as f:
    charged_fsK_dict_fromPV = pickle.load(f)

    
with open(os.path.join(data_savepath,'flavtag_KS.pkl'), 'rb') as f:
    fsKS_dict = pickle.load(f)


#prompt charge K plots
savepath = 'plots/paper_plots/JHEP_proofs_replies'

#savepath = 'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/flavour_tagging/'
make_maxpK_efficiency_plot(charged_fsK_dict_fromPV,K_type = ' signal-side final-state $K^\pm$ from PV', savepath=savepath,plottype='chargedK')
make_nK_plot(charged_fsK_dict_fromPV,K_type = ' signal-side final-state $K^\pm$ from PV', savepath=savepath,plottype='chargedK')


#KS plots
make_maxpKS_efficiency_plot(fsKS_dict , savepath=savepath,plottype='KS')
make_nKS_plot(fsKS_dict, savepath=savepath,plottype='KS')
