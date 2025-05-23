import dill
import os
import numpy as np
import pandas as pd

import config as cfg
import bdt_lh_cut_opt_with_n_interp

save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/'

# make N interp plots
with open(os.path.join(bdt_lh_cut_opt_with_n_interp.set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(bdt_lh_cut_opt_with_n_interp.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/paper_plots'

'''
bdt_lh_cut_opt_with_n_interp.plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,normalised = True, separate_cbar = True, slice=False, save_path=plotpath)
'''

# make final binning plot

#Load dataframe with bdtlh version applied
data={}
dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut' 
for sample in cfg.samples:
    data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))

# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus

bdt_lh_cut_opt_with_n_interp.make_final_binning_plot(full_data, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 500, signal_BF=1e-6, eventsProcessed_dict = cfg.eventsProcessed , histbins=(2,2), components =  ['hadronic_background','combined_signal'], binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                            plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True)
