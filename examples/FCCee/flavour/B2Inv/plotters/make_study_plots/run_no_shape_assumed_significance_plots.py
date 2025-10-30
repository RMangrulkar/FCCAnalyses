import dill
import os
import numpy as np
import pandas as pd
import pickle
from tabulate import tabulate

from ... import config as cfg
from ... import bdt_lh_cut_opt_significance

output_dir = cfg.fccana_opts['outputDir']['prelim_cuts_full']

save_path=os.path.join(output_dir,'BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/')


# open N interp dict
with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

plotpath = os.path.join(cfg.FCCAnalysesPath,'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation/no_shape_assumed')

#Load dataframe with bdtlh version applied
data={}
folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut' 
for sample in cfg.samples:
    data[sample] = pd.read_pickle(os.path.join(output_dir,folder,f'{sample}.pkl'))

# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus

dict_path = os.path.join(output_dir,'BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation/')
bdt_lh_cut_opt_significance.plot_BF_sensitivities_no_shape_assumed(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(np.log10(1e-9),np.log10(4e-5),400), incl_other_syst=True, incl_toys_fit=True, full_df=full_data, ntoys=2000,plot=True,saveplotpath = plotpath ,toyplotpath = plotpath, dict_path = dict_path)
