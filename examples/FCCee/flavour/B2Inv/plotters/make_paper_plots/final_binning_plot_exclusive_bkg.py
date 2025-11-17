import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tabulate import tabulate



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import bdt_lh_cut_opt_significance
from bdt_lh_cut_opt_significance import flatten_list

save_path='outputs/prelim_cuts_full_data/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995'

#read N interp dicts
with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

plotpath = 'plots/exclusive_background_studies'


samples_allocations = ["combined_signal", "hadronic_background","Bu2lnu_background",] ##"Bc2lnu_background"
samples = flatten_list([cfg.sample_allocations[component] for component in samples_allocations])
 

#Load dataframe with bdtlh version applied
data={}

for sample in samples:
    print(sample)
    if sample in cfg.exclusive_backgrounds:
        allocation = next((key for key, samples_list in cfg.sample_allocations.items() if sample in samples_list),None)

        if allocation is None:
            print(f"Set up needed for other exclusive backgrounds: {sample}")
            continue  # skip this sample or handle appropriately

        dir = cfg.fccana_opts["outputDir"][f"{allocation}_no_lepton_veto"]
        folder = 'dataframes_cutEVT_hemisEmin_nLept----0/bdtlh_099cut' 

        pkl_files = glob(os.path.join(dir, folder,sample, "*.pkl"))
    
        if len(pkl_files) == 0:
            raise FileNotFoundError("No pickle files found in the folder.")

        dfs = [pd.read_pickle(f) for f in pkl_files]

        data[sample] = pd.concat(dfs, ignore_index=True)


    else:
        #inclusive backgrounds and signal samples
        dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
        folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut'
        data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))



# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus


bdt_lh_cut_opt_significance.make_final_binning_plot(full_data, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 200, signal_BF=7.5e-8, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components =  samples_allocations, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

