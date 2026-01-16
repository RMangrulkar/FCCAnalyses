import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tabulate import tabulate
import matplotlib.pyplot as plt


import config as cfg
import bdt_lh_cut_opt_significance
from bdt_lh_cut_opt_significance import flatten_list, plot_N, plot_interpolted_effs
import basic_functions
import bdt_lh_cut_opt_significance_exclusive_backgrounds

samples_allocations_all = ["combined_signal", "hadronic_background","Bu2lnu_background","Bc2lnu_background"] #["derived_B2lnu_samples"]#
samples_all = flatten_list([cfg.sample_allocations[component] for component in samples_allocations_all])

#Load dataframe with bdtlh version applied
data={}

for sample in samples_all:
    print(sample)
    if sample in cfg.exclusive_backgrounds:

        allocation = next((key for key, samples_list in cfg.sample_allocations.items() if sample in samples_list),None) #iterates over each (key, samples_list) pair giving the first key whose list contains sample. Use to select if Bs or Bd sample for file path

        if allocation is None:
            print(f"Set up needed for other exclusive backgrounds: {sample}")
            continue  # skip this sample or handle appropriately

                
        if sample in cfg.sample_allocations["derived_B2lnu_samples"]:

            sample_derived_from = cfg.samples_to_derive_new_Tau2HNu[sample]
            
            dir = cfg.fccana_opts["outputDir"][f"{allocation}_no_lepton_veto"]
            folder = 'dataframes/bdtlh_099cut' #use samples that DONT HAVE SS lepton veto applied to model hadronic one-prong decay of tau

            pkl_files = glob(os.path.join(dir, folder,sample_derived_from, "*.pkl"))
        
            if len(pkl_files) == 0:
                raise FileNotFoundError("No pickle files found in the folder.")

            dfs = [pd.read_pickle(f) for f in pkl_files]

            data[sample] = pd.concat(dfs, ignore_index=True)

            data[sample]["decay"] = sample # need to change "decay" column in data to new label for create_N_map function to behave!!

            #print(data[sample]["decay"])

            #print(data.keys())

        else:
            dir = cfg.fccana_opts["outputDir"][f"{allocation}_no_lepton_veto"]
            folder = 'dataframes_cutEVT_hemisEmin_nLept----0/bdtlh_099cut' #use samples that already have lepton veto applied

            pkl_files = glob(os.path.join(dir, folder,sample, "*.pkl"))
        
            if len(pkl_files) == 0:
                raise FileNotFoundError("No pickle files found in the folder.")

            dfs = [pd.read_pickle(f) for f in pkl_files]

            data[sample] = pd.concat(dfs, ignore_index=True)

        #print(len(data[sample]))

    else:
        #inclusive backgrounds and signal samples
        dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
        folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut'
        data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))



# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data = full_data.assign(
    P_signal=full_data['bdt_score_2'],
    P_not_heavy=1 - full_data['bdt_score_1'],
    P_not_light=1 - full_data['bdt_score_0'],
)

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus

path_for_interpolated_dicts_0999 =  'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'

bdt_lh_cut_opt_significance.create_N_map(full_data,lrange=(0.999,1) ,hrange=(0.999,1),nlh_highstats=20,nlh_midstats=8,  nlh_lowstats=4, smoothing=False, kx=2,ky=2, save_path=path_for_interpolated_dicts_0999)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "N_remaining_dictionary"), "rb") as dill_file:
    N_dict = dill.load(dill_file)

bdt_lh_cut_opt_significance.plot_N(N_dict, interp_N_dict, lrange=(0.999,1) ,hrange=(0.999,1),nlh_highstats=20, nlh_midstats=8, nlh_lowstats=4,normalised=True, separate_cbar = True,slice=True, save_path='plots/exclusive_background_studies/N_interp_0999/8x8xmidstats_4x4lowstats/incl_Tau2HNu')