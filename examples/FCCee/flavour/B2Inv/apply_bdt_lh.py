# very quick script to apply baseline-plus bdt_hl to all data and save back in same format as original datagrames

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from yaml import safe_load, YAMLError, dump
import config as cfg
import bdt_plotter_multiclass as bp


def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath

# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars

branching_fractions = cfg.branching_fractions
samples = cfg.sample_allocations["hadronic_background"]#cfg.samples

df_dict={}
df_sample_dict={}
eff_bdtlh_cut = {}
N_before_cut = {}

#cut_val = 0.5
#save_base_path = set_outputpath(os.path.join(cfg.fccana_opts["outputDir"]["prelim_cuts_full"],'baseline_plus_bdtlh_dataframes'))
#full_save_path = set_outputpath(os.path.join(save_base_path,'full_sample_medium_bdtlh_cut'))

cut_val = 0.9
bdtcut_name = 'bdtlh_09cut' #'full_sample_medium_bdtlh_cut'
save_base_path = set_outputpath(os.path.join(cfg.fccana_opts["outputDir"]["process_with_MC_full_prelim"],'baseline_plus_bdtlh_dataframes'))
raw_df_path = os.path.join(cfg.fccana_opts["outputDir"]["process_with_MC_full_prelim"],'dataframes')
full_save_path = set_outputpath(os.path.join(save_base_path,bdtcut_name))


with open(os.path.join(full_save_path,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
    log_file.write(f'BDT_lh cut at: {cut_val} for 1-P(h) and 1-P(l)\n')

for sample in samples:
    print(sample)

    df_folder = os.path.join(raw_df_path,sample)
    files = os.listdir(df_folder)
    save_folder = set_outputpath(os.path.join(save_base_path,sample))

    N_pre=0
    N_post=0
    for f in files:
        fpath = os.path.join(df_folder,f)
        model, bdtname, dataframe_chunk = bp.load_bdt_and_apply(pickled_df_path = fpath, 
                            config_bdtopts = cfg.optimised_bdt_lh_opts,
                            training_round = "baseline-plus-hps",
                            hps_dict_name = "baseline-plus-hps",
                            features_list_name = "bdtlh-vars-v1",
                            bdt_label = '_lh',
                            test_train_valid = False)
        
        dataframe_chunk['decay'] = sample
        #dataframe_chunk['eventsProcessed'] = cfg.eventsProcessed[sample]

        savepath = os.path.join(save_folder,f)
        dataframe_chunk.to_pickle(savepath)

        N_pre += len(dataframe_chunk)
        
        cut_df_chunk = dataframe_chunk[((1-dataframe_chunk['bdt_score_1'])>cut_val)&((1-dataframe_chunk['bdt_score_0'])>cut_val)]
        df_sample_dict[f] = cut_df_chunk #If only keeping small number of vars combine into one big dataframe - however if many vars this will get big too quickly
        N_post += len(cut_df_chunk)
    
    N_before_cut[sample] = N_pre 
    eff_bdtlh_cut[sample] = N_post/N_pre                                      
    
    #dataframe = pd.concat( [df_sample_dict[s] for s in files], ignore_index=True )
    #dataframe.to_pickle(os.path.join(full_save_path,f'{sample}.pkl'))

    with open(os.path.join(full_save_path,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
        log_file.write(f'{sample}\n')
        log_file.write(f'medium cut efficiency: {eff_bdtlh_cut[sample]}\n')
        log_file.write(f'number events pre cut: {N_before_cut[sample]}\n')



'''
yamlpath = cfg.fccana_opts['yamlPath']
bdtvars_list_old = cfg.bdt_lh_opts['mvaBranchList']
bdtvars_list_optimised = cfg.optimised_bdt_lh_opts['mvaBranchList']
responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot
vars_to_plot      = list(set(vars_fromyaml(yamlpath, bdtvars_list_old) + vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))


for var in vars_to_plot:
  bp.post_bdt_variable_plot(df,var, outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = '(bdt_score_1<0.1) & (bdt_score_0<0.1) ',density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
''' 

