import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import os
import dill

import config as cfg 
import bdt_lh_cut_opt_with_n_interp as cutopt
plt.style.use('fcc.mplstyle')
import gc
import glob
import sys
import ROOT
import math
from tabulate import tabulate
from yaml import safe_load, YAMLError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
 
import efficiency_finder
import bdt_plotter_multiclass as bp

ROOT.EnableImplicitMT()

# Return list of variables to use in the bdts as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)

    return bdtvars

def check_inputpath(inputpath):
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"{inputpath} does not exist")
    return inputpath


def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath


# Function to load the BDT model from a JSON file
def load_bdt_model(json_path):
    bdt_model = xgb.Booster()
    bdt_model.load_model(json_path)
    return bdt_model


# very quick script to apply baseline-plus bdt_hl to  data for flavourtagging
'''
branching_fractions = cfg.branching_fractions
samples =  cfg.sample_allocations['combined_signal']#cfg.samples

df_dict={}
df_sample_dict={}
eff_bdtlh_cut = {}
N_before_cut = {}

cut_val = 0.5
'''
save_base_path = set_outputpath(os.path.join(cfg.fccana_opts["outputDir"]["prelim_cuts_full"],'baseline_plus_bdtlh_dataframes','flavtag_dataframes'))
full_save_path = set_outputpath(os.path.join(save_base_path,'full_sample_medium_bdtlh_cut'))
'''
dataframe={}
for sample in samples:
    print(sample)

    df_folder = os.path.join(cfg.fccana_opts["outputDir"]["prelim_cuts_full"],'flavtag_dataframes',sample)
    files = os.listdir(df_folder)

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
        dataframe_chunk['eventsProcessed'] = cfg.eventsProcessed[sample]

        cut_df_chunk = dataframe_chunk[((1-dataframe_chunk['bdt_score_1'])>cut_val)&((1-dataframe_chunk['bdt_score_0'])>cut_val)]
        df_sample_dict[f] = cut_df_chunk
  
    dataframe[sample] = pd.concat( [df_sample_dict[s] for s in files], ignore_index=True )

# Concatenate all DataFrames
full_data = pd.concat(dataframe.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus


# add optimal cut
#open saved N_remaining interpolations
save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/'
    
with open(os.path.join(set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

with open(os.path.join(set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
    N_dict = dill.load(dill_file)
'''
##Add BDT cut - for BF=1e-6
BF=1e-6
'''
#find optimum cut
FOM, _, _, _, _, _, lsearch, hsearch, _  = cutopt.run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BF=BF, incl_ZqqBFerror=False)

## finding maximum so can plot slices
indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
l_cut = lsearch[indices[0]]
h_cut = hsearch[indices[1]]

#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
cut_data = full_data.copy().query(f'(EVT_hemisEmax_n>10)&(P_not_light>{l_cut})&(P_not_heavy>{h_cut})')
'''
#keep desired branches
flavtag_branches = ["Rec_p","Rec_px","Rec_py","Rec_pz","Rec_pt","Rec_true_PDG","Rec_in_hemisEmin","Rec_indvtx","Rec_vtx_isPV", "decay" ] # would be good to add "Rec_track_absd0","Rec_track_absnormd0"
reco_space_branches = ["Rec_p","Rec_px","Rec_py","Rec_pz","Rec_pt","Rec_true_PDG","Rec_in_hemisEmin","Rec_indvtx"] # would be good to add "Rec_track_absd0","Rec_track_absnormd0"
'''
data = cut_data.filter(items=flavtag_branches).reset_index(drop=True)


del dataframe
del cut_data
del full_data
gc.collect()

df_listified = data.applymap(lambda x: list(x) if type(x)!= str else x)

df_listified.to_pickle(os.path.join(save_base_path,f'BF{BF}_selected_signal_flavtag_dataframe.pkl'))
'''

df_listified = pd.read_pickle('/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts_full_data/baseline_plus_bdtlh_dataframes/flavtag_dataframes/BF1e-06_selected_signal_flavtag_dataframe.pkl')

# further filter so that only signal side K from PV
pv=[]
#create empty dfs to fill
filtered_df = pd.DataFrame(columns= reco_space_branches )
filtered_df_fully = pd.DataFrame(columns= reco_space_branches )
#loop through events
for i in range(len(df_listified["Rec_p"])):
    evt = df_listified.iloc[i]
    mask = (np.array(evt['Rec_in_hemisEmin']) == 1) & ((np.array(np.round(evt['Rec_true_PDG'])) == 321)| (np.array(np.round(evt['Rec_true_PDG'])) == -321))
    for var in reco_space_branches:
        filtered_df.loc[i, var] = np.array(evt[var])[mask]
    
    from_PV=[]
    for k in range(len(filtered_df['Rec_indvtx'][i])):
        if filtered_df['Rec_indvtx'][i][k] == -999:
            from_PV.append(0)
        else:
            from_PV.append(np.array(df_listified['Rec_vtx_isPV'][i])[filtered_df['Rec_indvtx'][i][k]])
    pv.append(from_PV)
    mask = np.array(from_PV)==1
    for var in reco_space_branches:
        filtered_df_fully.loc[i, var] = filtered_df.loc[i, var][mask]

#add decay back on

filtered_df_fully['decay'] = df_listified['decay'].values()


filtered_df_fully.to_pickle(os.path.join(save_base_path,f'BF{BF}_prompt_signal_K_flavtag_dataframe.pkl'))