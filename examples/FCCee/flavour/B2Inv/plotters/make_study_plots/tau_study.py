# tau study
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
import json
from time import time
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import log_loss


from ... import config as cfg
from ...efficiency_tools import efficiency_finder
from .. import variable_plotter as vp
import post_bdt_application as bp


style_path = os.path.join(os.path.dirname(__file__), '..', 'fcc.mplstyle')
plt.style.use(style_path)


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

# tau df path

branching_fractions = cfg.branching_fractions
samples = cfg.sample_allocations['tau_background']+ cfg.sample_allocations['combined_signal']

df_dict={}
df_sample_dict={}

for sample in samples:

    df_folder = os.path.join(cfg.fccana_opts["outputDir"]["prelim_cuts_full"],'dataframes',sample)
    files = os.listdir(df_folder)

    for f in files:
        fpath = os.path.join(df_folder,f)
        model, bdtname, dataframe_chunk = bp.load_bdt_and_apply(pickled_df_path = fpath, 
                            config_bdtopts = cfg.bdt_lh_opts,
                            training_round = "multiclass_baseline",
                            hps_dict_name = "default-hps",
                            features_list_name = "bdth-plus-vars",
                            bdt_label = '_lh')
        df_sample_dict[f] = dataframe_chunk

    dataframe = pd.concat( [df_sample_dict[s] for s in files], ignore_index=True )
    weight = branching_fractions[sample][0] * dataframe["eff_presel"][0] #need zero index for bf as branching fractions is a tuple in the yaml
    dataframe["w1"] = weight / len(dataframe)
    df_dict[sample] = dataframe

df = pd.concat( [df_dict[s] for s in samples], ignore_index=True )


outputpath = set_outputpath('/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/prelim_cuts_full_data/tau_investigation/')

#bp.post_bdt_variable_plot(df,'bdt_score_2', outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = 'bdt_score_2>0.95',density=False ) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
labels = {'signal':2,'heavy_background':1 ,'light_background':0}
labels_dict_inverted = {v: k for k, v in labels.items()}
#bp.plot_2d_bdt_output_any_sample(df, bdt_name = "BDT_lh",output_file_name = '2dbdt_response_plot',outpath=outputpath,bdt_probs = [0,1],labels=labels_dict_inverted, vmax=2000, xrange=[0,0.1],yrange=[0,0.1],density=True,sample = 'p8_ee_Ztautau_ecm91')
#bp.post_bdt_variable_plot(df,'Rec_vtx_ntracks_max_hemisEmax', outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = '(bdt_score_1<0.05) & (bdt_score_0<0.05) ',density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
#bp.post_bdt_variable_plot(df,'EVT_hemisEmax_nDV', outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = '(bdt_score_1<0.05) & (bdt_score_0<0.05) ',density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
#bp.post_bdt_variable_plot(df,'Rec_PV_ntracks', outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = '(bdt_score_1<0.05) & (bdt_score_0<0.05) ',density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
bp.post_bdt_variable_plot(df,'EVT_hemisEmin_e', outpath=outputpath, components=["Bssignal", "tau_background"],density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)


'''
yamlpath = cfg.fccana_opts['yamlPath']
bdtvars_list_old = cfg.bdt_lh_opts['mvaBranchList']
bdtvars_list_optimised = cfg.optimised_bdt_lh_opts['mvaBranchList']
responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot
vars_to_plot      = list(set(vars_fromyaml(yamlpath, bdtvars_list_old) + vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))


for var in vars_to_plot:
  bp.post_bdt_variable_plot(df,var, outpath=outputpath, components=["Bssignal", "tau_background"],bdt_cut = '(bdt_score_1<0.1) & (bdt_score_0<0.1) ',density=True) #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue)
''' 

