import os
import glob
import sys

import ROOT
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from yaml import safe_load, YAMLError


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg 
from efficiency_tools import efficiency_finder
import basic_functions
from df_makers import data_to_pickle_function


ROOT.DisableImplicitMT() #need to disable otherwise results are off 

#path to data and outputs
yamlpath     = basic_functions.check_inputpath(cfg.fccana_opts['yamlPath'])

#Get list of vars to save
bdtvars_list_optimised = cfg.optimised_bdt_lh_opts["mvaBranchList"]
responsevars = ["EVT_hemisEmin_Emiss"] 

bdtvars      = list(set(basic_functions.vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))
flavtag_vars = list(basic_functions.vars_fromyaml(yamlpath, "flavour-tag-vars"))
truth_vars = list(basic_functions.vars_fromyaml(yamlpath,"MCtruth-vars"))

vars_to_save = list(set(bdtvars+truth_vars+["EVT_hemisEmin_nLept"]))


# list of inputs to say which BDT to use
BDT_params_dict = {"config_bdtopts": cfg.optimised_bdt_lh_opts,
                   "training_round": "baseline-plus-hps",
                   "hps_dict_name":"baseline-plus-hps",
                   "features_list_name": "bdtlh-vars-v1",
                   "bdt_label": "_lh"}

'''
samples = ["p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU", "p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu"] #["p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu"]#cfg.sample_allocations["Bu2lnu_background"]# MUST BE A LIST 
runmode = "Bu2lnu_background_no_lepton_veto"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) 
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)


samples = cfg.sample_allocations["Bc2lnu_background"]# MUST BE A LIST 
runmode = "Bc2lnu_background_no_lepton_veto"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) 
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)


samples = ["p8_ee_Zbb_ecm91"]# MUST BE A LIST 
runmode ="process_with_MC_full_prelim"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) #BDT_cut_value = 0.99)
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)

samples = ["p8_ee_Zcc_ecm91"]# MUST BE A LIST 
runmode ="process_with_MC_full_prelim"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) #BDT_cut_value = 0.99)
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)

samples = cfg.sample_allocations["light_hadronic_background"]# MUST BE A LIST 
runmode ="process_with_MC_full_prelim"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) #BDT_cut_value = 0.99)
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)

samples = cfg.sample_allocations["combined_signal"]# MUST BE A LIST 
runmode ="process_with_MC_full_prelim"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) #BDT_cut_value = 0.99)
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)
'''

samples = cfg.processList["ella_INVestigations"].keys()# MUST BE A LIST 
runmode ="ella_INVestigations"
data_to_pickle_function.add_BDT_to_new_root_files(runmode, samples, vars_to_save, BDT_params = BDT_params_dict) #BDT_cut_value = 0.99)
data_to_pickle_function.add_friends_and_bdtcut(runmode, samples, bdtcut=0.99965)


