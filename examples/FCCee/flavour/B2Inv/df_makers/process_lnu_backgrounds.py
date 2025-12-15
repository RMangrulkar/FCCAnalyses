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


ROOT.EnableImplicitMT()

#path to data and outputs
inputpath    = basic_functions.check_inputpath(cfg.fccana_opts['outputDir']['Bc2lnu_background_no_lepton_veto']) # note this data has full preselctio except hemisEmin_nlepton cut
yamlpath     = basic_functions.check_inputpath(cfg.fccana_opts['yamlPath'])
samples = ["p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU", "p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu"]#cfg.sample_allocations["Bc2lnu_background"]# MUST BE A LIST 
runmode = "Bc2lnu_background_no_lepton_veto"

# print statements to check loading things expect
print(f"----> INFO: Loading files from")
print(f"{15*' '}{inputpath}")


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

#Apply without lepton veto
#data_to_pickle_function.root_data_to_pickle_df(runmode, samples, vars_to_save, cut = None, BDT_params = BDT_params_dict, BDT_cut_value = 0.99)

#Apply with lepton veto
data_to_pickle_function.root_data_to_pickle_df(runmode, samples, vars_to_save, cut = "EVT_hemisEmin_nLept == 0", BDT_params = BDT_params_dict, BDT_cut_value = 0.99)