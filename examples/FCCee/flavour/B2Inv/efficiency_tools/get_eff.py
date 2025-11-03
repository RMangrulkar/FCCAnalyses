
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
from tabulate import tabulate
import os
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from efficiency_tools import efficiency_finder


def check_inputpath(inputpath):
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"{inputpath} does not exist")
    return inputpath

samples = cfg.samples
inputpath    = check_inputpath(cfg.fccana_opts['outputDir']['prelim_cuts_full']) 

#calculating efficiencies and also saving files paths used to calculate efficiencies to ensure save same ones
selection_efficiency = efficiency_finder.get_efficiencies('custom',
                                                    further_analysis=True,
                                                    samples = samples,
                                                    raw=True, #ie. want full efficiency including tupling and prelim cuts
                                                    custompath=inputpath,
                                                    verbose=False,
                                                    return_files_list=False)
#nb if want to specify a certain number of chunks to use do it in here and will follow through into filepaths

#filepaths_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_files')}
efficiencies_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_eff')}
efficiencies_err_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_err')}


# going to print the efficiencies and BF for each now so can manually check
print("Runmode = full preliminary cuts on all data")
print(f"Data taken from {inputpath}")
print("Efficiencies and err:"+'\n')
eff_to_print = [[ key , efficiencies_dict[key+'_eff'], efficiencies_err_dict[key+'_err']] for key in samples]
print( tabulate(  eff_to_print, headers=["decay", "efficiency", "efficiency error"] ) +'\n')

print('In tuple from...')
print(selection_efficiency)


with open(os.path.join(inputpath,'total_prelim_efficicncies.log'), 'a') as log_file:
    log_file.write(f'Selection efficiency of full prelim cuts\n')
    log_file.write(f'{tabulate(  eff_to_print, headers=["decay", "efficiency", "efficiency error"])}\n')
    log_file.write(f'no rounding\n')
    log_file.write(f'{selection_efficiency}\n')
    print(f"logfile saved to {os.path.join(inputpath,'total_prelim_efficicncies.log')}")

