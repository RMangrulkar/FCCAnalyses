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
from basic_functions import vars_fromyaml, check_inputpath, set_outputpath, chunk_list
from efficiency_tools import efficiency_finder

ROOT.EnableImplicitMT()

runmode = 'process_with_MC_full_prelim'

#################################
## PREPROCESSING AND CREATING DF
#################################
print(f"{30*'-'}")
print(f"CREATING AND SAVING df each sample")
print(f"{30*'-'}\n")

#path to data and outputs
inputpath    = check_inputpath(cfg.fccana_opts["outputDir"][runmode]) 
outputpath   = set_outputpath(os.path.join(inputpath,"dataframes"))
yamlpath     = check_inputpath(cfg.fccana_opts["yamlPath"])

#Getting BDT vars for training from yaml
#bdtvars_list_old = cfg.baseline_bdt_lh_opts['mvaBranchList']
bdtvars_list_optimised = cfg.optimised_bdt_lh_opts["mvaBranchList"]
responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot
#bdtvars      = list(set(vars_fromyaml(yamlpath, bdtvars_list_old) + vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))
bdtvars      = list(set( vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))
flavtag_vars = list(vars_fromyaml(yamlpath, "flavour-tag-vars"))
truth_vars = list(vars_fromyaml(yamlpath,"MCtruth-vars"))

saved_vars = list(set(bdtvars+truth_vars))

# print statements to check loading things expect
print(f"----> INFO: Loading files from")
print(f"{15*' '}{inputpath}")
print(f"----> INFO: Output will be saved to")
print(f"{15*' '}{outputpath}")

samples = cfg.sample_allocations["ud_only"]+cfg.sample_allocations["combined_signal"]


#calculating efficiencies and also saving files paths used to calculate efficiencies to ensure save same ones
selection_efficiency = efficiency_finder.get_efficiencies('custom',
                                                    further_analysis=True,
                                                    samples = samples,
                                                    raw=True, #ie. want full efficiency including tupling and prelim cuts
                                                    custompath=inputpath,
                                                    verbose=False,
                                                    return_files_list=True)
#nb if want to specify a certain number of chunks to use do it in here and will follow through into filepaths

filepaths_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_files')}
efficiencies_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_eff')}
efficiencies_err_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_err')}


# going to print the efficiencies and BF for each now so can manually check
print("Efficiencies and BFs:"+'\n')
eff_to_print = [[ key , efficiencies_dict[key+'_eff']] for key in samples]
print( tabulate(  eff_to_print, headers=["decay", "efficiency"] ) +'\n')


#########################################################################
## Collecting events into dfs and adding preselection efficoiency columns
#########################################################################

# now collect relevant events into a dataframe
for decay in samples:
    eff = efficiencies_dict[decay+'_eff']
    filepaths = filepaths_dict[decay+'_files']
    populated_filepaths=[]
    #only try and get events from files that have non-zero number of selected events
    for file in filepaths:
        # Read TParameter objects from the original ROOT file
        input_file = ROOT.TFile.Open(file)
        tparam = input_file.Get("eventsSelected")
        evt_selected_val = tparam.GetVal()
        if evt_selected_val  !=0:
            populated_filepaths.append(file)

    # if over 10 files, chunk into multiple dataframes
    if len(populated_filepaths)>10:
        chunked_populated_files = chunk_list(populated_filepaths,10)
        nchunks = math.ceil(len(populated_filepaths)/10)

    else:
        nchunks = 1
        chunked_populated_files = [populated_filepaths]


    for n in range(nchunks):
        files =chunked_populated_files[n]
        Rdf = ROOT.RDataFrame("events", files)
        Rdf_np = Rdf.AsNumpy(columns= saved_vars)
        sub_df = pd.DataFrame(Rdf_np)
        sub_df["decay"] = decay
        sub_df["eff_presel"] = eff

        # want to make sure that integer types are actually set as integers - currenlty stored as float
        #if changed branches significantly might be worth checking the list is still right, with current branches expected integers in yaml
        integer_branches = [s for s in saved_vars if '_n' in s and '_norm' not in s]
        for integer_branch in integer_branches:
            sub_df[integer_branch] = sub_df[integer_branch].astype(np.int32)


        decay_outpath = set_outputpath(os.path.join(outputpath,f'{decay}'))

        # Save the DataFrame to a pickle file for now - can look at cuts
        sub_df.to_pickle(os.path.join(decay_outpath,f'{decay}_dataframe_chunk{n}.pkl'))
        print(f"DataFrame {decay} chunk {n} saved successfully!")

        # Write key info about df to log file
        with open(os.path.join(outputpath,'full_data_df.log'), 'a') as log_file:
            log_file.write(f'{decay}\n')
            log_file.write(f'chunk {n}\n')
            log_file.write(f'files used: {files}\n')
        
        
with open(os.path.join(outputpath,'full_data_df.log'), 'a') as log_file:           
    log_file.write(f'data taken from path: {inputpath}\n')  
    log_file.write(f'Efficiencies and BF used: {tabulate( eff_to_print, headers=["decay", "efficiency"])}\n')



