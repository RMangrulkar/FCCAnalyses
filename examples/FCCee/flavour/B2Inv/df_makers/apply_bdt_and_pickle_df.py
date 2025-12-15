import os
import glob
import sys
import gc
import ROOT
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from yaml import safe_load, YAMLError
from xgboost import XGBClassifier


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg 
from  efficiency_tools import efficiency_finder
import basic_functions

#ROOT.EnableImplicitMT()

def load_bdt_and_apply(df, 
                        config_bdtopts = cfg.baseline_bdt_lh_opts,
                        training_round = "multiclass_baseline",
                        hps_dict_name = "default-hps",
                        features_list_name = "bdth-plus-vars",
                        bdt_label = '_lh',
                        ): # hps dict and features_list_name specift BDT used
    
    
    #path to data and outputs
    outputpath   = config_bdtopts['outputPath']
    yamlpath = cfg.fccana_opts['yamlPath']

    #Getting BDT vars for training from yaml
    bdtvars      = basic_functions.vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'

    #path to pickled data

    #path to saved bdt
    bdt_json_path = os.path.join(outputpath,training_round,f"{bdtname}.json")

    # Load the BDT model
    try:
        print("Loading BDT model...")
        bdt_model = basic_functions.load_bdt_model_sklearn(bdt_json_path)
        print("BDT model loaded successfully.")
    except Exception as e:
        print(f"Error loading BDT model: {e}")
        quit()

    # Read df saved to pickle and add BDT 

    #Getting BDT vars for training from yaml

    #add bdt score
    class_names = bdt_model.classes_
    probabilities = bdt_model.predict_proba(df[bdtvars])
    for i in range(probabilities.shape[1]):
        df[f'bdt_score_{class_names[i]}'] = probabilities[:, i]

    return bdt_model, bdtname, df




if __name__ == '__main__':

    runmode = 'process_with_MC_full_prelim'

    #################################
    ## PREPROCESSING AND CREATING DF
    #################################
    print(f"{30*'-'}")
    print(f"CREATING AND SAVING df each sample")
    print(f"{30*'-'}\n")

    #path to data and outputs
    inputpath    = basic_functions.check_inputpath(cfg.fccana_opts["outputDir"][runmode]) 
    yamlpath     = basic_functions.check_inputpath(cfg.fccana_opts["yamlPath"])
    save_base_path = basic_functions.set_outputpath(os.path.join(cfg.fccana_opts["outputDir"]["process_with_MC_full_prelim"],'baseline_plus_bdtlh_dataframes'))


    #variables for BDT cut and subsequent naming
    cut_val = 0.999
    # Convert to a string (e.g., 0.999 -> "0999")
    cut_str =str(cut_val).replace('.','')
    # Create the name
    bdtcut_name = f"bdtlh_{cut_str}cut"
    full_save_path = basic_functions.set_outputpath(os.path.join(save_base_path,bdtcut_name))


    #Getting BDT vars for training from yaml
    #bdtvars_list_old = cfg.baseline_bdt_lh_opts['mvaBranchList']
    bdtvars_list_optimised = cfg.optimised_bdt_lh_opts["mvaBranchList"]
    responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot
    bdtvars      = list(set( basic_functions.vars_fromyaml(yamlpath, bdtvars_list_optimised)  + responsevars))
    flavtag_vars = list(basic_functions.vars_fromyaml(yamlpath, "flavour-tag-vars"))
    truth_vars = list(basic_functions.vars_fromyaml(yamlpath,"MCtruth-vars"))

    saved_vars = list(set(bdtvars+truth_vars))

    # print statements to check loading things expect
    print(f"----> INFO: Loading files from")
    print(f"{15*' '}{inputpath}")
    print(f"----> INFO: Output will be saved to")
    print(f"{15*' '}{full_save_path}")

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


    # going to print the efficiencies for each now so can manually check
    print("Preselection Efficiencies:"+'\n')
    eff_to_print = [[ key , efficiencies_dict[key+'_eff']] for key in samples]
    print( tabulate(  eff_to_print, headers=["decay", "efficiency"] ) +'\n')


    #########################################################################
    ## Collecting events into dfs and adding preselection efficoiency columns
    #########################################################################

    # now collect relevant events into a dataframe
    eff_bdtlh_cut = {}
    N_before_cut = {}

    for decay in samples:

        decay_outpath = basic_functions.set_outputpath(os.path.join(full_save_path,f'{decay}'))

        df_sample_dict={}
        

        eff = efficiencies_dict[decay+'_eff']
        filepaths = filepaths_dict[decay+'_files']
        populated_filepaths=[]

        N_processed = 0

        #only try and get events from files that have non-zero number of selected events
        for file in filepaths:
            # Read TParameter objects from the original ROOT file
            input_file = ROOT.TFile.Open(file)
            tparam = input_file.Get("eventsSelected")
            eventsProcessed_tparam = input_file.Get("eventsProcessed")
            evt_processed_val = eventsProcessed_tparam.GetVal()
            evt_selected_val = tparam.GetVal()
            if evt_selected_val  !=0:
                populated_filepaths.append(file)

            N_processed += evt_processed_val


        # if over 20 files, chunk into multiple dataframes
        if len(populated_filepaths)>10:
            chunked_populated_files = basic_functions.chunk_list(populated_filepaths,10)
            nchunks = math.ceil(len(populated_filepaths)/10)

        else:
            nchunks = 1
            chunked_populated_files = [populated_filepaths]

        #write to log
        with open(os.path.join(full_save_path,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
            log_file.write(f'BDT_lh cut at: {cut_val} for 1-P(h) and 1-P(l)\n')

        save_folder = basic_functions.set_outputpath(os.path.join(full_save_path,decay))

        N_pre=0
        N_post=0

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

            #####################
            # apply BDT and cut #
            #####################
            
            model, bdtname, dataframe_chunk = load_bdt_and_apply(sub_df, 
                                config_bdtopts = cfg.optimised_bdt_lh_opts,
                                training_round = "baseline-plus-hps",
                                hps_dict_name = "baseline-plus-hps",
                                features_list_name = "bdtlh-vars-v1",
                                bdt_label = '_lh')


            N_pre += len(dataframe_chunk)
            
            cut_df_chunk = dataframe_chunk[((1-dataframe_chunk['bdt_score_1'])>cut_val)&((1-dataframe_chunk['bdt_score_0'])>cut_val)]
            #df_sample_dict[n] = cut_df_chunk #If only keeping small number of vars combine into one big dataframe - however if many vars this will get big too quickly
            N_post += len(cut_df_chunk)

            
            # Save the DataFrame to a pickle file for now - can look at cuts
            cut_df_chunk.to_pickle(os.path.join(decay_outpath,f'{decay}_dataframe_chunk{n}.pkl'))
            print(f"DataFrame {decay} chunk {n} saved successfully!")

            # Write key info about df to log file
            with open(os.path.join(full_save_path,'df.log'), 'a') as log_file:
                log_file.write(f'{decay}\n')
                log_file.write(f'chunk {n}\n')
                log_file.write(f'files used: {files}\n')

            # CLEAN-UP to free memory after each chunk
            del Rdf_np
            del Rdf
            del sub_df
            del dataframe_chunk
            del cut_df_chunk
            gc.collect()
        
        N_before_cut[decay] = N_pre 
        eff_bdtlh_cut[decay] = N_post/N_pre 
        total_eff = N_post /N_processed                                  
        
        #dataframe = pd.concat( [df_sample_dict[n] for n in range(nchunks)], ignore_index=True )
        #dataframe['eff_BDT_cut'] = eff_bdtlh_cut[decay] 
        #dataframe['eff_total'] = total_eff 
        #dataframe['eventsProcessed'] = N_processed 

        #dataframe.to_pickle(os.path.join(full_save_path,f'{decay}.pkl'))


        with open(os.path.join(full_save_path,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
            log_file.write(f'{decay}\n')
            log_file.write(f'BDT cut efficiency: {eff_bdtlh_cut[decay]}\n')
            log_file.write(f'number events processed by tupling: {N_processed}\n')


            



