import os
import glob
import sys
import re
import gc
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
from apply_bdt_and_pickle_df import load_bdt_and_apply

ROOT.EnableImplicitMT()

def sanitize_filename(s):
    """
    Remove or replace characters that are not safe for filenames.
    """
    # Remove any character that is not alphanumeric, dash, underscore, or dot
    safe_str = re.sub(r'[^A-Za-z0-9._-]', '-', s)
    return safe_str

def root_data_to_pickle_df(runmode, samples, vars_to_save, 
                           cut = None,  
                           BDT_params = None,
                           BDT_cut_value = 0.9, #Currenty square cut in two BDT outputs space - 0 for no cut
                           yamlpath = check_inputpath(cfg.fccana_opts["yamlPath"])):
    """ 
    Turn root df into pandas so can feed into bdt
    Note: cut must be string that can be passed to Cpp .Filter()
    BDT_params must be dictionary with keys: (config_bdtopts, training_round, hps_dict_name,  features_list_name, bdt_label), 
    necessary inputs for most recent BDT (cfg.optimised_bdt_lh_opts, "baseline-plus-hps", "baseline-plus-hps", "bdtlh-vars-v1", '_lh')
    """

    print(f"{30*'-'}")
    print(f"CREATING AND SAVING df for each sample")
    print(f"{30*'-'}\n")

    #path to data and outputs
    inputpath    = check_inputpath(cfg.fccana_opts["outputDir"][runmode])

    if cut:
        cut_name = sanitize_filename(cut)
        outputpath   = set_outputpath(os.path.join(inputpath,f"dataframes_cut{cut_name}")) 
    else:
        outputpath   = set_outputpath(os.path.join(inputpath,"dataframes"))

    if BDT_params:
        #convert BDT cut value into a string(e.g., 0.999 -> "0999")
        BDT_cut_str =str(BDT_cut_value).replace('.','')
        # Create the name
        bdtcut_name = f"bdtlh_{BDT_cut_str}cut"
        outputpath = set_outputpath(os.path.join(outputpath, bdtcut_name))
    
    # print statements to check loading things expect
    print(f"----> INFO: Loading files from")
    print(f"{15*' '}{inputpath}")
    print(f"----> INFO: Output will be saved to")
    print(f"{15*' '}{outputpath}")

    #calculating efficiencies and also saving files paths used to calculate efficiencies to ensure save same ones
    selection_efficiency = efficiency_finder.get_efficiencies('custom',
                                                        further_analysis=True,
                                                        samples = samples,
                                                        raw=True, #ie. want full efficiency including tupling and prelim cuts
                                                        cut = cut,
                                                        custompath=inputpath,
                                                        verbose=False,
                                                        return_files_list=True)
    #nb if want to specify a certain number of chunks to use do it in here and will follow through into filepaths

    filepaths_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_files')}
    efficiencies_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_eff')}
    efficiencies_err_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_err')}


    # going to print the efficiencies for each now so can manually check
    print("Efficiencies:"+'\n')
    eff_to_print = [[ key , efficiencies_dict[key+'_eff']] for key in samples]
    print( tabulate(  eff_to_print, headers=["decay", "efficiency"] ) +'\n')


    #########################################################################
    ## Collecting events into dfs and adding preselection efficiency columns
    #########################################################################

    if BDT_params:
        eff_bdtlh_cut = {}
        N_before_bdtlh_cut = {}

    # now collect relevant events into a dataframe
    for decay in samples:
        print(f'Starting processing decay: {decay}')
        decay_outpath = set_outputpath(os.path.join(outputpath,f'{decay}'))
        eff = efficiencies_dict[decay+'_eff']
        filepaths = filepaths_dict[decay+'_files']
        populated_filepaths=[]

        if BDT_params:
            N_pre=0
            N_post=0

        #only try and get events from files that have non-zero number of selected events
        for file in filepaths:
            # Read TParameter objects from the original ROOT file
            input_file = ROOT.TFile.Open(file)
            tparam = input_file.Get("eventsSelected")
            evt_selected_val = tparam.GetVal()
            if evt_selected_val  !=0:
                populated_filepaths.append(file)

        # if over 5 files, chunk into multiple dataframes
        if len(populated_filepaths)>5:
            chunked_populated_files = chunk_list(populated_filepaths,5)
            nchunks = math.ceil(len(populated_filepaths)/5)

        else:
            nchunks = 1
            chunked_populated_files = [populated_filepaths]


        for n in range(nchunks):
            print(f'---> Starting processing chunk {n} of {nchunks}')
            files =chunked_populated_files[n]
            Rdf = ROOT.RDataFrame("events", files)
            if cut:
                Rdf = Rdf.Filter(cut)
            Rdf_np = Rdf.AsNumpy(columns= vars_to_save)
            sub_df = pd.DataFrame(Rdf_np)
            sub_df["decay"] = decay
            sub_df["eff_presel"] = eff

            # want to make sure that integer types are actually set as integers - currenlty stored as float
            #if changed branches significantly might be worth checking the list is still right, with current branches expected integers in yaml
            integer_branches = [s for s in vars_to_save if '_n' in s and '_norm' not in s]
            for integer_branch in integer_branches:
                sub_df[integer_branch] = sub_df[integer_branch].astype(np.int32)


            if BDT_params:

                #####################
                # apply BDT and cut #
                #####################
                
                model, bdtname, dataframe_chunk = load_bdt_and_apply(sub_df, 
                                    config_bdtopts = BDT_params["config_bdtopts"],
                                    training_round = BDT_params["training_round"],
                                    hps_dict_name = BDT_params["hps_dict_name"],
                                    features_list_name = BDT_params["features_list_name"],
                                    bdt_label = BDT_params["bdt_label"])
                

                N_pre += len(dataframe_chunk)
                cut_df_chunk = dataframe_chunk[((1-dataframe_chunk['bdt_score_1'])>BDT_cut_value)&((1-dataframe_chunk['bdt_score_0'])>BDT_cut_value)]
                N_post += len(cut_df_chunk)
                
                #write to log BDT efficiencies
                with open(os.path.join(outputpath,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
                    log_file.write(f'BDT_lh cut at: {BDT_cut_value} for 1-P(h) and 1-P(l)\n')

                #Ovewrite sub_df so can save in same way as if no BDT applied
                sub_df = cut_df_chunk


            sub_df.to_pickle(os.path.join(decay_outpath,f'{decay}_dataframe_chunk{n}.pkl'))
            print(f"DataFrame {decay} chunk {n} saved successfully!")

            # Write key info about df to log file
            with open(os.path.join(outputpath,'data_df.log'), 'a') as log_file:
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

        #Add BDT cut efficiencies to log file
        if BDT_params:
            N_before_bdtlh_cut[decay] = N_pre 
            eff_bdtlh_cut[decay] = N_post/N_pre 
                                 
            with open(os.path.join(outputpath,f'{bdtcut_name}_efficiencies.log'), 'a') as log_file:
                log_file.write(f'{decay}\n')
                log_file.write(f'BDT cut efficiency: {eff_bdtlh_cut[decay]}\n')


    with open(os.path.join(outputpath,'data_df.log'), 'a') as log_file:           
        log_file.write(f'data taken from path: {inputpath}\n')  
        log_file.write(f'Efficiencies used: {tabulate( eff_to_print, headers=["decay", "efficiency"])}\n')

                
            
        

            








