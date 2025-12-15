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
from df_makers.apply_bdt_and_pickle_df import load_bdt_and_apply

#ROOT.EnableImplicitMT()
ROOT.DisableImplicitMT()    # disable ROOT parallel threads

def sanitize_filename(s):
    """
    Remove or replace characters that are not safe for filenames.
    """
    # Remove any character that is not alphanumeric, dash, underscore, or dot
    safe_str = re.sub(r'[^A-Za-z0-9._-]', '-', s)
    return safe_str

def remove_dot_etc(s):
    """
    Remove or replace characters that are not safe for filenames.
    """
    # Remove any character that is not alphanumeric, dash, underscore, or dot
    safe_str = re.sub(r'[^A-Za-z0-9_-]', '-', s)
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

                
#Note this must be run in a newer version of ROOT than in key4hep
def add_BDT_to_new_root_files(runmode, samples, vars_to_save,   
                           BDT_params = None,
                           #BDT_cut_value = 0.9, #Currenty square cut in two BDT outputs space - 0 for no cut
                           yamlpath = check_inputpath(cfg.fccana_opts["yamlPath"])):
    """ 
    Turn root df into pandas so can feed into bdt
    Note: cut must be string that can be passed to Cpp .Filter()
    BDT_params must be dictionary with keys: (config_bdtopts, training_round, hps_dict_name,  features_list_name, bdt_label), 
    necessary inputs for most recent BDT (cfg.optimised_bdt_lh_opts, "baseline-plus-hps", "baseline-plus-hps", "bdtlh-vars-v1", '_lh')
    """

    cut = None #'cut must be none to be able to reattach friend tree!!!!!S

    print(f"{30*'-'}")
    print(f"CREATING AND SAVING new root files containing BDT information")
    print(f"{30*'-'}\n")

    #path to data and outputs
    inputpath  = check_inputpath(cfg.fccana_opts["outputDir"][runmode])

    if cut:
        cut_name = sanitize_filename(cut)
        outputpath   = set_outputpath(os.path.join(inputpath,f"root_bdtscores_cut{cut_name}")) 
    else:
        outputpath   = set_outputpath(os.path.join(inputpath,"root_bdtscores"))

    if BDT_params:
        #convert BDT cut value into a string(e.g., 0.999 -> "0999")
        #BDT_cut_str =str(BDT_cut_value).replace('.','')
        # Create the name
        #bdtcut_name = f"bdtlh_{BDT_cut_str}cut"
        bdtcut_name = "bdtlh_nocut"
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

    #if BDT_params:
    #    eff_bdtlh_cut = {}
    #    N_before_bdtlh_cut = {}

    # now collect relevant events into a dataframe
    for decay in samples:
        print(f'Starting processing decay: {decay}')
        decay_outpath = set_outputpath(os.path.join(outputpath,f'{decay}'))
        eff = efficiencies_dict[decay+'_eff']
        filepaths = filepaths_dict[decay+'_files']
        populated_filepaths=[]

        #if BDT_params:
        #    N_pre=0
        #    N_post=0

        #only try and get events from files that have non-zero number of selected events
        for file in filepaths:
            # Read TParameter objects from the original ROOT file
            input_file = ROOT.TFile.Open(file)
            tparam = input_file.Get("eventsSelected")
            evt_selected_val = tparam.GetVal()
            if evt_selected_val  !=0:
                populated_filepaths.append(file)

        #do each file separately
        nchunks = len(populated_filepaths)

        for n in range(nchunks):
            print(f'---> Starting processing chunk {n} of {nchunks}')
            file =populated_filepaths[n]
            filename = os.path.basename(file)
            print(filename)
            Rdf = ROOT.RDataFrame("events", file)

            if cut:
                Rdf = Rdf.Filter(cut)

            # Define a synthetic event ID
            Rdf = Rdf.Define("evt_id", "rdfentry_")

            # Also add a column with chunk number (use as fake run number!)
            # Extract chunk number from filename
            basename = os.path.basename(file)          
            chunk_str = basename.replace(".root", "")  
            chunk_num = int(chunk_str.split("_")[1])
            Rdf = Rdf.Define("chunk", str(chunk_num))

            # Snapshot with evt_id and chunk included
            out_file = os.path.join(decay_outpath, filename.replace(".root", "_with_evtid.root"))
            all_cols = list(Rdf.GetColumnNames())
            cols_to_save = [c for c in all_cols if c != "Rec_vtx_indRP"] #Rec_vtx_indRP problematic and not needed for BDT
            Rdf.Snapshot("events", out_file, cols_to_save)  


            Rdf_np = Rdf.AsNumpy(columns= vars_to_save + ["evt_id"] + ["chunk"]) ##add vars that can be used as ID to vars to save so that later can create event ID and use to match events
            sub_df = pd.DataFrame(Rdf_np)
            sub_df["decay"] = decay
            sub_df["eff_presel"] = eff

            #print(Rdf_np["evt_id"])

            # want to make sure that integer types are actually set as integers - currenlty stored as float
            #if changed branches significantly might be worth checking the list is still right, with current branches expected integers in yaml
            integer_branches = [s for s in vars_to_save if '_n' in s and '_norm' not in s] + ["evt_id","chunk"]
            for integer_branch in integer_branches:
                sub_df[integer_branch] = sub_df[integer_branch].astype(np.int32) #must be int32 to match BDT training data

            #print(np.array(sub_df["evt_id"]))


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
                

                #convert back to arrays
                bdt0 = dataframe_chunk['bdt_score_0'].to_numpy()
                bdt1 = dataframe_chunk['bdt_score_1'].to_numpy()
                bdt2 = dataframe_chunk['bdt_score_2'].to_numpy()
                chunk = dataframe_chunk['chunk'].to_numpy()
                evt_id = dataframe_chunk['evt_id'].to_numpy()

                ##############################################################################################
                # Save BDT scores into a friend tree + index branches
                ##############################################################################################
                # Create a new ROOT file for the friend tree
                friend_out = os.path.join(decay_outpath, filename.replace(".root", "_bdt_friend.root"))
                fout = ROOT.TFile(friend_out, "RECREATE")
                friend = ROOT.TTree("bdt_scores", "BDT output scores")

                # Buffers for branches
                bdt0_buf   = np.zeros(1, dtype=np.float64)
                bdt1_buf   = np.zeros(1, dtype=np.float64)
                bdt2_buf   = np.zeros(1, dtype=np.float64)
                evt_id_buf   = np.zeros(1, dtype=np.int32)
                chunk_buf   = np.zeros(1, dtype=np.int32)

                
                friend.Branch("bdt_score_0", bdt0_buf, "bdt_score_0/D")
                friend.Branch("bdt_score_1", bdt1_buf, "bdt_score_1/D")
                friend.Branch("bdt_score_2", bdt2_buf, "bdt_score_2/D")
                friend.Branch("evt_id", evt_id_buf, "evt_id/L")
                friend.Branch("chunk", chunk_buf, "chunk/L")

                # Fill the friend tree from your arrays
                for evtid, ch, v0, v1, v2 in zip(evt_id,chunk, bdt0, bdt1, bdt2):
                    evt_id_buf[0] = evtid
                    chunk_buf[0] = ch
                    bdt0_buf[0]   = v0
                    bdt1_buf[0]   = v1
                    bdt2_buf[0]   = v2
                    friend.Fill()

                friend.Write()
                fout.Close()


def add_friends_and_bdtcut(runmode, samples, bdtcut=0.9996):
    
    inputpath_base  = check_inputpath(cfg.fccana_opts["outputDir"][runmode])
    inputpath   = check_inputpath(os.path.join(inputpath_base,"root_bdtscores"))
    bdtcut_name = "bdtlh_nocut"
    inputpath = check_inputpath(os.path.join(inputpath, bdtcut_name))
    

    trees = {}

    for decay in samples:

        print(decay)
        decay_inputpath = check_inputpath(os.path.join(inputpath,f'{decay}'))
        main_files = glob.glob(os.path.join(decay_inputpath, '*_with_evtid.root'))

        #get max number 
        numbers = []
        pattern = re.compile(r'(\d+)_with_evtid\.root$')

        for f in main_files:
            m = pattern.search(os.path.basename(f))
            if m:
                numbers.append(int(m.group(1)))

        highest = max(numbers) if numbers else None
        print("Highest number:", highest) 
        
        for n in range(highest+1):
            print(f"chunk {n}")


            filename = f"chunk_{n}.root"
            main_file = os.path.join(decay_inputpath, filename.replace(".root", "_with_evtid.root"))
            friend_file = os.path.join(decay_inputpath, filename.replace(".root", "_bdt_friend.root"))

            if os.path.exists(main_file)==False:
                continue
            #print(main_file)
            #print(friend_file)

            # Build the main chain
            main_chain = ROOT.TChain("events")
            main_chain.Add(main_file)

            # Build the friend chain
            friend_chain = ROOT.TChain("bdt_scores")
            friend_chain.Add(friend_file)

            # Build index so ROOT can match entries
            main_chain.BuildIndex("chunk","evt_id")
            friend_chain.BuildIndex("chunk","evt_id")
            

            # Attach friend chain to main chain
            main_chain.AddFriend(friend_chain)

            tree = main_chain

            n_events = tree.GetEntries()
            print(f"n events pre cuts: {n_events}")

            # Apply any pre-selection cuts that may not have been applied already
            cut_string = "EVT_hemisEmax_n > 10  && EVT_hemisEmin_nLept == 0"

            if bdtcut is not None:
                onemcut = 1-bdtcut
                bdtcut_string = f"bdt_score_1<{str(onemcut)} && bdt_score_0 <{str(onemcut)}"
                cut_string = bdtcut_string + "&&" + cut_string


            # Apply the cut: CopyTree returns a new TTree object
            cut_tree = main_chain.CopyTree(cut_string)

            # Save to a new file
            outputpath = os.path.join(decay_inputpath, filename.replace(".root", f"_with_bdtcut{remove_dot_etc(str(bdtcut))}.root"))
            out_file = ROOT.TFile(outputpath, "RECREATE")
            cut_tree.Write()
            out_file.Close()

            print("Filtered tree saved with", cut_tree.GetEntries(), "entries")



    #Note can combined files can be created using: 
    # hadd -f combined_chunks_with_evtid.root *evtid.root
    #hadd -f combined_chunks_bdt_friend.root *friend.root
    #hadd -f combined_chunks_with_bdtcut0-99965.root *with_bdtcut0-99965.root
   



#apply BDT cut

'''
# Now create a single RDataFrame from the combined chain
rdf = ROOT.RDataFrame(main_chain)

rdf = ROOT.RDataFrame(main_chain)
rdf_friend = ROOT.RDataFrame(friend_chain)

print(rdf_friend.AsNumpy(["evt_id"]))
print(rdf.AsNumpy(["evt_id"]))
'''
                

