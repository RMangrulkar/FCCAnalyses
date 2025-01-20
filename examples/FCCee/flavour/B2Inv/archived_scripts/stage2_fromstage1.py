# stage2_fromstage1.py - must be in the same directory as config.py
# Saves BDT2 and BDTCOMB values for stage1 files
import os
import ROOT

import numpy as np

from glob import glob
from yaml import safe_load
from time import time
from datetime import timedelta

import config as cfg

ROOT.EnableImplicitMT()

if __name__ == "__main__":
    
    ##############################
    ## INITIALISE
    ##############################
    start = time()
    print(f"{30*'-'}")
    print(f"SAVE STAGE2 FILES FROM STAGE1")
    print(f"{30*'-'}\n")
    print("Initialising...")

    stage1path = cfg.fccana_opts['outputDir']['stage1']
    stage2path = cfg.fccana_opts['outputDir']['stage2']
    bdt2path = cfg.bdt2_opts['mvaPath']
    bdtCombpath = cfg.bdtComb_opts['mvaPath']

    # Info
    print(f"----> INFO: Using files from")
    print(f"{15*' '}{stage1path}")
    print(f"----> INFO: Saving files to")
    print(f"{15*' '}{stage2path}")
    print(f"----> INFO: Loading BDT2 model from")
    print(f"{15*' '}{bdt2path}")
    print(f"----> INFO: Loading BDTCOMB model from")
    print(f"{15*' '}{bdtCombpath}")

    # Load branch names used by the BDTs    
    with open(cfg.fccana_opts['yamlPath'], 'r') as stream:
        yaml_file = safe_load(stream)
        bdt2vars = yaml_file[cfg.bdt2_opts['mvaBranchList']]
        bdtCombvars = yaml_file[cfg.bdtComb_opts['mvaBranchList']]
        branchList = yaml_file[cfg.fccana_opts['outBranchList2']]

    # Get a list of all stage1 files
    stage1filepaths = {sample: os.path.join(stage1path, sample, "*.root") for sample in cfg.samples}
    stage1files = {sample: glob(stage1filepaths[sample]) for sample in cfg.samples}

    ##############################
    ## LOAD TMVA FILES
    ##############################
    ROOT.gInterpreter.ProcessLine(f'''
    TMVA::Experimental::RBDT bdt2("{cfg.bdt2_opts['mvaRBDTName']}", "{bdt2path}");
    auto computeModel2 = TMVA::Experimental::Compute<{len(bdt2vars)}, float>(bdt2);
    ''')

    ROOT.gInterpreter.ProcessLine(f'''
    TMVA::Experimental::RBDT bdtComb("{cfg.bdtComb_opts['mvaRBDTName']}", "{bdtCombpath}");
    auto computeModelComb = TMVA::Experimental::Compute<{len(bdtCombvars)}, float>(bdtComb); 
    ''')

    ##############################
    ## PROCESS STAGE1 FILES
    ##############################
    print(f"\n{30*'-'}\n")
    for sample in cfg.samples:
        print(f"----> INFO: Found {len(stage1files[sample])} stage1 files for sample")
        print(f"{15*' '}{sample}")

        if len(stage1files[sample]) == 0:
            print(f"----> WARNING: No files found for {sample}, skipping...")
            continue
        
        # If output directory doesn't exist create it
        if not(os.path.exists(os.path.join(stage2path, sample))):
            os.makedirs(os.path.join(stage2path, sample))
            print(f"----> INFO: Ouput directory does not already exist, creating")
            print(f"{15*' '}{os.path.join(stage2path, sample)}")

        for file in stage1files[sample]:
            # Load the ROOT file
            f = ROOT.TFile(file, "READ")
            t = f.Get("events")
            process = f.Get("eventsProcessed").GetVal()
            select  = f.Get("eventsSelected").GetVal()

            # Load the stage1 data
            df = ROOT.RDataFrame(t)

            # Save BDT prediction
            df2 = (
                df
                .Define("MVA2Vec",     ROOT.computeModel2, bdt2vars)
                .Define("MVACombVec",  ROOT.computeModelComb, bdtCombvars)
                .Define("EVT_MVA2",    "MVA2Vec.at(0)")       
                .Define("EVT_MVAComb", "MVACombVec.at(0)")
            )

            # Save the stage2 data
            df2.Snapshot("events", os.path.join(stage2path, sample, os.path.basename(file)), branchList)
            
            # Close the stage1 root file
            f.Close()

            # Open the newly created file and add the TParameters
            f = ROOT.TFile(os.path.join(stage2path, sample, os.path.basename(file)), "UPDATE")
            param1 = ROOT.TParameter("int")("eventsProcessed", process)
            param2 = ROOT.TParameter("int")("eventsSelected",   select)
            param1.Write()
            param2.Write()
            f.Close()

        print(f"----> INFO: Saved {len(stage1files[sample])} {sample} stage2 files\n")
    
    # Summary
    end = time()
    print(f"{30*'-'}")
    print(f"Execution time = {timedelta(seconds=end-start)}")
    print(f"{30*'-'}")
