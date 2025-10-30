import os
import sys
import ROOT
from yaml import safe_load
from math import sqrt


# Config and yaml file must be in this directory by default
# Absolute path must be supplied for the script to work in batch mode
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg

# Read list of feature names used in the BDT from the config YAML file
with open(cfg.fccana_opts['yamlPath']) as stream:
    yaml = safe_load(stream)
    BDT1branchList = yaml[cfg.bdt1_opts['mvaBranchList']]


folder = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts/"
samples = cfg.samples
for sample in samples:
    sample_path = os.path.join(folder, f'{sample}/')
    files = os.listdir(sample_path)
    for filename in files:
        root_filepath = os.path.join(sample_path, f'{filename}')
        tree_name = "events"

        # Create a ROOT RDataFrame from the TTree
        df = ROOT.RDataFrame(tree_name, root_filepath)

        ROOT.gInterpreter.ProcessLine(f'''
        TMVA::Experimental::RBDT bdt1("{cfg.bdt1_opts['mvaRBDTName']}", "{cfg.bdt1_opts['mvaPath']}");
        auto computeModel1 = TMVA::Experimental::Compute<{len(BDT1branchList)}, float> (bdt1);
        ''')

        df2 = (
            df
            #############################################
            ##                Build BDT                ##
            #############################################
            .Define("MVAVec",    ROOT.computeModel1, BDT1branchList)
            .Define("EVT_MVA1",  "MVAVec.at(0)")
        )

        # Read TParameter objects from the original ROOT file
        input_file = ROOT.TFile.Open(root_filepath)
        param1 = input_file.Get("eventsProcessed")  
        param2 = input_file.Get("eventsSelected") 

        # Write df2 to a new ROOT file
        output_file_path = os.path.join(sample_path, f"{os.path.splitext(filename)[0]}_withMVA1.root")
        print(output_file_path)
        df2.Snapshot("events", output_file_path)

        # Write TParameter objects to the new ROOT file
        output_file = ROOT.TFile.Open(output_file_path, "UPDATE")
        param1.Write()
        param2.Write()
 
