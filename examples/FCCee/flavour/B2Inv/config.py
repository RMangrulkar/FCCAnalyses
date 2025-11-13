# config.py
# This file contains all of the important configuration options used throughtout the analysis
# This file, along with all other analysis scripts, are expected to be in the `FCCAnalysesPath` directory
# Also contains branching fractions, cut efficiencies, etc which are manually filled in for now
import os
import matplotlib.pyplot as plt
import numpy as np

# MANDATORY ----> replace the default string with the path to the B2Inv directory in the FCCAnalyses repo
FCCAnalysesPath = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/"
FCCAnalysesPath = os.path.abspath(FCCAnalysesPath)
SavedOutputsPath = "/r02/lhcb/ejnw2/FCC_outputs_2024/outputs" #this is where old BDTs are saved
# RUNNING MODE
run_mode_choices = ['no_selection','no_selection_taus','n_lept_cut_failed','prelim_cuts','prelims_incltauveto_evtdisp','prelim_cuts_full', 'process_with_MC_full_prelim', 'Bu2lnu_background_no_lepton_veto', 'Bc2lnu_background_no_lepton_veto'] #when add run mode, now need to add to processList, fccana_opts AND PROCESS_TUPLES!!!

#BDTh - single hadronic BDT, used to separate signal from all hadronic bkgs in one go
#BDTl - BDT to discriminate against light hadronic bkgs (u,d,s)
#BDTmE - BDT to look for missing energy events in events that pass BDTl


run_mode = 'Bc2lnu_background_no_lepton_veto'
if run_mode not in run_mode_choices:
    raise RuntimeError(f'{run_mode} is not a valid run mode')


##############################
## CONFIG DICTS
##############################
# processList to pass to `fccanalysis run`
processList = {
    # Size of winter2023 samples in /eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/:
    # p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu == 13G (2,000,000 events)
    # p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu ~= 13G (2,200,000 events)
    # p8_ee_Zbb_ecm91                == 3.3T (438,738,637 events)
    # p8_ee_Zcc_ecm91                == 3.5T (499,786,495 events)
    # p8_ee_Zss_ecm91                == 3.3T (499,842,440 events)
    # p8_ee_Zud_ecm91                == 3.3T (497,658,654 events)
    # p8_ee_Ztautau_ecm91            == 140G  (100,000,000 events)
    # p8_ee_Zmumu_ecm91              == 93G  (100,000,000 events)
    # p8_ee_Zee_ecm91                == 97G  (100,000,000 events)
    # p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu == 1,000,000 events
    # p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU  ==  10,000,000 events
    # p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu  == 10,000,000 events
    # p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu == 5,000,000 events
    # p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU  ==  10,900,000 events
    # p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu  == 10,000,000 events 


    "no_selection": {  # 100,000 events per sample just to look at
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 0.05, "chunks": 1},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 0.05, "chunks": 1},
        "p8_ee_Zbb_ecm91": {"fraction": 0.00025, "chunks": 2},
        "p8_ee_Zcc_ecm91": {"fraction": 0.00025, "chunks": 2},
        "p8_ee_Zss_ecm91": {"fraction": 0.00025, "chunks": 2},
        "p8_ee_Zud_ecm91": {"fraction": 0.00025, "chunks": 2},
        "p8_ee_Ztautau_ecm91": {"fraction": 0.05, "chunks": 40}, 
        "p8_ee_Zmumu_ecm91": {"fraction": 0.05, "chunks": 40},
        "p8_ee_Zee_ecm91": {"fraction": 0.05, "chunks": 40},
    },

    "no_selection_taus": {  # 100,000 events per sample just to look at
        "p8_ee_Ztautau_ecm91": {"fraction": 0.1, "chunks": 40}, 
    },
    
    
    "prelim_cuts": {  # ~2G or ~500k events per sample 
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 0.32, "chunks": 5},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 0.30, "chunks": 5},
        "p8_ee_Zbb_ecm91": {"fraction": 0.028, "chunks": 16},
        "p8_ee_Zcc_ecm91": {"fraction": 0.028, "chunks": 16},
        "p8_ee_Zss_ecm91": {"fraction": 0.028, "chunks": 16},
        "p8_ee_Zud_ecm91": {"fraction": 0.048, "chunks": 32},
        #"p8_ee_Ztautau_ecm91": {"fraction": 1., "chunks": 40}, 
        #"p8_ee_Zmumu_ecm91": {"fraction": 1., "chunks": 100},
        #"p8_ee_Zee_ecm91": {"fraction": 1., "chunks": 100},    
    },

    "prelims_incltauveto_evtdisp":{ 
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 0.001, "chunks": 1},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 0.001, "chunks": 1},
        "p8_ee_Zbb_ecm91": {"fraction": 0.0001, "chunks": 1},
        "p8_ee_Zcc_ecm91": {"fraction": 0.0001, "chunks": 1},
        "p8_ee_Zss_ecm91": {"fraction": 0.0001, "chunks": 1},
        "p8_ee_Zud_ecm91": {"fraction": 0.0001, "chunks": 1},
        "p8_ee_Ztautau_ecm91": {"fraction": 0.1, "chunks":10}, 
        #"p8_ee_Zmumu_ecm91": {"fraction": 1., "chunks": 100},
        #"p8_ee_Zee_ecm91": {"fraction": 1., "chunks": 100}, 
    },

    "prelim_cuts_full": {  # processing all data
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 1, "chunks": 20},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 1, "chunks": 20},
        "p8_ee_Zbb_ecm91": {"fraction": 1, "chunks": 800},
        "p8_ee_Zcc_ecm91": {"fraction": 1, "chunks": 800},
        "p8_ee_Zss_ecm91": {"fraction": 1, "chunks": 800},
        "p8_ee_Zud_ecm91": {"fraction": 1, "chunks": 800},
        "p8_ee_Ztautau_ecm91": {"fraction": 1., "chunks": 50}, 
        "p8_ee_Zmumu_ecm91": {"fraction": 1., "chunks": 100},
        "p8_ee_Zee_ecm91": {"fraction": 1., "chunks": 100},
    },

    
    "n_lept_cut_failed": {  # ~2G or ~500k events per sample 
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 1., "chunks": 8},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 1., "chunks": 8},
        "p8_ee_Zbb_ecm91": {"fraction": 0.001, "chunks": 5},
        "p8_ee_Zcc_ecm91": {"fraction": 0.002, "chunks": 5},
        "p8_ee_Ztautau_ecm91": {"fraction": 0.1, "chunks": 1},    
    },

    "process_with_MC_full_prelim": {
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu":{"fraction": 0.1, "chunks": 2},
        "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": {"fraction": 0.1, "chunks": 2},
        "p8_ee_Zbb_ecm91": {"fraction": 0.35, "chunks": 350},
        "p8_ee_Zcc_ecm91": {"fraction": 0.35, "chunks": 350},
        "p8_ee_Zss_ecm91": {"fraction": 0.35, "chunks": 350},
        "p8_ee_Zud_ecm91": {"fraction": 0.5, "chunks": 500},
    },

    "Bu2lnu_background_no_lepton_veto":{
        "p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu":{"fraction":1,"chunks":10},
        "p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU":{"fraction":1,"chunks":100},
        "p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu":{"fraction":1,"chunks":100},
    },

    "Bc2lnu_background_no_lepton_veto":{
        "p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu":{"fraction":1,"chunks":50},
        "p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU":{"fraction":1,"chunks":100},
        "p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu":{"fraction":1,"chunks":100},
    }


}

# Default options to pass to `fccanalysis run`
fccana_opts = {
    "prodTag":   "FCCee/winter2023/IDEA",
    "outputDir": {
        "no_selection": os.path.join(FCCAnalysesPath, "outputs/no_selection/"),
        "no_selection_taus": os.path.join(FCCAnalysesPath, "outputs/no_selection_taus/"),
        "prelim_cuts": os.path.join(FCCAnalysesPath, "outputs/full_prelim_cuts_500k/"),
        "prelims_incltauveto_evtdisp": os.path.join(FCCAnalysesPath, "outputs/prelims_incltauveto_evtdisp/"),
        "prelim_cuts_full": os.path.join(FCCAnalysesPath, "outputs/prelim_cuts_full_data/"),
        "n_lept_cut_failed":os.path.join(FCCAnalysesPath, "outputs/n_lept_cut_failed/"),
        "stage1_training": os.path.join(FCCAnalysesPath, "outputs/stage1_training/"),
        "process_with_MC_full_prelim":os.path.join(FCCAnalysesPath, "outputs/MC_particles_full_prelim/"),
        "Bu2lnu_background_no_lepton_veto":os.path.join(FCCAnalysesPath, "outputs/Bu2lnu_background_no_lepton_veto/"),
        "Bc2lnu_background_no_lepton_veto":os.path.join(FCCAnalysesPath, "outputs/Bc2lnu_background_no_lepton_veto/"),
     
    },

    "testFile": {
        "Bs": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu/events_026683563.root",
        "Bd": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu/events_004838962.root",
        "bb": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91/events_000083138.root",
        "cc": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zcc_ecm91/events_000046867.root",
        "ss": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zss_ecm91/events_000099129.root",
        "ud": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zud_ecm91/events_000071896.root",
        "tautau": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Ztautau_ecm91/events_000143148.root",
        "mumu": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zmumu_ecm91/events_000128808.root",
        "ee": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zee_ecm91/events_000132426.root",
        "Bu2taunu2HAD":"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU/events_000843366.root",
        "Bu2taunu2mu":	"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu/events_011554472.root",
        "Bu2munu":"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu/events_000407298.root",
        "Bc2munu":"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu/events_000811090.root",
        "Bc2taunu2HAD":"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU/events_001034647.root",
        "Bc2taunu2mu":	"root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu/events_004620103.root",
    
    },
    "analysisName":   "b2inv",
    "nCPUS":          8,
    "runBatch":       True,
    "batchQueue":     "workday",
    "compGroup":      "group_u_FCC.local_gen",
    "yamlPath":       os.path.join(FCCAnalysesPath, "B2Inv.yaml"),  # Path to the YAML file containing feature names
    "outputBranches": {
        "no_selection":"full-vars",
        "no_selection_taus":"full-vars",
        "prelim_cuts": "full-vars",
        "prelims_incltauveto_evtdisp": "eventdisplay-vars",
        "prelim_cuts_full": "full-vars",
        "n_lept_cut_failed": "full-vars",
        "process_with_MC_full_prelim":"full-vars-plus-MCtruth",
        "Bu2lnu_background_no_lepton_veto":"full-vars-plus-MCtruth",
        "Bc2lnu_background_no_lepton_veto":"full-vars-plus-MCtruth",

    },
}


# TMVA options

baseline_bdt_lh_opts = {
    "label":               '_lh',
    #"training":           True,                  
    "inputPath":          fccana_opts['outputDir']['prelim_cuts'], #ie. want to train on data with just preliminary cuts
    "outputPath":         os.path.join(fccana_opts['outputDir']['prelim_cuts'], "bdt_lh_outputs/"),
    "mvaBranchList":      "bdth-plus-vars",  # key in the yaml file pointing to the feature list 
    "signalAllocation":   ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "backgroundAllocation_light":  ["p8_ee_Zss_ecm91", "p8_ee_Zud_ecm91"],
    "backgroundAllocation_heavy":  ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91"],
}



optimised_bdt_lh_opts = {
    "label":               '_lh',
    #"training":           True,                  
    "inputPath":          fccana_opts['outputDir']['prelim_cuts'], #ie. want to train on data with just preliminary cuts
    "outputPath":         os.path.join(fccana_opts['outputDir']['prelim_cuts'], "bdt_lh_outputs/"),
    "mvaBranchList":      "bdtlh-vars-v1",  # key in the yaml file pointing to the feature list 
    "signalAllocation":   ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "backgroundAllocation_light":  ["p8_ee_Zss_ecm91", "p8_ee_Zud_ecm91"],
    "backgroundAllocation_heavy":  ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91"],
}

bdttau_opts = {
    "label":               'tau',
    "inputPath":          os.path.join(fccana_opts["outputDir"]["prelim_cuts_full"],'dataframes'),       
    "outputPath":         os.path.join(fccana_opts['outputDir']['prelim_cuts_full'], "bdttau_outputs/"),
    "mvaBranchList":      "bdttau-baseline-vars",  # key in the yaml file pointing to the feature list 
    "signalAllocation":   ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "backgroundAllocation":  ["p8_ee_Ztautau_ecm91"],
}

bdttau_opts_nonNeutrals = {
    "label":               'tau',
    "inputPath":          os.path.join(fccana_opts["outputDir"]["prelim_cuts_full"],'dataframes'),       
    "outputPath":         os.path.join(fccana_opts['outputDir']['prelim_cuts_full'], "bdttau_outputs/"),
    "mvaBranchList":      "bdttau-nonNeutrals-vars",  # key in the yaml file pointing to the feature list 
    "signalAllocation":   ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "backgroundAllocation":  ["p8_ee_Ztautau_ecm91"],
}



hp_opts = {
    "default-hps": {'n_estimators': 400, 
                    'learning_rate': 0.1, #xgb default=0.3
                    'max_depth': 3, #xgb default=6
                    'gamma': 0, #xgb default (min_split_loss)
                    'min_child_weight': 1, #xgb default
                    'max_delta_step': 0, #xgb default
                    'subsample':1, #xgb default
                    'reg_alpha':0, #xgb default
                    'reg_lambda':1}, #xgb default
    
    "baseline-plus-hps": {'n_estimators': 400, 
                    'learning_rate': 0.1, #xgb default=0.3
                    'max_depth': 4, #xgb default=6
                    'gamma': 0, #xgb default (min_split_loss)
                    'min_child_weight': 1, #xgb default
                    'max_delta_step': 0, #xgb default
                    'subsample':1, #xgb default
                    'reg_alpha':0, #xgb default
                    'reg_lambda':1}, #xgb default


    # for all hp configs that are not 'default_hps' only need to specify changes from default above

    "multiclass-optimum-overtrained":{'n_estimators': 339, 
                'learning_rate': 0.19912998125518783, 
                'max_depth': 10, 
                'gamma': 1.687590612253541, 
                'reg_alpha':2.1524117081043257,
                'reg_lambda':1.8036463746236047}, 

    "multiclass-optimum-smallovertraining":{'n_estimators': 452, 
                          'learning_rate': 0.14133102917518017, 
                          'max_depth': 6, 
                          'gamma': 0.23241307970342628, 
                          'reg_alpha': 0.5171660043477355, 
                          'reg_lambda': 8.027583972204095}, 
    
    "multiclass-optimum":{'n_estimators': 452, 
                          'learning_rate': 0.14133102917518017, 
                          'max_depth': 5, 
                          'gamma': 0.23241307970342628, 
                          'reg_alpha': 0.5171660043477355, 
                          'reg_lambda': 8.027583972204095}, 
                          
    "multiclass-smalltest-optimum":{'gamma': 1.2084995905144988,
                                    'learning_rate': 0.19693924973969526,
                                    'max_depth': 8,
                                    'n_estimators': 493,
                                    'reg_alpha': 7.565153585635452,
                                    'reg_lambda': 9.828959456263297,},


    "default-hps-tau": {'n_estimators': 300, 
                    'learning_rate': 0.1, #xgb default=0.3
                    'max_depth': 4, #xgb default=6
                    'gamma': 0, #xgb default (min_split_loss) 
                    'min_child_weight': 1, #xgb default
                    'max_delta_step': 0, #xgb default
                    'subsample':1, }, #xgb default

    "hp1": {'learning_rate': 0.3,} ,

}


#BSC taken from https://github.com/HEP-FCC/FCCeePhysicsPerformance/blob/master/General/README.md#generating-events-under-realistic-fcc-ee-environment-conditions and agreement checked with MC samples
#nb. if spring2021 values used for winter2023, BSC is too tight --> error and slow fitting: `VertexFit::RegInv: null determinant for N = 2`
BSC_opts = {
    "winter2023": [5.96,23.8e-3,0.397e3], # vertex sigma [x,y,z] in micrometers
    "spring2021": [4.5,20e-3,0.3e3],
}


##############################
## SAMPLE OPTIONS
##############################
samples = [
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu",
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu",
    "p8_ee_Zbb_ecm91",
    "p8_ee_Zcc_ecm91",
    "p8_ee_Zss_ecm91",
    "p8_ee_Zud_ecm91",
    "p8_ee_Ztautau_ecm91",
    "p8_ee_Zmumu_ecm91",
    "p8_ee_Zee_ecm91",
    "p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu",
    "p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU",
    "p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu",
    "p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu",
    "p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU",
    "p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu"
]

sample_allocations = {
    "hadronic_background": ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91", "p8_ee_Zss_ecm91", "p8_ee_Zud_ecm91"],
    "bb_only":    ["p8_ee_Zbb_ecm91"],
    "heavy_hadronic_background": ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91"],
    "light_hadronic_background": ["p8_ee_Zss_ecm91", "p8_ee_Zud_ecm91"], 
    "ud_only": ["p8_ee_Zud_ecm91"], 
    "leptonic_background": ["p8_ee_Ztautau_ecm91","p8_ee_Zmumu_ecm91","p8_ee_Zee_ecm91"],
    "tau_background":  ["p8_ee_Ztautau_ecm91"],
    "light_leptonic_background": ["p8_ee_Zmumu_ecm91","p8_ee_Zee_ecm91"],
    "Bssignal":     ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu"],
    "Bdsignal":   ["p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "combined_signal": ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"],
    "Bu2lnu_background": ["p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU", "p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu"],
    "Bc2lnu_background": ["p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTAUHADNU", "p8_ee_Zbb_ecm91_EvtGen_Bc2MuNu"],
}


sample_colors = {
    "combined_signal": plt.cm.Blues( np.linspace(0, 1, 6)[3:-1] ),
    "hadronic_background": plt.cm.Reds_r( np.linspace(0, 1, 6)[1:-1] ),
    "heavy_hadronic_background": plt.cm.Reds_r( np.linspace(0, 1, 6)[1:3] ),
    "light_hadronic_background": plt.cm.Reds_r( np.linspace(0, 1, 6)[3:-1] ),
    "Bssignal": [ plt.cm.Blues( np.linspace(0, 1, 6)[3] ) ],
    "Bdsignal": [ plt.cm.Blues( np.linspace(0, 1, 6)[-2] ) ],
    "bb_only": [ plt.cm.Reds_r( np.linspace(0, 1, 6)[1] ) ],
    "tau_background": [plt.cm.tab20c(12)],#[ plt.cm.tab20b((4+ np.linspace(0, 1, 5)[1])/5 ) ],
    "leptonic_background": [plt.cm.tab20c([12,13,14])],#plt.cm.tab20b( (4+np.linspace(0, 1, 5)[1:-1]) /5 ),
    "light_leptonic_background": [plt.cm.tab20c([13,14])]#plt.cm.tab20b((4+ np.linspace(0, 1, 5)[2:-1])/5 )
}

sample_total = {
    "combined_signal": None,
    "hadronic_background": 'black',
    "heavy_hadronic_background": 'darkred',
    "light_hadronic_background": 'indianred',
    "leptonic_background": 'indigo',
    "light_leptonic_background": 'mediumvioletred',
    "bb_only": None,
    "tau_background": None,
}

sample_hatches = {
    "combined_signal": [r'////', r'\\\\'],
    "hadronic_background": None,
    "heavy_hadronic_background": None,
    "light_hadronic_background": None,
    "bb_only": None,
    "tau_background": None,
    "leptonic_background": None,
    "light_leptonic_background": None,
    "Bdsignal": [r'////'],
    "Bssignal": [r'\\\\'],
}

sample_shorthand = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": "Bs2NuNu",
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": "Bd2NuNu",
    "p8_ee_Zbb_ecm91":                "Z2bb",
    "p8_ee_Zcc_ecm91":                "Z2cc",
    "p8_ee_Zss_ecm91":                "Z2ss",
    "p8_ee_Zud_ecm91":                "Z2ud",
    "p8_ee_Ztautau_ecm91":            "Z2tautau",
    "p8_ee_Zmumu_ecm91":              "Z2mumu",
    "p8_ee_Zee_ecm91":                "Z2ee",
}

titles = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": r"$B_s^0 \to \nu \bar{\nu}$",
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": r"$B^0 \to \nu \bar{\nu}$",
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_invis": r"$B_s^0 \to$ invisible",
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu_invis": r"$B^0 \to$ invisible",
    "p8_ee_Zbb_ecm91": r"$Z \to b \bar{b}$",
    "p8_ee_Zcc_ecm91": r"$Z \to c \bar{c}$",
    "p8_ee_Zss_ecm91": r"$Z \to s \bar{s}$",
    "p8_ee_Zud_ecm91": r"$Z \to d \bar{d} + Z \to u \bar{u}$",
    "p8_ee_Zud_ecm91_old": r"$Z \to q \bar{q}$, $q \in [u,d]$",
    "p8_ee_Ztautau_ecm91":r"$Z \to \tau^{+} \tau^{-}$",
    "p8_ee_Zmumu_ecm91":r"$Z \to \mu^{+} \mu^{-}$",
    "p8_ee_Zee_ecm91":r"$Z \to e^{+} e^{-}$",
    "heavy_background": r"$Z \to b \bar{b}+Z \to c \bar{c}$",
    "light_background": r"$Z \to s \bar{s}+Z \to d \bar{d}+Z \to u \bar{u}$",
    "signal": r"$B_{(s)}^0 \to \nu \bar{\nu}$",
    "hadronic_background": "Total hadronic background",
    "heavy_hadronic_background": "Total heavy hadronic background",
    "light_hadronic_background": "Total light hadronic background"
}

##############################
## NUMERICAL DATA
##############################
# from PDG -> B production fractions
prod_frac = {
    "Bu":  (0.408, 0.007),#0.43,
    "Bd":  (0.408, 0.007),#0.43,
    "Bs": (0.100, 0.008),#0.096,
    #"Lb": #0.037, 
    #"Bc": #0.0004,Not measured by LEP
    # Actually, use the sample name to make integration with Bd2NuNu easier
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (0.100, 0.008),#(0.096, 0), #
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": (0.408, 0.007),#(0.43, 0), #
}


# from PDG (value, error)
# Z branching fractions
# Z->ss = (Z->dd+ss+bb)/3 * 3 - Z->bb / 2
# Z->uu/cc = 2 * (11.6 +/- 0.6) = 23.2 +/- 1.2
# Z->dd/ss/bb = 3 * (15.6 +/- 0.4) = 46.8 +/- 1.2
#Z->τ+τ− = (3.3696±0.0083) %
#Z->mu+mu− = (3.3662±0.0066) %
#Z->e+e− = (3.3632±0.0042) %

branching_fractions = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (1, 0),  # a dummy value
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": (1, 0),  # a dummy value
    "p8_ee_Zbb_ecm91": (0.1512, 0.0005),
    "p8_ee_Zcc_ecm91": (0.1203, 0.0021),
    "p8_ee_Zss_ecm91": (0.1584, 0.0060),
    "p8_ee_Zud_ecm91": (0.2701, 0.0136),
    "p8_ee_Ztautau_ecm91":(0.033696,0.000083),
    "p8_ee_Zmumu_ecm91":(0.033662,0.000066),
    "p8_ee_Zee_ecm91":(0.033632,0.000042),
}

mass_Z = 91.188  # Ecm used in the winter2023 samples

N_z = 6e12 # total number of Nz expected across all experiments during tera-Z run (from https://arxiv.org/pdf/2309.11353 Matt/Aidan paper)

mass_KS = 0.497611 #\pm 0.000013 GeV
mass_pi0 = 0.1349768#\pm0.0000005 GeV

BELLEII_projected_limits = {'p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu':1.1e-5, 'p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu':1.5e-6}
BELLEII_projected_lumi = {'p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu':r'5 ab$^{-1}$ $\Upsilon(5S)$', 'p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu':r'50 ab$^{-1}$ $\Upsilon(4S)$'}


prelim_cut_effs = {'p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu': (0.876092, 0.00023297513169984185),
 'p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu': (0.8826863636363637, 0.00021695324861459309),
 'p8_ee_Zbb_ecm91': (0.05767557918542743, 1.112995134404382e-05),
 'p8_ee_Zcc_ecm91': (0.041443724884963125, 8.915504876430558e-06), 
 'p8_ee_Zss_ecm91': (0.04597152896410762, 9.367329187236206e-06),  
 'p8_ee_Zud_ecm91': (0.024312405153808926, 6.902017868782445e-06),
 'p8_ee_Ztautau_ecm91': (0.04472734, 2.0670463638239323e-05), 
 'p8_ee_Zmumu_ecm91': (2.29e-06, 1.5140986460763067e-07), 
 'p8_ee_Zee_ecm91': (1.4e-06, 1.1842710888770522e-07)
}

Presel_eff_incl_tau = {'p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_eff': (0.8591695,0.00024596470709481455),
 'p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu_eff': (0.8659218181818182,0.00022972437147417146),
 'p8_ee_Zbb_ecm91_eff': (0.05682641759221219,1.1052690619586802e-05),
 'p8_ee_Zcc_ecm91_eff': (0.039435947543960743,8.705967643483914e-06),
 'p8_ee_Zss_ecm91_eff': (0.041542190314042576,8.925279735547131e-06),
 'p8_ee_Zud_ecm91_eff': (0.021884987304170968, 6.556541853171283e-06),
 'p8_ee_Ztautau_ecm91_eff': (3.434e-05,5.860146762460286e-07),
 'p8_ee_Zmumu_ecm91_eff': (1.2e-07,3.49999975928571e-08),
 'p8_ee_Zee_ecm91_eff': (6e-08, 2.4999999029999996e-08)}

eventsProcessed = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 2000000,
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": 2200000,
    "p8_ee_Zbb_ecm91":438738637,
    "p8_ee_Zcc_ecm91":499786495,
    "p8_ee_Zss_ecm91":499825860,
    "p8_ee_Zud_ecm91":497950940,
    "p8_ee_Ztautau_ecm91":100000000,
    "p8_ee_Zmumu_ecm91":100000000,
    "p8_ee_Zee_ecm91":100000000,
}

eventsSelected_preBDT_notau_veto = { #nb excluding tau veto
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 1752184,
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": 1941910,
    "p8_ee_Zbb_ecm91":25304505,
    "p8_ee_Zcc_ecm91":20713014,
    "p8_ee_Zss_ecm91":22977759,
    "p8_ee_Zud_ecm91":12106385,#12046354,IDK why this was the initial number in here, I think a typo - these also arent used anywhere I dont think so it's not super importnat
    "p8_ee_Ztautau_ecm91":4472734,
    "p8_ee_Zmumu_ecm91":229,
    "p8_ee_Zee_ecm91":140,
}

eventsSelected_preBDT_withtau_veto = { 
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 1718339,
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": 1905028,
    "p8_ee_Zbb_ecm91":24931945,
    "p8_ee_Zcc_ecm91":19709554,
    "p8_ee_Zss_ecm91":20763861,
    "p8_ee_Zud_ecm91":10897650,
    "p8_ee_Ztautau_ecm91":3434,
    "p8_ee_Zmumu_ecm91":12,
    "p8_ee_Zee_ecm91":6,
}