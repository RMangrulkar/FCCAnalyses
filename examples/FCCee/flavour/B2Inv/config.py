# config.py
# This file contains all of the important configuration options used throughtout the analysis
# This file, along with all other analysis scripts, are expected to be in the `FCCAnalysesPath` directory
# Also contains branching fractions, cut efficiencies, etc which are manually filled in for now
import os

# MANDATORY ----> replace the default string with the path to the B2Inv directory in the FCCAnalyses repo
FCCAnalysesPath = "/r02/lhcb/rrm42/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/"
FCCAnalysesPath = os.path.abspath(FCCAnalysesPath)

##############################
## CONFIG DICTS
##############################
# processList to pass to `fccanalysis run`
processList = {
    # Size of winter2023 samples in /eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/:
    # p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu == 13G
    # p8_ee_Zbb_ecm91                == 3.3T
    # p8_ee_Zcc_ecm91                == 3.5T
    # p8_ee_Zss_ecm91                == 3.3T
    # p8_ee_Zud_ecm91                == 3.3T
    "stage1_training": {  # ~2G or ~500k events per sample
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": {"fraction": 0.3, "chunks": 10},
        "p8_ee_Zbb_ecm91": {"fraction": 0.01, "chunks": 50},
        "p8_ee_Zcc_ecm91": {"fraction": 0.01, "chunks": 50},
        "p8_ee_Zss_ecm91": {"fraction": 0.01, "chunks": 50},
        "p8_ee_Zud_ecm91": {"fraction": 0.01, "chunks": 50},
    },
    "stage1": {  # ~10G or 2M events per sample after preselection and a loose BDT1 cut
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": {"fraction": 1, "chunks": 1000},
        "p8_ee_Zbb_ecm91": {"fraction": 0.5, "chunks": 3000},
        "p8_ee_Zcc_ecm91": {"fraction": 0.5, "chunks": 3000},
        "p8_ee_Zss_ecm91": {"fraction": 0.5, "chunks": 3000},
        "p8_ee_Zud_ecm91": {"fraction": 0.5, "chunks": 3000},
    },
    "stage2_training": {
    },
    "stage2": {
    },
}

# Default options to pass to `fccanalysis run`
fccana_opts = {
    "prodTag":   "FCCee/winter2023/IDEA",
    "outputDir": {
        "stage1_training": os.path.join(FCCAnalysesPath, "outputs/stage1_training/"),
        "stage1":          os.path.join(FCCAnalysesPath, "outputs/stage1/"),
        "stage2_training": os.path.join(FCCAnalysesPath, "outputs/stage1/"),  # By default stage2 trained on output of stage1
        "stage2":          os.path.join(FCCAnalysesPath, "outputs/stage2/"),
    },
    "testFile": {
        "Bs": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu/events_026683563.root",
        "bb": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91/events_000083138.root",
        "cc": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zcc_ecm91/events_000046867.root",
        "ss": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zss_ecm91/events_000099129.root",
        "ud": "root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zud_ecm91/events_000071896.root",
    },
    "analysisName":   "",
    "nCPUs":          8,
    "runBatch":       True,
    "batchQueue":     "workday",
    "compGroup":      "group_u_FCC.local_gen",
    "yamlPath":       os.path.join(FCCAnalysesPath, "B2Inv.yaml"),  # Path to the YAML file containing feature names
    "outBranchList0": "stage0-vars",  # key in the yaml file that gives stage0 branches
    "outBranchList1": "stage1-vars",  # ---- " ---- stage1 branches
    "outBranchList2": "stage2-vars",  # ---- " ---- stage2 branches
}

# TMVA options
bdt1_opts = {
    "training":           True,                  # True == stage1 does not use BDT1
    "inputPath":          fccana_opts['outputDir']['stage1_training'],
    "outputPath":         os.path.join(FCCAnalysesPath, "outputs/bdt1out/"),
    "jsonPath":           os.path.join(FCCAnalysesPath, "outputs/bdt1out/bdt1.json"),
    "mvaPath":            os.path.join(FCCAnalysesPath, "outputs/bdt1out/tmva1.root"),
    "mvaRBDTName":        "bdt",                 # Name of the TMVA TObject in the .root file
    "mvaCut":             0.2,
    "mvaBranchList":      "bdt1-training-vars",  # key in the yaml file pointing to the feature list
    "efficiencyKey":      "presel",              # efficiencies used to calculate sample weights
    "optHyperParamsFile": os.path.join(FCCAnalysesPath, "outputs/bdt1out/best_params_bdt1.yaml"),
}

# BDT2 does not use BDT1 as a feature
bdt2_opts = {
    "training":           True,
    "inputPath":          fccana_opts['outputDir']['stage2_training'],
    "outputPath":         os.path.join(FCCAnalysesPath, "outputs/bdt2out/"),
    "jsonPath":           os.path.join(FCCAnalysesPath, "outputs/bdt2out/bdt2.json"),
    "mvaPath":            os.path.join(FCCAnalysesPath, "outputs/bdt2out/tmva2.root"),
    "mvaRBDTName":        "bdt",
    "mvaCut":             0.2,
    "mvaBranchList":      "bdt2-training-vars",
    "efficiencyKey":      "presel+bdt1>0.2",     # Efficiency key to use to calculate weights
    "optHyperParamsFile": os.path.join(FCCAnalysesPath, "outputs/bdt2out/best_params_bdt2.yaml"),
}

# bdtComb is essentially BDT2 which also uses BDT1 score as a feature
bdtComb_opts = {
    "training":           True,
    "inputPath":          fccana_opts['outputDir']['stage2_training'],
    "outputPath":         os.path.join(FCCAnalysesPath, "outputs/bdtCombout/"),
    "jsonPath":           os.path.join(FCCAnalysesPath, "outputs/bdtCombout/bdtComb.json"),
    "mvaPath":            os.path.join(FCCAnalysesPath, "outputs/bdtCombout/tmvaComb.root"),
    "mvaRBDTName":        "bdt",
    "mvaCut":             0.2,
    "mvaBranchList":      "bdtComb-training-vars",
    "efficiencyKey":      "presel+bdt1>0.2",
    "optHyperParamsFile": os.path.join(FCCAnalysesPath, "outputs/bdtCombout/best_params_bdtComb.yaml"),
}

# Options for post stage2 analysis (efficiency maps, maximising FOM, final limit on bf etc)
poststage2_opts = {
    "inputPath":  fccana_opts['outputDir']['stage2'],
    "outputPath": os.path.join(FCCAnalysesPath, "outputs/post_stage2"),
}

##############################
## SAMPLE OPTIONS
##############################
samples = [
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu",
    "p8_ee_Zbb_ecm91",
    "p8_ee_Zcc_ecm91",
    "p8_ee_Zss_ecm91",
    "p8_ee_Zud_ecm91",
]

sample_allocations = {
    "signal":     ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu"],
    "background": ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91", "p8_ee_Zss_ecm91", "p8_ee_Zud_ecm91"],
    "bb only":    ["p8_ee_Zbb_ecm91"],
}

sample_shorthand = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": "Bs2NuNu",
    "p8_ee_Zbb_ecm91":                "Z2bb",
    "p8_ee_Zcc_ecm91":                "Z2cc",
    "p8_ee_Zss_ecm91":                "Z2ss",
    "p8_ee_Zud_ecm91":                "Z2ud",
}

titles = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": r"$B_s^0 \to \nu \bar{\nu}$",
    "p8_ee_Zbb_ecm91": r"$Z \to b \bar{b}$",
    "p8_ee_Zcc_ecm91": r"$Z \to c \bar{c}$",
    "p8_ee_Zss_ecm91": r"$Z \to s \bar{s}$",
    "p8_ee_Zud_ecm91": r"$Z \to q \bar{q}$, $q \in [u,d]$",
}

##############################
## NUMERICAL DATA
##############################
# from PDG -> B production fractions
prod_frac = {
    "Bu": 0.43,
    "Bd": 0.43,
    "Bs": 0.096,
    "Lb": 0.037,
    "Bc": 0.0004,
    # Actually, use the sample name to make integration with Bd2NuNu easier
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (0.096, 0),
}


# from PDG (value, error)
# Z branching fractions
# Z->ss = (Z->dd+ss+bb)/3 * 3 - Z->bb / 2
# Z->uu/cc = 2 * (11.6 +/- 0.6) = 23.2 +/- 1.2
# Z->dd/ss/bb = 3 * (15.6 +/- 0.4) = 46.8 +/- 1.2
branching_fractions = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (1, 0),  # a dummy value
    "p8_ee_Zbb_ecm91": (0.1512, 0.0005),
    "p8_ee_Zcc_ecm91": (0.1203, 0.0021),
    "p8_ee_Zss_ecm91": (0.1584, 0.0060),
    "p8_ee_Zud_ecm91": (0.2701, 0.0136),
}

mass_Z = 91.188  # Ecm used in the winter2023 samples
EVT_hemisEmin_e_withpresel_min = 0.5*mass_Z - 40
EVT_hemisEmin_e_withpresel_max = 0.5*mass_Z

efficiencies = {
    "unity": {
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 1,
        "p8_ee_Zbb_ecm91": 1,
        "p8_ee_Zcc_ecm91": 1,
        "p8_ee_Zss_ecm91": 1,
        "p8_ee_Zud_ecm91": 1,
    },
    "presel": {
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (0.9315, 0.0003),
        "p8_ee_Zbb_ecm91": (0.08440, 0.00013),
        "p8_ee_Zcc_ecm91": (0.08470, 0.00012),
        "p8_ee_Zss_ecm91": (0.1144,   0.0001),
        "p8_ee_Zud_ecm91": (0.08720, 0.00013),
    },
    "presel+bdt1>0.2": {
        "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": (0.8851, 0.0002),
        "p8_ee_Zbb_ecm91": (0.008701, 0.000006),
        "p8_ee_Zcc_ecm91": (0.005605, 0.000005),
        "p8_ee_Zss_ecm91": (0.007871, 0.000006),
        "p8_ee_Zud_ecm91": (0.002103, 0.000003),
    },
}
