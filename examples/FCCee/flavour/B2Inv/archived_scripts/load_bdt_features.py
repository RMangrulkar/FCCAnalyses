import xgboost as xgb
import numpy as np
import sys
from yaml import safe_load, YAMLError, dump
import config as cfg
import os

# Path to config.py and variable_plotter.py
sys.path.append("/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/")


# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)

    return bdtvars

def check_inputpath(inputpath):
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"{inputpath} does not exist")
    return inputpath

yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])
bdt1vars= vars_fromyaml(yamlpath, cfg.bdt1_opts['mvaBranchList'])
bdt2vars= vars_fromyaml(yamlpath, cfg.bdt2_opts['mvaBranchList'])

# Load the model
model1=xgb.XGBClassifier()
model1.load_model('/r02/lhcb/rrm42/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/bdt1out/bdt1.json')
feature_importance_dict1 = dict(zip(bdt1vars, model1.feature_importances_))
# Sort by importance
sorted_importance1 = sorted(feature_importance_dict1.items(), key=lambda kv: kv[1], reverse=True)

# Load the model
model2=xgb.XGBClassifier()
model2.load_model('/r02/lhcb/rrm42/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/bdt2out/bdt2.json')
feature_importance_dict2 = dict(zip(bdt2vars, model2.feature_importances_))
# Sort by importance 
sorted_importance2 = sorted(feature_importance_dict2.items(), key=lambda kv: kv[1], reverse=True)


# Display feature importances
print("Feature Importances for BDT1:")
for feature, importance in sorted_importance1:
    print(f"{feature} {importance} \n")



