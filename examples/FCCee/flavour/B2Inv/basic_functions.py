import os
import glob
import sys
import gc
import ROOT
import math
import numpy as np
import pandas as pd
from tabulate import tabulate
from yaml import safe_load, YAMLError
from xgboost import XGBClassifier


ROOT.EnableImplicitMT()

##########################################################
# function to retrive lists from yaml and check file paths
###########################################################
# Return list of variables to use in the bdts as a python list
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


def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# Function to load the BDT model from a JSON file
def load_bdt_model_sklearn(json_path):
    bdt_model = XGBClassifier()
    bdt_model.load_model(json_path)
    return bdt_model
