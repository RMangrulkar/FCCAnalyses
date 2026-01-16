import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tabulate import tabulate
import matplotlib.pyplot as plt

import basic_functions

path_for_interpolated_dicts_0999 =  'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "interpolated_N_remaining_dictionary_Tau2HNu_only"), "rb") as dill_file:
    interp_N_dict_Tau2HNu = dill.load(dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "N_remaining_dictionary_Tau2HNu_only"), "rb") as dill_file:
    N_dict_Tau2HNu = dill.load(dill_file)


with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "interpolated_N_remaining_dictionary_no_tau2hnu"), "rb") as dill_file:
    interp_N_dict_noTau2HNu = dill.load(dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999), "N_remaining_dictionary_no_tau2hnu"), "rb") as dill_file:
    N_dict_noTau2HNu = dill.load(dill_file)

interp_N_dict = interp_N_dict_Tau2HNu | interp_N_dict_noTau2HNu
N_dict = N_dict_Tau2HNu | N_dict_noTau2HNu


#print(interp_N_dict_Tau2HNu)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999),"N_remaining_dictionary_full"), "wb") as dill_file:
    dill.dump(N_dict, dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts_0999),"interpolated_N_remaining_dictionary_full"), "wb") as dill_file:
    dill.dump(interp_N_dict, dill_file)
