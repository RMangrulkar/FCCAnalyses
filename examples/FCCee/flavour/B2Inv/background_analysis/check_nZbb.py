import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tabulate import tabulate
import matplotlib.pyplot as plt


import config as cfg
import bdt_lh_cut_opt_significance
from bdt_lh_cut_opt_significance import flatten_list, plot_N, plot_interpolted_effs
import basic_functions
import bdt_lh_cut_opt_significance_exclusive_backgrounds

samples_allocations_all = ["hadronic_background"] #["derived_B2lnu_samples"]#
samples_all = flatten_list([cfg.sample_allocations[component] for component in samples_allocations_all])

#Load dataframe with bdtlh version applied
for sample in samples_all:
    print(sample)
    
    #inclusive backgrounds and signal samples
    dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
    folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut'
    data = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))

    #add extra rows so can work in range0.9-1
    data = data.assign(
        P_signal=data['bdt_score_2'],
        P_not_heavy=1 - data['bdt_score_1'],
        P_not_light=1 - data['bdt_score_0'],
    )

    #add any extra cuts need here###########################
    data = data.query('EVT_hemisEmax_n>10') #veto on taus
    #for BF = BF = 7.967265920370373e-09 cuts as below
    data = data.query('P_not_heavy>0.999598') #BDT_cut
    data = data.query('P_not_light>0.999623')#

    print(len(data))
