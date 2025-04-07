import os
import glob
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg


def get_total_eff_post_bdt(df, 
                        cut= None,
                        verbose = True,
                        eventsProcessed_dict = cfg.eventsProcessed):#cut in string form
    """
    Function to get the raw efficiencies after applying the BDT cut.
    """

    efficienies = {}
    efficiencies_err = {}
    N_remaining = {}

    
    for sample in df["decay"].unique():
        df_decay = df[df["decay"] == sample]
        eventsProcessed = eventsProcessed_dict[sample]

        if cut is not None:
            N_post = len(df_decay.copy().query(cut))
        
        else:
            N_post = len(df_decay)
           
        total_efficiency = N_post/eventsProcessed
        efficienies[sample] = total_efficiency
        N_remaining[sample] = N_post
        
        
        if verbose:
            print(f"Sample: {sample}")
            print(f"eventsProcessed: {eventsProcessed}") 
            print(f"N_post: {N_post}")
            print(f"total_efficiency: {total_efficiency}")


        # calculating error using bayesian error formula See <https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf>
        # Variance in an efficiency k/n is (k+1)(k+2)/(n+2)(n+3) - (k+1)^2/(n+2)^2
        var = ((N_post+1)*(N_post+2))/((eventsProcessed+2)*(eventsProcessed+3)) - ((N_post+1)/(eventsProcessed+2))**2
        error = np.sqrt(var)
        efficiencies_err[sample] = error
        if verbose:
            print(f"efficiency error: {error}")

    return efficienies, efficiencies_err, N_remaining



def get_n_expected(efficiencies, efficiencies_err, signal_bf=1e-6):
    print('Note: error on n_expected is currently only from efficiency (assuming that dominant)')
    
    # Dict to store output
    n_expect = {}
    n_err={}

    
    # COMPUTING EXPECTATION
    for sample in efficiencies.keys():
        bfs_val = cfg.branching_fractions[sample][0]
        eff_val = efficiencies[sample]
        eff_err_val = efficiencies_err[sample]
        if eff_val >0:
            frac_eff_err = eff_err_val/eff_val # for now assuming that errors from efficiency are the ones that dominate
        else:
            frac_eff_err = 0

        
        num = 6e12*bfs_val*eff_val

        if sample in cfg.sample_allocations['combined_signal']:
            num *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf

        num_err = num*frac_eff_err # nb. for now just includes the efficiency error

        n_expect[sample] = num
        n_err[sample] = num_err

    return n_expect, n_err

        