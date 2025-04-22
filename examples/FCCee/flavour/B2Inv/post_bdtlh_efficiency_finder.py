import os
import glob
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg

#####################################################################
## functions for if have multiple sets of cuts (ie. multiple bins) ##
#####################################################################

def get_eff_from_nMC_list(N_dict_MC,eventsProcessed_dict = cfg.eventsProcessed):
#N_dict_MC - keys are samples and each value is a list of the number of MC events in each bin after cuts 
    efficienies = {}
    efficiencies_err = {}
    
    for decay in N_dict_MC.keys():
        eventsProcessed = eventsProcessed_dict[decay]
        N_post = N_dict_MC[decay]
        
        #calc efficiency
        total_efficiency = N_post/eventsProcessed
        efficienies[decay] = total_efficiency
        #print(f"efficiency: {total_efficiency}")
    
        # calculating error using bayesian error formula See <https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf>
        # Variance in an efficiency k/n is (k+1)(k+2)/(n+2)(n+3) - (k+1)^2/(n+2)^2
        var = ((N_post+1)*(N_post+2))/((eventsProcessed+2)*(eventsProcessed+3)) - ((N_post+1)/(eventsProcessed+2))**2
        error = np.sqrt(var)
        efficiencies_err[decay] = error
        #print(f"efficiency error: {error}")
    
    return efficienies, efficiencies_err, N_dict_MC



def get_n_expected_components(efficiencies, efficiencies_err, signal_bf=1e-6): #set up to take efficiencies which is a disctionary of arrays or dict of floats
    
    # Dict to store output
    per_sample_n_expect_dict = {}
    per_sample_frac_eff_err={}
    per_sample_frac_BFZbb_err={}
    
    # COMPUTING EXPECTATION
    for sample in efficiencies.keys():
        N_z = cfg.N_z
        bfs_val = cfg.branching_fractions[sample][0] #value=1 for signal
        bfs_err = cfg.branching_fractions[sample][1] #value=0 for signal modes
        
        eff_val = efficiencies[sample]
        eff_err_val = efficiencies_err[sample]
        
        frac_eff_err = np.where(eff_val != 0, eff_err_val / eff_val, 0) 

        num = N_z*bfs_val*eff_val

        if sample in cfg.sample_allocations['combined_signal']:
            num *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf

        frac_BFZbb_err = bfs_err/bfs_val #0 for signal

        if sample in cfg.sample_allocations['combined_signal']:
            frac_BFZbb_err = cfg.branching_fractions['p8_ee_Zbb_ecm91'][1]/cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]

        per_sample_n_expect_dict[sample] = num
        per_sample_frac_eff_err[sample] = frac_eff_err
        per_sample_frac_BFZbb_err[sample] = frac_BFZbb_err

    return per_sample_n_expect_dict, per_sample_frac_eff_err, per_sample_frac_BFZbb_err        

#combine compoenents into S and B and corresponding eror
def get_total_SB(per_sample_n_expect_dict, per_sample_frac_eff_err, per_sample_frac_BFZbb_err, incl_BFZbb_err=True):
    B= np.sum(np.stack([per_sample_n_expect_dict[k] for k in cfg.sample_allocations['hadronic_background']]), axis=0)
    S= np.sum(np.stack([per_sample_n_expect_dict[k] for k in cfg.sample_allocations['combined_signal']]), axis=0)

    B_eff_var= np.sum(np.stack([(per_sample_frac_eff_err[k]*per_sample_n_expect_dict[k])**2 for k in cfg.sample_allocations['hadronic_background']]), axis=0)
    S_eff_var= np.sum(np.stack([(per_sample_frac_eff_err[k]*per_sample_n_expect_dict[k])**2 for k in cfg.sample_allocations['combined_signal']]), axis=0)

    B_BF_var = np.sum(np.stack([(per_sample_frac_BFZbb_err[k]*per_sample_n_expect_dict[k])**2 for k in cfg.sample_allocations['hadronic_background']]), axis=0)
    S_BF_var = np.sum(np.stack([(per_sample_frac_BFZbb_err[k]*per_sample_n_expect_dict[k]) for k in cfg.sample_allocations['combined_signal']]), axis=0)**2

    if incl_BFZbb_err == True:
        B_err = np.sqrt(B_eff_var+B_BF_var)
        S_err = np.sqrt(S_eff_var+S_BF_var)

    else:
        B_err = np.sqrt(B_eff_var)
        S_err = np.sqrt(S_eff_var)

    return S, B, S_err, B_err






#######################################
## soon to be legacy for comparison ### - To replace with above for one bin case in cut opt script!!
#######################################
################################################################
## functions for if only have single set of cuts (ie. one bin)##
################################################################

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



def get_n_expected(efficiencies, efficiencies_err, signal_bf=1e-6, calc_BFZbb_err=False): #set up to take efficiencies which is a disctionary of floats
    '''
    function to return the expected number of events for each sample and the efficiency error on that number'''
    
    
    
    #print('Note: error on n_expected is currently only from efficiency (assuming that dominant)')
    
    # Dict to store output
    n_expect_dict = {}
    n_err_dict={}
    BFZbb_err_dict={}
    BFZbb_err_dict_components={}

    
    # COMPUTING EXPECTATION
    for sample in efficiencies.keys():
        bfs_val = cfg.branching_fractions[sample][0] #value=1 for signal
        bfs_err = cfg.branching_fractions[sample][1] #value=0 for signal modes
        eff_val = efficiencies[sample]
        eff_err_val = efficiencies_err[sample]
        N_z = cfg.N_z
        if eff_val >0:
            frac_eff_err = eff_err_val/eff_val # for now assuming that errors from efficiency are the ones that dominate
        else:
            frac_eff_err = 0

        
        num = N_z*bfs_val*eff_val

        if sample in cfg.sample_allocations['combined_signal']:
            num *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf

        num_err = num*frac_eff_err # nb. for now just includes the efficiency error

        BFZbb_err = num*bfs_err/bfs_val #0 for signal

        if sample in cfg.sample_allocations['combined_signal']:
            BFZbb_err = num * cfg.branching_fractions['p8_ee_Zbb_ecm91'][1]/cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]

        n_expect_dict[sample] = num
        n_err_dict[sample] = num_err
        BFZbb_err_dict[sample] = BFZbb_err

    if calc_BFZbb_err == True:
    #due to different error formula for signal and background for BF error (as in signal Zbb common to both terms whilst for B Zqq different for each term)
    # make BF error per component (ie. S and B)
        print(efficiencies.keys())
        if all(sample in efficiencies.keys() for sample in cfg.sample_allocations['combined_signal']):
            BFZbb_err_dict_components['combined_signal'] = sum([BFZbb_err_dict[sample] for sample in cfg.sample_allocations['combined_signal']])
        if all(sample in efficiencies.keys() for sample in cfg.sample_allocations['hadronic_background']):
            BFZbb_err_dict_components['hadronic_background'] = np.sqrt(sum([BFZbb_err_dict[sample]**2 for sample in cfg.sample_allocations['hadronic_background']]))

        return n_expect_dict, n_err_dict, BFZbb_err_dict_components
    
    else:
        return n_expect_dict, n_err_dict
    