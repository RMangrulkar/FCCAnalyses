import os
import glob
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efficiency_tools import efficiency_finder
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
        #calc efficiency - deals with list separately to if just input a number
        if isinstance(N_post, (int, float)):
            total_efficiency, error = efficiency_finder.efficiency_calc(eventsProcessed, N_post)
        else:
            results = [efficiency_finder.efficiency_calc(eventsProcessed, val) for val in N_post.flat]
            total_efficiency, error = zip(*results)

            total_efficiency = np.array(total_efficiency).reshape(N_post.shape)
            error = np.array(error).reshape(N_post.shape)
                        
            efficienies[decay] = total_efficiency
            efficiencies_err[decay] = error

            #print(f"efficiency: {total_efficiency}")
            #print(f"efficiency error: {error}")
    
    return efficienies, efficiencies_err, N_dict_MC



def get_n_expected_components(efficiencies, efficiencies_err, signal_bf=1e-6, model_all_1prong_tau=False): #set up to take efficiencies which is a disctionary of arrays or dict of floats
    
    # Dict to store output
    per_sample_n_expect_dict = {}
    per_sample_eff_err={}
    per_sample_novereff = {}
    per_sample_frac_BFZbb_err={}
    per_sample_frac_fk_err={}
    
    # COMPUTING EXPECTATION
    for sample in efficiencies.keys():
        N_z = cfg.N_z

        if model_all_1prong_tau==True:
            if sample in ("p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu"):
                bfs_val = cfg.branching_fractions[f"{sample}_modelling_oneprong"][0] #value=1 for signal
                bfs_err = cfg.branching_fractions[f"{sample}_modelling_oneprong"][1]

            else:
                bfs_val = cfg.branching_fractions[sample][0] #value=1 for signal
                bfs_err = cfg.branching_fractions[sample][1] #value=0 for signal modes
        else:
            bfs_val = cfg.branching_fractions[sample][0] #value=1 for signal
            bfs_err = cfg.branching_fractions[sample][1] #value=0 for signal modes
        
        eff_val = efficiencies[sample]
        eff_err_val = efficiencies_err[sample]
        
        #frac_eff_err = np.where(eff_val != 0, eff_err_val / eff_val, 0) #cant use without setting error at n_exp = 0 to 0 - instead keep efficiency error and N/efficiency

        num = N_z*bfs_val*eff_val
        num_overeff = N_z*bfs_val

        frac_BFZbb_err = bfs_err/bfs_val #0 for signal
        frac_fk_err = 0 #not included in background

        if sample in cfg.sample_allocations['combined_signal']:
            num *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf
            num_overeff *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf
            
            frac_BFZbb_err = cfg.branching_fractions['p8_ee_Zbb_ecm91'][1]/cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]
            frac_fk_err = cfg.prod_frac[sample][1]/cfg.prod_frac[sample][0]

        elif sample in cfg.exclusive_backgrounds:
            num *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]
            num_overeff *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]

            frac_BFZbb_err *= cfg.branching_fractions['p8_ee_Zbb_ecm91'][1]/cfg.branching_fractions['p8_ee_Zbb_ecm91'][0] #need *= here as want error from both Zbb and B decay BFs
            frac_fk_err = cfg.prod_frac[sample][1]/cfg.prod_frac[sample][0]

        per_sample_n_expect_dict[sample] = num
        per_sample_eff_err[sample] = eff_err_val
        per_sample_novereff[sample] = num_overeff
        per_sample_frac_BFZbb_err[sample] = frac_BFZbb_err
        per_sample_frac_fk_err[sample] = frac_fk_err

    return per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err         


#combine compoenents into S and B and corresponding absolute (not fractional) error
def get_total_SB(per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err, incl_other_syst=True, exclusive_background_samples = None,individual_signal_contributions=False):
    
    '''
    Turns number from each decay into total S and B expectations with errors (per bin)
    "exclusive_background_samples": must be a LIST of exclusive additional background samples to incluives in hadronic_background
    '''
    backgrounds = [i for i in per_sample_n_expect_dict.keys() if i in cfg.background_samples]
    signals = [i for i in per_sample_n_expect_dict.keys() if i in cfg.signal_samples]
    
    S= np.sum(np.stack([per_sample_n_expect_dict[k] for k in signals]), axis=0)
    B= np.sum(np.stack([per_sample_n_expect_dict[k] for k in backgrounds]), axis=0)

    S_eff_var= np.sum(np.stack([(per_sample_eff_err[k]*per_sample_novereff[k])**2 for k in signals]), axis=0)
    B_eff_var= np.sum(np.stack([(per_sample_eff_err[k]*per_sample_novereff[k])**2 for k in backgrounds]), axis=0)

    S_BF_var = np.sum(np.stack([(per_sample_frac_BFZbb_err[k]*per_sample_n_expect_dict[k]) for k in signals]), axis=0)**2
    B_BF_var = np.sum(np.stack([(per_sample_frac_BFZbb_err[k]*per_sample_n_expect_dict[k])**2 for k in backgrounds]), axis=0)

    S_fk_var = np.sum(np.stack([(per_sample_frac_fk_err[k]*per_sample_n_expect_dict[k])**2 for k in signals]), axis=0)
    B_fk_var = np.zeros_like(S_fk_var)
         

    if exclusive_background_samples is not None:#for exclusive backgrounds need to include error from hadronisation fractiosn
        #check samples used are exclusive backgrounds consistent with list in config
        if not set(exclusive_background_samples).issubset(cfg.exclusive_backgrounds):
            raise ValueError("'exclusive_background_samples' given are not consistent with list in config - Ensure only exclusive bkg samples are input here and update config.excluisve_backgrounds list")
        else: 
            additionalB_fk_var = np.sum(np.stack([(per_sample_frac_fk_err[k]*per_sample_n_expect_dict[k])**2 for k in exclusive_background_samples]), axis=0)
            B_fk_var= additionalB_fk_var


    if incl_other_syst == True:
        if exclusive_background_samples is not None:
            B_err = np.sqrt(B_eff_var+B_BF_var+B_fk_var)
        else:
            B_err = np.sqrt(B_eff_var+B_BF_var)

        S_err = np.sqrt(S_eff_var+S_BF_var+S_fk_var)

    else:
        B_err = np.sqrt(B_eff_var)
        S_err = np.sqrt(S_eff_var)

    if individual_signal_contributions ==True:
        S_dict = {}
        S_eff_var_dict= {}
        S_BF_var_dict = {}
        S_fk_var_dict = {}
        S_err_dict = {}
        for sample in cfg.sample_allocations['combined_signal']:
            S_dict[sample]= per_sample_n_expect_dict[sample]
            S_eff_var_dict[sample]=(per_sample_eff_err[sample]*per_sample_novereff[sample])**2 
            S_BF_var_dict[sample] = (per_sample_frac_BFZbb_err[sample]*per_sample_n_expect_dict[sample])**2
            S_fk_var_dict[sample] = (per_sample_frac_fk_err[sample]*per_sample_n_expect_dict[sample])**2 

            if incl_other_syst == True:
                S_err_dict[sample] = np.sqrt(S_eff_var_dict[sample]+S_BF_var_dict[sample]+S_fk_var_dict[sample])

            else:
                S_err_dict[sample]  = np.sqrt(S_eff_var_dict[sample])
        
        return S, B, S_err, B_err, S_dict, S_err_dict
    else: 
        return S, B, S_err, B_err






#######################################
## soon to be legacy for comparison ### - To replace with above for one bin case in cut opt script!!
#######################################
################################################################
## functions for if only have single set of cuts (ie. one bin)## - needed for post bdt variable plotter script
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
           
        total_efficiency, error = efficiency_finder.efficiency_calc(eventsProcessed, N_post)
        
        efficienies[sample] = total_efficiency
        N_remaining[sample] = N_post
        efficiencies_err[sample] = error
        
        
        if verbose:
            print(f"Sample: {sample}")
            print(f"eventsProcessed: {eventsProcessed}") 
            print(f"N_post: {N_post}")
            print(f"total_efficiency: {total_efficiency}")
            print(f"efficiency error: {error}")

    return efficienies, efficiencies_err, N_remaining



def get_n_expected(efficiencies, efficiencies_err, signal_bf=1e-6, calc_BFZbb_err=False, model_all_1prong_tau = False): #set up to take efficiencies which is a disctionary of floats
    
    #function to return the expected number of events for each sample and the efficiency error on that number
    
    
    
    #print('Note: error on n_expected is currently only from efficiency (assuming that dominant)')
    
    # Dict to store output
    n_expect_dict = {}
    n_err_dict={}
    BFZbb_err_dict={}
    BFZbb_err_dict_components={}

    
    # COMPUTING EXPECTATION
    for sample in efficiencies.keys():
        if model_all_1prong_tau==True:
            if sample in ("p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu"):
                bfs_val = cfg.branching_fractions[f"{sample}_modelling_oneprong"][0] #value=1 for signal
                bfs_err = cfg.branching_fractions[f"{sample}_modelling_oneprong"][1]

            else:
                bfs_val = cfg.branching_fractions[sample][0] #value=1 for signal
                bfs_err = cfg.branching_fractions[sample][1] #value=0 for signal modes
        else:
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
