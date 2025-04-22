import os
import glob
import sys
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from yaml import safe_load, YAMLError, dump
from tabulate import tabulate
from scipy.interpolate import RectBivariateSpline
from scipy.stats import poisson, norm
from iminuit import Minuit

import config as cfg 
import post_bdtlh_efficiency_finder as eff_finder
plt.style.use('fcc.mplstyle')

# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars

def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath

#function to turn #sigma to CL
def sigma_to_percentage(sigma):
    # Calculate the percentage
    percentage = norm.cdf(sigma) * 100
    return percentage

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


#plan: create and save map of N and N interpolated
#can then use to calculate efficiency and error properly!

def create_N_map(df,lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, smoothing=False, kx=2,ky=2, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/'): #df should be full data for final result
    '''
    function to extract N remaining for various 2D BDT cuts and interpolate it to create a smooth map
    '''
    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)
    
    #define lists need etc
    s_values_dict={}
    N_dict={}
    eff_dict={}
    err_dict={}
    interp_N_dict={}

    print('Creating efficiency map for decay:')
    
    for decay in df["decay"].unique():
        #to remove when want to include Bd too#########################
        #if decay =='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu':
        #    continue
        ###############################################################
        print(decay)

        N_df=np.zeros((len(lsearch), len(hsearch)))
        eff_df=np.zeros((len(lsearch), len(hsearch)))
        err_df=np.zeros((len(lsearch), len(hsearch)))
        
        subf = df[df["decay"]==decay]
        l_arr=[]
        h_arr=[]
    
        for l in np.arange(0,len(lsearch),1):
            for h in np.arange(0,len(hsearch),1):
                l_arr.append(lsearch[l])
                h_arr.append(hsearch[h])
                lh_cut = f"(P_not_heavy>{hsearch[h]})&(P_not_light>{lsearch[l]})"

                eff_lh, err_lh, N_remaining = eff_finder.get_total_eff_post_bdt(subf, cut= lh_cut, verbose = False, eventsProcessed_dict = cfg.eventsProcessed)
                N_df[l,h] = N_remaining[decay]
                eff_df[l,h] = eff_lh[decay] 
                err_df[l,h] = err_lh[decay]     
    
        N_dict[decay] = N_df
        eff_dict[decay] = N_df
        err_dict[decay] = N_df

        if smoothing == True:
            s_value = np.sum(N_df) #sum of variances
            s_values_dict[decay] = s_value

        else:
            s_value = 0
            s_values_dict[decay] = s_value
   

        interp_N = RectBivariateSpline(lsearch, hsearch, N_df,kx=kx, ky=ky, s=s_value)
        interp_N_dict[decay] = interp_N

    print(f'S-values using for smoothing:{s_values_dict}')
    
    with open(os.path.join(set_outputpath(save_path),"N_remaining_dictionary"), "wb") as dill_file:
        dill.dump(N_dict, dill_file)

    with open(os.path.join(set_outputpath(save_path),"interpolated_N_remaining_dictionary"), "wb") as dill_file:
        dill.dump(interp_N_dict, dill_file)
    
    return N_dict, interp_N_dict, s_values_dict, eff_dict, err_dict



def plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,slice=False, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/'):

    '''
    function to plot 2d N with and without interpolation to check smoothness as well as slices
    '''
    #define arrays want to calculate efficiency over
    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)

    lsearchinterp = np.linspace(*lrange,nlh*10) 
    hsearchinterp = np.linspace(*hrange,nlh*10)

    for decay in N_dict.keys():

        #to remove ee sa literally one MC event remaining
        if decay =='p8_ee_Zee_ecm91':
            continue

        #plotting raw N distribution
        plt.figure()
        plt.imshow(N_dict[decay], origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='Raw Number of surviving MC events')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearch), round(len(lsearch)/5))
        xtick_indices = np.arange(0, len(hsearch), round(len(hsearch)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearch[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearch[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'{cfg.titles[decay]}')
        plt.savefig(os.path.join(set_outputpath(save_path),f'N_{decay}.pdf'))

        #plotting interpolated N map

        
        #plotting raw N distribution
        plt.figure()
        gridded_interp = [[interp_N_dict[decay](lsearchinterp[l], hsearchinterp[h]).item() for h in range(len(hsearchinterp))] for l in range(len(lsearchinterp))]
        plt.imshow(gridded_interp, origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='Raw Number of surviving MC events')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearchinterp[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'{cfg.titles[decay]}')
        plt.savefig(os.path.join(set_outputpath(save_path),f'N_interp_{decay}.pdf'))



    if slice == True:
        i=6
        for decay in N_dict.keys():

            #to remove ee
            if decay =='p8_ee_Zee_ecm91':
                continue

        
            plt.figure()
            plt.plot(hsearch,N_dict[decay][round(nlh*i/10),:], label=r'N slice at 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
            N_interp_slice = [interp_N_dict[decay](lsearch[round(nlh*i/10)] , hsearchinterp[h]).item() for h in range(len(hsearchinterp))]
            plt.plot(hsearchinterp,N_interp_slice, label=r'Interpolated N slice at 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(heavy)')
            plt.ylabel(r'Number of MC events remaining')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_{decay}_heavy.pdf'))


            plt.figure()
            plt.plot(lsearch,N_dict[decay][:,round(nlh*i/10)], label=r'N slice at 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
            N_interp_slice = [interp_N_dict[decay](lsearchinterp[l], hsearch[round(nlh*i/10)]).item()  for l in range(len(lsearchinterp))]
            plt.plot(lsearchinterp,N_interp_slice, label=r'Interpolated N slice at 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(light)')
            plt.ylabel(r'Number of MC events remaining')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_{decay}_light.pdf'))



def interp_N_to_eff_err(interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh_plot=200,eventsProcessed_dict = cfg.eventsProcessed,savepath=None): #need lhranges to be within region of interp_N_dict
    '''
    function to extract eff and error from saved interpolated N remaining with given 2D BDT cuts
    '''

    #define arrays want to calculate interpolated efficiency over
    lsearchinterp = np.linspace(*lrange,nlh_plot) 
    hsearchinterp = np.linspace(*hrange,nlh_plot)

    interp_eff_dict = {}
    interp_eff_err_dict = {}
    
    for sample in interp_N_dict.keys():

        fine_grid_splined_eff =np.zeros((len(lsearchinterp), len(hsearchinterp)))
        fine_grid_splined_eff_err =np.zeros((len(lsearchinterp), len(hsearchinterp)))

        #retreive events processed per sample
        eventsProcessed = eventsProcessed_dict[sample]

        # evaluate efficiency and error at each point on array
        for l in np.arange(0,len(lsearchinterp),1):
            for h in np.arange(0,len(hsearchinterp),1):
                N_post = interp_N_dict[sample](lsearchinterp[l],hsearchinterp[h],grid=False)
                total_efficiency = N_post/eventsProcessed
                # calculating error using bayesian error formula See <https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf>
                # Variance in an efficiency k/n is (k+1)(k+2)/(n+2)(n+3) - (k+1)^2/(n+2)^2
                var = ((N_post+1)*(N_post+2))/((eventsProcessed+2)*(eventsProcessed+3)) - ((N_post+1)/(eventsProcessed+2))**2
                error = np.sqrt(var)

                fine_grid_splined_eff[l,h] = total_efficiency.item() 
                fine_grid_splined_eff_err[l,h] = error.item() 
        
        interp_eff_dict[sample] = fine_grid_splined_eff
        interp_eff_err_dict[sample] = fine_grid_splined_eff_err


    if savepath:

        with open(os.path.join(set_outputpath(savepath),"interpolated_eff_dictionary"), "wb") as dill_file:
            dill.dump(interp_eff_dict, dill_file)

        with open(os.path.join(set_outputpath(savepath),"interpolated_eff_err_dictionary"), "wb") as dill_file:
            dill.dump(interp_eff_err_dict, dill_file)


    return interp_eff_dict, interp_eff_err_dict


def raw_N_to_eff_err(N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,eventsProcessed_dict = cfg.eventsProcessed): #need lhranges to be within region of interp_N_dict
    '''
    function to extract eff and error from saved interpolated N remaining with given 2D BDT cuts
    '''
    #define arrays want to calculate efficiency over
    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)

    eff_dict = {}
    err_dict = {}
    
    for sample in N_dict.keys():

        eff =np.zeros((len(lsearch), len(hsearch)))
        eff_err =np.zeros((len(lsearch), len(hsearch)))

        #retreive events processed per sample
        eventsProcessed = eventsProcessed_dict[sample]

        # evaluate efficiency and error at each point on array
        for l in np.arange(0,len(lsearch),1):
            for h in np.arange(0,len(hsearch),1):
                N_post = N_dict[sample][l,h]
                total_efficiency = N_post/eventsProcessed
                # calculating error using bayesian error formula See <https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf>
                # Variance in an efficiency k/n is (k+1)(k+2)/(n+2)(n+3) - (k+1)^2/(n+2)^2
                var = ((N_post+1)*(N_post+2))/((eventsProcessed+2)*(eventsProcessed+3)) - ((N_post+1)/(eventsProcessed+2))**2
                error = np.sqrt(var)

                eff[l,h] = total_efficiency.item() 
                eff_err[l,h] = error.item() 
        
        eff_dict[sample] = eff
        err_dict[sample] = eff_err


    return  eff_dict, err_dict


def plot_interpolted_effs(interp_N_dict, N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, slice=False,save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto'):

    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)


    lsearchinterp = np.linspace(*lrange,(nlh-1)*10+1) 
    hsearchinterp = np.linspace(*hrange,(nlh-1)*10+1)


    # generate eff from N amp
    eff_dict, err_dict = raw_N_to_eff_err(N_dict, lrange=lrange ,hrange=hrange,nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed)
    interp_eff_dict, interp_eff_err_dict = interp_N_to_eff_err(interp_N_dict, lrange=lrange ,hrange=hrange,nlh_plot=(nlh-1)*10+1,eventsProcessed_dict = cfg.eventsProcessed,savepath=None)

    
    for decay in interp_eff_dict.keys():
        
        #to remove ee
        if decay =='p8_ee_Zee_ecm91':
            continue

        #plotting splined efficiencies themselves (finely gridded)
        plt.figure()
        plt.imshow(interp_eff_dict[decay], origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='efficiency')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearchinterp[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'{cfg.titles[decay]}')
        plt.savefig(os.path.join(set_outputpath(save_path),f'frac_efficiency_{decay}_interpN.pdf'))

        

        # plot fractional error in efficiency extracted from 
        plt.figure()
        plt.imshow(interp_eff_err_dict[decay]/interp_eff_dict[decay], origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='Fractional error on the efficiency')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearchinterp[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'{cfg.titles[decay]}')
        plt.savefig(os.path.join(set_outputpath(save_path),f'frac_eff_error_{decay}_interpN.pdf'))

        # plot  error in efficiency extracted from 
        plt.figure()
        plt.imshow(interp_eff_err_dict[decay], origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='Error on the efficiency')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearchinterp[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'{cfg.titles[decay]}')
        plt.savefig(os.path.join(set_outputpath(save_path),f'eff_error_{decay}_interpN.pdf'))

            

            
        if slice == True:
            i=6

            for decay in interp_eff_dict.keys():
                        #to remove ee
                if decay =='p8_ee_Zee_ecm91':
                    continue

        
                plt.figure()
                plt.errorbar(hsearch,eff_dict[decay][round(nlh*i/10),:], err_dict[decay][round(nlh*i/10),:],label=r'raw efficiency slice 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
                eff_interp_slice = interp_eff_dict[decay][round((nlh*i/10)*10),:]
                plt.plot(hsearchinterp,eff_interp_slice, label=r'interpolated efficiency slice 1-P(light) $>$'+f'{round(lsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(heavy)')
                plt.ylabel(r'Efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'efficinecy_{decay}_heavy.pdf'))


                plt.figure()
                plt.errorbar(lsearch,eff_dict[decay][:,round(nlh*i/10)], err_dict[decay][:,round(nlh*i/10)], label=r'raw efficiency slice 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
                eff_interp_slice = interp_eff_dict[decay][:, round((nlh*i/10)*10)]
                plt.plot(lsearchinterp,eff_interp_slice, label=r'interpolated efficiency slice 1-P(heavy) $>$'+f'{round(hsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(light)')
                plt.ylabel(r'Efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'efficinecy_{decay}_light.pdf'))

                plt.figure()
                plt.plot(hsearch,err_dict[decay][round(nlh*i/10),:]/eff_dict[decay][round(nlh*i/10),:], label=r'raw error slice 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
                frac_err_interp_slice = interp_eff_err_dict[decay][round((nlh*i/10)*10),:] /interp_eff_dict[decay][round((nlh*i/10)*10),:]
                plt.plot(hsearchinterp,frac_err_interp_slice, label=r'interpolated error slice  1-P(light) $>$'+f'{round(lsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(heavy)')
                plt.ylabel(r'Fractional error in the efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'frac_eff_error_{decay}_heavy.pdf'))


                plt.figure()
                plt.plot(lsearch,err_dict[decay][:,round(nlh*i/10)]/eff_dict[decay][:,round(nlh*i/10)], label=r'raw error slice 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
                frac_err_interp_slice = interp_eff_err_dict[decay][:, round((nlh*i/10)*10)]/interp_eff_dict[decay][:, round((nlh*i/10)*10)]
                plt.plot(lsearchinterp,frac_err_interp_slice, label=r'interpolated error slice 1-P(heavy) $>$'+f'{round(hsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(light)')
                plt.ylabel(r'Fractional error in the efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'frac_eff_error_{decay}_light.pdf'))

                plt.figure()
                plt.plot(hsearch,err_dict[decay][round(nlh*i/10),:], label=r'raw error slice 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
                err_interp_slice = interp_eff_err_dict[decay][round((nlh*i/10)*10),:] 
                plt.plot(hsearchinterp,err_interp_slice, label=r'interpolated error slice  1-P(light) $>$'+f'{round(lsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(heavy)')
                plt.ylabel(r'Error in the efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'eff_error_{decay}_heavy.pdf'))


                plt.figure()
                plt.plot(lsearch,err_dict[decay][:,round(nlh*i/10)], label=r'raw error slice 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
                err_interp_slice = interp_eff_err_dict[decay][:, round((nlh*i/10)*10)]
                plt.plot(lsearchinterp,err_interp_slice, label=r'interpolated error slice 1-P(heavy) $>$'+f'{round(hsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(light)')
                plt.ylabel(r'Error in the efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'eff_error_{decay}_light.pdf'))




  
def run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=1000, sig_BF=1e-7,incl_ZqqBFerror = True):
    
    lsearch = np.linspace(*lrange_plot,nlh) #need to be the same lrange and hrange as efficiency map was generated with 
    hsearch = np.linspace(*hrange_plot,nlh)

    #create efficiency deictionary from interpolated N
    interp_eff_dict, interp_eff_err_dict = interp_N_to_eff_err(interp_N_dict, lrange=lrange_plot ,hrange=hrange_plot,nlh_plot=nlh,eventsProcessed_dict = cfg.eventsProcessed,savepath=None)


    S_arr = np.zeros((len(lsearch), len(hsearch)))
    B_arr = np.zeros((len(lsearch), len(hsearch)))
    S_error_arr = np.zeros((len(lsearch), len(hsearch)))
    B_error_arr = np.zeros((len(lsearch), len(hsearch)))
    FOM = np.zeros((len(lsearch), len(hsearch)))
    err_FOM = np.zeros((len(lsearch), len(hsearch)))
    full_err_FOM = np.zeros((len(lsearch), len(hsearch)))
    full_S_error_arr = np.zeros((len(lsearch), len(hsearch)))
    full_B_error_arr = np.zeros((len(lsearch), len(hsearch)))


    print('Calculating S and B')
    
    ''' 
    # alternative method using post bdtlh efficiency finder script  probably less clear to follow but gives same reuslts :)
    for l in np.arange(0,len(lsearch), 1):
        for h in np.arange(0,len(hsearch), 1):

            lh_interp_eff_dict = {sample: interp_eff_dict[sample][l,h] for sample in interp_eff_dict.keys()}
            lh_interp_eff_err_dict = {sample: interp_eff_err_dict[sample][l,h] for sample in interp_eff_err_dict.keys()}
 
            lh_n_expect_dict, lh_n_err_dict, lh_BFZbb_err_dict_components = eff_finder.get_n_expected(lh_interp_eff_dict, lh_interp_eff_err_dict, signal_bf=sig_BF,  BFZbb_err=True)

            S = sum([lh_n_expect_dict[sample] for sample in cfg.sample_allocations["combined_signal"]])
            B = sum([lh_n_expect_dict[sample] for sample in cfg.sample_allocations["hadronic_background"]])
            
            S_arr[l,h]=S
            B_arr[l,h]=B

            var_S = sum([lh_n_err_dict[sample]**2 for sample in cfg.sample_allocations["combined_signal"]])
            var_B = sum([lh_n_err_dict[sample]**2 for sample in cfg.sample_allocations["hadronic_background"]])
            
            S_error_arr[l,h]=np.sqrt(var_S)
            B_error_arr[l,h]=np.sqrt(var_B)

            if S+B>0:
                FOM[l,h] = S/np.sqrt(S+B)
                #also calculate error in FOM itself from S,B error
                err_FOM[l,h] = np.sqrt(1/(4*(S+B)**3)*((2*B+S)**2*var_S + S**2*var_B))
            else:
                FOM[l,h] = 0
                err_FOM[l,h] = 0
            
            
            if incl_ZqqBFerror == True:

                var_S_BF = lh_BFZbb_err_dict_components["combined_signal"]**2
                var_B_BF = lh_BFZbb_err_dict_components["hadronic_background"]**2

                full_var_S = var_S + var_S_BF
                full_var_B = var_B + var_B_BF

                full_S_error_arr[l,h]=np.sqrt(full_var_S)
                full_B_error_arr[l,h]=np.sqrt(full_var_B)

                if S+B>0:
                    #also calculate error in FOM itself from S,B error
                    full_err_FOM[l,h] = np.sqrt(1/(4*(S+B)**3)*((2*B+S)**2*full_var_S + S**2*full_var_B))
                else:
                    FOM[l,h] = 0
                    full_err_FOM[l,h] = 0


    '''#Alternative method without using eff_finder script as cross check - proabably easier to follow
    
    #defining constants needed
    N_z = cfg.N_z
    k = 2 * N_z * cfg.branching_fractions["p8_ee_Zbb_ecm91"][0] * sig_BF # common part of signal expectation
    
    for l in np.arange(0,len(lsearch), 1):
        for h in np.arange(0,len(hsearch), 1):
            
            S = k * sum([cfg.prod_frac[decay][0]*interp_eff_dict[decay][l,h] for decay in cfg.sample_allocations["combined_signal"]])
            B = N_z* sum([cfg.branching_fractions[decay][0]*interp_eff_dict[decay][l,h] for decay in cfg.sample_allocations["hadronic_background"]])
            S_arr[l,h]=S
            B_arr[l,h]=B

            #also calculating S and B errors
            var_S = k**2 * sum([(cfg.prod_frac[decay][0]*interp_eff_err_dict[decay][l,h])**2 for decay in cfg.sample_allocations["combined_signal"]])
            var_B = N_z**2 * sum([(cfg.branching_fractions[decay][0]*interp_eff_err_dict[decay][l,h])**2 for decay in cfg.sample_allocations["hadronic_background"]])
           
            sigma_S = np.sqrt(var_S)
            sigma_B = np.sqrt(var_B)
            
            S_error_arr[l,h]=sigma_S
            B_error_arr[l,h]=sigma_B

            if S+B>0:
                FOM[l,h] = S/np.sqrt(S+B)
                #also calculate error in FOM itself from S,B error
                err_FOM[l,h] = np.sqrt(1/(4*(S+B)**3)*((2*B+S)**2*sigma_S**2 + S**2*sigma_B**2))
            else:
                FOM[l,h] = 0
                err_FOM[l,h] = 0

            
            if incl_ZqqBFerror == True:

                full_var_S = var_S + ((S/cfg.branching_fractions["p8_ee_Zbb_ecm91"][0])*cfg.branching_fractions["p8_ee_Zbb_ecm91"][1])**2
                full_var_B = var_B + N_z**2 * sum([(cfg.branching_fractions[decay][1]*interp_eff_dict[decay][l,h])**2 for decay in cfg.sample_allocations["hadronic_background"]])
            

                full_sigma_S = np.sqrt(full_var_S)
                full_sigma_B = np.sqrt(full_var_B)

                full_S_error_arr[l,h]=full_sigma_S
                full_B_error_arr[l,h]=full_sigma_B

                if S+B>0:
                    #also calculate error in FOM itself from S,B error
                    full_err_FOM[l,h] = np.sqrt(1/(4*(S+B)**3)*((2*B+S)**2*full_sigma_S**2 + S**2*full_sigma_B**2))
                else:
                    FOM[l,h] = 0
                    full_err_FOM[l,h] = 0
    


    ## finding maximum so can plot slices
    max_sigma =FOM.max()
    indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
    l = lsearch[indices[0]]
    h = hsearch[indices[1]]

    # Prepare the data for the table
    table_data = [["Optimal FOM", "Optimal 1-P(l) cut", "Optimal 1-P(h) cut"],
                  [f"{max_sigma:.4f}", f"{l:.4f}", f"{h:.4f}"]]
    
    # Print the table
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    #print(S_error_arr)

    #print(B_error_arr)
    
    if incl_ZqqBFerror == True:
        return FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, full_S_error_arr, full_B_error_arr, full_err_FOM    
    else:
        return FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF


def plot_2d_optimisation(FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sigBF,vmax=5,SB_plots = False, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/'):
    
    #set output path
    set_outputpath(save_path)

    #plot FOM space
    plt.figure()
    plt.imshow(FOM, origin='lower',vmax=vmax)
    plt.xlabel('BDT_lh 1-P(heavy)')
    plt.ylabel('BDT_lh 1-P(light)')
    plt.colorbar(label=r'$S/\sqrt{S+B}$')
    plt.title('FOM with for common signal BF = '+f'{sigBF:.2e}')
    
    # Set tick labels for every 10th bin
    ytick_indices = np.arange(0,len(lsearch), round(len(lsearch)/5))
    xtick_indices = np.arange(0, len(hsearch), round(len(hsearch)/5))
    
    # Use ytick_indices and xtick_indices to set the ticks
    plt.yticks(ytick_indices, [round(lsearch[i],5) for i in ytick_indices])
    plt.xticks(xtick_indices, [round(hsearch[i],5) for i in xtick_indices], rotation=90)
    
    plt.savefig(os.path.join(save_path,f'FOM_heatmap.pdf'))


    ## finding maximum so can plot slices
    max_sigma =FOM.max()
    indices = np.unravel_index(np.argmax(FOM), np.shape(FOM))
    l = lsearch[indices[0]]
    h = hsearch[indices[1]]
    
    plt.figure()
    plt.errorbar(hsearch,FOM[indices[0],:], err_FOM[indices[0],:], label=r'FOM slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
    plt.xlabel('BDT_lh 1-P(heavy)')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.legend()
    plt.title('FOM slice with for common signal BF = '+f'{sigBF:.2e}')
    plt.savefig(os.path.join(save_path,f'FOM_slice_heavy.pdf'))

    plt.figure()
    plt.plot(lsearch,FOM[:,indices[1]],err_FOM[:,indices[1]], label=r'FOM slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
    plt.xlabel('BDT_lh 1-P(light)')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.legend()
    plt.title('FOM slice with for common signal BF = '+f'{sigBF:.2e}')
    plt.savefig(os.path.join(save_path,f'FOM_slice_light.pdf'))



    if SB_plots == True:
        #plot S
        plt.figure()
        plt.errorbar(lsearch,S_arr[:,indices[1]],S_error_arr[:,indices[1]], label=r'S slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'S')
        plt.legend()
        plt.title('Signal expectation slice with for common signal BF = '+f'{sigBF:.2e}')
        plt.savefig(os.path.join(save_path,f'S_slice_light.pdf'))

        plt.figure()
        plt.errorbar(hsearch,S_arr[indices[0],:],S_error_arr[indices[0],:], label=r'S slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'S')
        plt.legend()
        plt.title('Signal expectation slice with for common signal BF = '+f'{sigBF:.2e}')
        plt.savefig(os.path.join(save_path,f'S_slice_heavy.pdf'))
        
        #plot B
        plt.figure()
        plt.errorbar(lsearch,B_arr[:,indices[1]],B_error_arr[:,indices[1]], label=r'B slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'B')
        plt.legend()
        plt.savefig(os.path.join(save_path,f'B_slice_light.pdf'))

        plt.figure()
        plt.plot(hsearch,B_arr[indices[0],:],B_error_arr[indices[0],:], label=r'B slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'B')
        plt.legend()
        plt.savefig(os.path.join(save_path,f'B_slice_heavy.pdf'))


def make_final_binning_plot(df, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),signal_BF=1e-6, eventsProcessed_dict = cfg.eventsProcessed , histbins=(2,2), components =  ['hadronic_background','combined_signal'], nMC_plots_path=None, final_plot_path = None):

    def histogram_settings():
        hist_settings = { allocation: {} for allocation in cfg.sample_allocations }
        total_color = { allocation: {} for allocation in cfg.sample_allocations }
        for allocation in cfg.sample_allocations:
            samples = cfg.sample_allocations[allocation]
            if allocation=='combined_signal':
                hist_settings[allocation]['edgecolor'] =plt.cm.Blues( np.linspace(0, 1, 6)[3:-1] ) 
                hist_settings[allocation]['facecolor'] = ['none','none']
                hist_settings[allocation]['hatch'] = ['////',r'\\\\']
            elif allocation=='hadronic_background':
                hist_settings[allocation]['facecolor'] = plt.cm.Reds_r( np.linspace(0, 1, 6)[1:-1] )
                hist_settings[allocation]['edgecolor'] = ['none','none','none','none'] 
                hist_settings[allocation]['hatch'] =  [None,None,None,None] 
        return hist_settings
    
    print('--> Finding optimum cut')
    # find optimum cut
    FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_interp_N_dict ,hrange_plot=hrange_interp_N_dict, nlh=500 , sig_BF=signal_BF, incl_ZqqBFerror = False)
    indices = np.unravel_index(np.argmax(FOM), np.shape(FOM))
    l_cut = lsearch[indices[0]]
    h_cut = hsearch[indices[1]]

    #cut df on optimal BDT cuts
    cut_data = df.copy().query(f'(P_not_light>{l_cut})&(P_not_heavy>{h_cut})')

    # define samples want from components input 
    samples = flatten_list([cfg.sample_allocations[component] for component in components])

    print('--> Finding number of events per bin')
    # get number of events per bin in MC using np.2d histogram
    N_dict_MC={}
    
    for decay in samples:
        
        if nMC_plots_path is not None:
            plt.figure()
            h=plt.hist2d(cut_data[cut_data['decay']==decay]["P_not_heavy"], cut_data[cut_data['decay']==decay]["P_not_light"], bins=histbins, cmap=plt.cm.Blues,vmin=0,density=False,range = [[h_cut, 1], [l_cut, 1]])
            plt.ylabel('1-P(light)')
            plt.xlabel('1-P(heavy)') 
            plt.title(cfg.titles[decay])
            plt.colorbar(h[3])
            N_dict_MC[decay] =  h[0] 
            plt.savefig(os.path.join(set_outputpath(nMC_plots_path),f'NMC_remaining_{decay}_at_BF={signal_BF}_optcut.pdf'))
    
        else:
            h=np.histogram2d(cut_data[cut_data['decay']==decay]["P_not_heavy"], cut_data[cut_data['decay']==decay]["P_not_light"], bins=histbins,density=False,range = [[h_cut, 1], [l_cut, 1]])
            N_dict_MC[decay] =  h[0] #take counts per bin rather than bin edges

    #calculating per bin efficiencies from N MC remaining and convert into per bin S, B and errors (systematics include S and B from efficiency (finite MC size) and BF(Z--> qq) error [based on current measurements - would improve with FCCee])
    efficienies, efficiencies_err, N_dict_MC = eff_finder.get_eff_from_nMC_list(N_dict_MC)
    per_sample_n_expect_dict, per_sample_frac_eff_err, per_sample_frac_BFZbb_err=eff_finder.get_n_expected_components(efficienies, efficiencies_err,signal_bf=signal_BF)
    S, B, S_err, B_err = eff_finder.get_total_SB(per_sample_n_expect_dict, per_sample_frac_eff_err, per_sample_frac_BFZbb_err)

    if final_plot_path:
        if histbins==(2,2):
            plt.figure()
            tot_arr=[0,0,0,0]
            for allocation in cfg.sample_allocations:
                i=0
        
                if allocation not in components:
                    continue
                
                for sample in cfg.sample_allocations[allocation]:
                    h = per_sample_n_expect_dict[sample]
                    x = np.array([['Baseline','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']])
                    hist_opts = histogram_settings()[allocation]
                    plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[sample], bottom=tot_arr, width=1.0, lw=2,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                    i+=1
                    tot_arr = np.add(tot_arr, [h[1,0],h[0,0],h[0,1],h[1,1]])
            
            # sorting ticks so at edges but name still at centre
            bars = plt.gca().patches
            # sets major ticks so that name but no visible tick mark
            plt.tick_params(axis='x', which='major', length=0)  # hide tick marks at centres
            # Edge ticks (visible, no labels)
            edges = [b.get_x() for b in bars] + \
                    [b.get_x() + b.get_width() for b in bars]
            plt.gca().set_xticks(edges, minor=True)     # use gca just for minor ticks
            plt.tick_params(axis='x', which='minor', length=4)  # show edge ticks
    
            #add systematic error to B - error bar
            
            plt.errorbar([x[1,0],x[0,0],x[0,1],x[1,1]],[B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]],label='Systematic error on B',  fmt='None', ecolor='black',lw=1.5)
            '''
            #add systematic error to B - lines instead of error bar
            # Loop over the error values and draw horizontal lines at the top and bottom of the error bars
            for i, (x_label, b_val, b_err) in enumerate(zip([x[1,0], x[0,0], x[0,1], x[1,1]], 
                                                           [B[1,0], B[0,0], B[0,1], B[1,1]], 
                                                           [B_err[1,0], B_err[0,0], B_err[0,1], B_err[1,1]])):
                # Convert x_label to a numerical index
                x_val = i 
            
                # Top and bottom of the error bar
                top_error = b_val + b_err
                bottom_error = b_val - b_err
            
                # Draw horizontal lines at the top and bottom of the error bars
                plt.hlines(top_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=2)                                       
                if i ==3:
                    plt.hlines(bottom_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=2,label='Systematic error band on B')
                else:
                    plt.hlines(bottom_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=2)
            '''
            plt.title(r'Signal $\mathcal{B}(B^0_{(s)}\rightarrow{}$invisibles)$=$ '+ f'{signal_BF}')
            plt.legend()
            plt.ylabel('Expected Counts')
            plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}.pdf'))

        else:
            print('Warning: currently only set up to plot 2x2 binning')

    return per_sample_n_expect_dict, S, B, S_err, B_err, signal_BF


def likelihood_model_builder_max_err(S, B, S_err, B_err, signal_BF, # these need to be binned
                             ntoys = 250,
                             fit_plotpath=None, spread_plotpath=None):

    """ 
    likelihood_model_builder(**opts ) will return optimum point from minimising signal error on fit to toys

    """

    poisson_expectation = B + S
    max_background_error = np.max(B_err/B)  #need fractional error as it propagates through on scale factor                            
    
    ## define fit to toy (this is the negative log likelihood to minimize)
    def poisson_likelihood(sc_b, sc_s): #scale S and B separately, assuming know shape perfectly
        expectation = sc_b * B + sc_s * S # assumes know shape perfectly
        poiss_term = -np.sum( poisson.logpmf(toy_data, expectation)) #logpmf = Log of the probability mass function
        bkg_constraint_term = -norm.logpdf( sc_b, 1, max_background_error )
        return poiss_term + bkg_constraint_term
    
    significance_arr=[]
    av_significance_for_bf = []
    stdev_significance_for_bf=[]

    # throw and refit toys
    for n in range(ntoys):
        toy_data = np.random.poisson(poisson_expectation) #throw toys
        mi = Minuit(poisson_likelihood, sc_b=1, sc_s=1 ) #fit toy
        mi.migrad()
        mi.hesse()
        sc_s = mi.values['sc_s']
        sc_b = mi.values['sc_b']
        sc_s_err = mi.errors['sc_s']
        sc_b_err = mi.errors['sc_b']

                                
        # refit with S=0 fixed for significance
        mi0 = Minuit(poisson_likelihood, sc_b=1, sc_s=0 )
        mi0.fixed['sc_s'] = True
        mi0.migrad()
        mi0.hesse()
        sc_s0 = mi0.values['sc_s']
        sc_b0 = mi0.values['sc_b']
           

        significance = np.sqrt(abs(2*(mi.fval-mi0.fval)))
        significance_arr.append(significance)  

        if fit_plotpath:
            if n ==0:
                x = np.array([['A','B'],['C','D']])
                plt.figure() 
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]],label='Fit B', width=1.0, edgecolor='red', facecolor='none',hatch=r'\\\\')
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]],label='Fit S', bottom=[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], width=1.0, edgecolor='blue',hatch='////', facecolor='none')
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[max_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], bottom =np.subtract(np.add([sc_b *i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], [sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]]),[max_background_error*sc_b*i/2 for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]), label='Gaussian constraint from maximum per-bin \n error on generator B', color='black', alpha=0.4, width=1)
                plt.plot([x[1,0],x[0,0],x[0,1],x[1,1]], [toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]],'+',label='Toy Data', color='k')
                plt.legend()
                plt.ylabel('Counts')
                plt.title(f'Example toy fit for signal BF = {signal_BF}')
                plt.savefig(os.path.join(set_outputpath(fit_plotpath),'toy_fit_for_first_toy.pdf'))
    
    if spread_plotpath:
        plt.figure()        
        plt.title(f'Histogram of significance values over {ntoys} toys')
        plt.hist(significance_arr)
        plt.xlabel('Significance')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(set_outputpath(fit_plotpath),'histogram_of_all_toys.pdf'))
    
        
    significance_av = np.average(significance_arr)
    significance_stdev = np.std(significance_arr)    
    av_significance_for_bf.append(significance_av)
    stdev_significance_for_bf.append(significance_stdev)
        

    return av_significance_for_bf, stdev_significance_for_bf



def plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BFs=np.logspace(-9,-4,250),incl_ZqqBFerror=True, incl_toys_fit=False, full_df=None, ntoys=5000,plot=True,savepath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995'):
    
    #create dictionaries to store results
    max_FOM = np.zeros(len(sig_BFs))
    max_FOM_err = np.zeros(len(sig_BFs))
    BFs=np.zeros(len(sig_BFs))
    light_cut=np.zeros(len(sig_BFs))
    heavy_cut=np.zeros(len(sig_BFs))
    S_exp = np.zeros(len(sig_BFs))
    B_exp = np.zeros(len(sig_BFs))
    S_err = np.zeros(len(sig_BFs))
    B_err = np.zeros(len(sig_BFs))
    full_S_err= np.zeros(len(sig_BFs))
    full_B_err= np.zeros(len(sig_BFs))
    toys_significance = np.zeros(len(sig_BFs))
    toys_sig_spread = np.zeros(len(sig_BFs))


    i=0

    for BF in sig_BFs:
        
        FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, full_S_error_arr, full_B_error_arr, full_err_FOM   = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF, incl_ZqqBFerror=True)

        ## finding maximum so can plot slices
        indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
        l = lsearch[indices[0]]
        h = hsearch[indices[1]]

        max_FOM[i] = FOM.max()
        max_FOM_err[i] = err_FOM[indices[0],indices[1]]
        light_cut[i] = l
        heavy_cut[i] = h    
        BFs[i] = sig_BF
        S_exp[i] = S_arr[indices[0],indices[1]]
        B_exp[i] = B_arr[indices[0],indices[1]]
        S_err[i] = S_error_arr[indices[0],indices[1]]
        B_err[i] = B_error_arr[indices[0],indices[1]]
        
        if incl_ZqqBFerror == True:
            full_S_err[i] = full_S_error_arr[indices[0],indices[1]]
            full_B_err[i] = full_B_error_arr[indices[0],indices[1]]

        if incl_toys_fit == True:
            if full_df is None:
                print('Warning: no data provided for toys fit')
                continue
            else:
                toys_per_sample_n_expect_dict, toys_S, toys_B, toys_S_err, toys_B_err, signal_BF  = make_final_binning_plot(full_df, interp_N_dict, signal_BF=BF, histbins=(2,2), nMC_plots_path=None, final_plot_path = None)
                av_significance_for_bf, stdev_significance_for_bf = likelihood_model_builder_max_err(toys_S, toys_B, toys_S_err, toys_B_err, signal_BF=signal_BF, ntoys = ntoys, fit_plotpath=None, spread_plotpath=None)
                toys_significance[i] = av_significance_for_bf[0]
                toys_sig_spread[i] = stdev_significance_for_bf[0]

        i+=1


    #find 3 sigma and 5 sigma points

    closest_index_5 = np.argmin(np.abs(np.array(max_FOM) - 5))
    highlight_x_5 = BFs[closest_index_5]
    highlight_y_5 = max_FOM[closest_index_5]

    closest_index_3 = np.argmin(np.abs(np.array(max_FOM) - 3))
    highlight_x_3 = BFs[closest_index_3]
    highlight_y_3 = max_FOM[closest_index_3]

    print(f"5 sigma BFs = {BFs[closest_index_5]}" )
    print(f"3 sigma BFs = {BFs[closest_index_3]}" )

    #calulating S/sqrt(S+B+varS+varB)
    significance_incl_error = S_exp/np.sqrt(S_exp+B_exp+S_err**2+B_err**2)
    inclerr_closest_index_5 = np.argmin(np.abs(np.array(significance_incl_error) - 5))
    inclerr_highlight_x_5 = BFs[inclerr_closest_index_5]
    inclerr_highlight_y_5 = significance_incl_error[inclerr_closest_index_5]

    inclerr_closest_index_3 = np.argmin(np.abs(np.array(significance_incl_error) - 3))
    inclerr_highlight_x_3 = BFs[inclerr_closest_index_3]
    inclerr_highlight_y_3 = significance_incl_error[inclerr_closest_index_3]

    print(f"5 sigma BFs incl syst error = {BFs[inclerr_closest_index_5]}" )
    print(f"3 sigma BFs incl syst error  = {BFs[inclerr_closest_index_3]}" )




    if plot==True:

        #plotting S/sqrt(S+B) line
        plt.figure()
        plt.plot(BFs,max_FOM, label = r'$S/\sqrt{S+B}$')
        plt.fill_between(BFs, max_FOM-max_FOM_err, max_FOM+max_FOM_err,alpha=0.55)
        if incl_ZqqBFerror == True:
            significance_incl_error = S_exp/np.sqrt(S_exp+B_exp+full_S_err**2+full_B_err**2)
            plt.plot(BFs,significance_incl_error, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n including '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})}$, $\sigma_{\varepsilon_i}$')
            plt.fill_between(BFs, significance_incl_error, significance_incl_error,alpha=0)
        else:
            plt.plot(BFs,significance_incl_error, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n neglecting '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})} $')
            plt.fill_between(BFs, significance_incl_error, significance_incl_error,alpha=0)
        if incl_toys_fit == True:
            plt.plot(BFs,toys_significance, label = r'$\sqrt{2\Delta\ln{\mathcal{L}}}$ mean'+' \n over '+f'{ntoys} toys')
            plt.fill_between(BFs, toys_significance-toys_sig_spread, toys_significance+toys_sig_spread,alpha=0.55)
        plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.ylabel(r'Significance')
        plt.xscale('log')
        plt.ylim(0,8)

        # Add vertical and horizontal lines stopping at the points
        plt.vlines(x=highlight_x_5, ymin=0, ymax=highlight_y_5, color='deeppink', linestyle='--', label=r'5$\sigma$')
        plt.vlines(x=inclerr_highlight_x_5, ymin=0, ymax=inclerr_highlight_y_5, color='deeppink', linestyle='--')
        #plt.hlines(y=highlight_y_5, xmin=min(BFs), xmax=inclerr_highlight_x_5, color='deeppink', linestyle='--')

        plt.vlines(x=highlight_x_3, ymin=0, ymax=highlight_y_3, color='purple', linestyle='--', label=r'3$\sigma$')
        plt.vlines(x=inclerr_highlight_x_3, ymin=0, ymax=inclerr_highlight_y_3, color='purple', linestyle='--')
        #plt.hlines(y=highlight_y_3, xmin=min(BFs), xmax=inclerr_highlight_x_3, color='purple', linestyle='--')

        #Add equivalent lines for with error

        plt.legend()
        plt.title(r'Optimum FOM as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        if incl_ZqqBFerror == True:
            if incl_toys_fit == True:
                plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_inclfullerr_wtoys.pdf'))
            else:
                plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_inclfullerr.pdf'))

        else:
            if incl_toys_fit == True:
                plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_wtoys.pdf'))
            else:
                plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF.pdf'))


    
    #convert sigma to CL - for now one sided

    CL = sigma_to_percentage(max_FOM)
    CL_incl_error = sigma_to_percentage(significance_incl_error) # this dependent on if have full error or not above
    CL_toys = sigma_to_percentage(toys_significance)

    #find 90 % and 95 % points

    closest_index_90 = np.argmin(np.abs(np.array(CL) - 90))
    highlight_x_90 = BFs[closest_index_90]
    highlight_y_90 = CL[closest_index_90]

    closest_index_95 = np.argmin(np.abs(np.array(CL) - 95))
    highlight_x_95 = BFs[closest_index_95]
    highlight_y_95 = CL[closest_index_95]


    #repeat for incl error
    inclerr_closest_index_90 = np.argmin(np.abs(np.array(CL_incl_error) - 90))
    inclerr_highlight_x_90 = BFs[inclerr_closest_index_90]
    inclerr_highlight_y_90 = CL_incl_error[inclerr_closest_index_90]

    inclerr_closest_index_95 = np.argmin(np.abs(np.array(CL_incl_error) - 95))
    inclerr_highlight_x_95 = BFs[inclerr_closest_index_95]
    inclerr_highlight_y_95 = CL_incl_error[inclerr_closest_index_95]


    if plot == True: 
        plt.figure()
        plt.plot(BFs,CL,label='$S/\sqrt{S+B}$' )
        plt.fill_between(BFs, CL-max_FOM_err/max_FOM*CL, CL+max_FOM_err/max_FOM*CL,alpha=0.55)
        if incl_ZqqBFerror == True:
            plt.plot(BFs,CL_incl_error, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n including '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})}$, $\sigma_{\varepsilon_i}$')
            plt.fill_between(BFs, CL_incl_error, CL_incl_error,alpha=0)
        else:
            plt.plot(BFs,CL_incl_error, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n neglecting '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})} $')
            plt.fill_between(BFs, CL_incl_error, CL_incl_error,alpha=0)
        if incl_toys_fit == True:
            plt.plot(BFs,CL_toys, label = r'$\sqrt{2\Delta\ln{\mathcal{L}}}$ mean'+' \n over '+f'{ntoys} toys')
            plt.fill_between(BFs, CL_toys-toys_sig_spread/toys_significance*CL_toys, CL_toys+toys_sig_spread/toys_significance*CL_toys,alpha=0.55)
        plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.ylabel(r'1-CL (1-sided test)')
        plt.xscale('log')

        # Add vertical and horizontal lines stopping at the points
        plt.vlines(x=highlight_x_95, ymin=52, ymax=highlight_y_95, color='deeppink', linestyle='--', label=r'95$\%$')
        plt.vlines(x=inclerr_highlight_x_95, ymin=52, ymax=inclerr_highlight_y_95, color='deeppink', linestyle='--')
        #plt.hlines(y=highlight_y_95, xmin=min(BFs), xmax=inclerr_highlight_x_95, color='deeppink', linestyle='--')

        plt.vlines(x=highlight_x_90, ymin=52, ymax=highlight_y_90, color='purple', linestyle='--', label=r'90$\%$')
        plt.vlines(x=inclerr_highlight_x_90, ymin=52, ymax=inclerr_highlight_y_90, color='purple', linestyle='--')
        #plt.hlines(y=highlight_y_90, xmin=min(BFs), xmax=inclerr_highlight_x_90, color='purple', linestyle='--')
        plt.ylim(52,102)
        plt.legend()
        plt.title(r'1-CL as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        if incl_ZqqBFerror == True:
            if incl_toys_fit == True:
                plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_inclfullerr_wtoys.pdf'))
            else:
                plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_inclfullerr.pdf'))

        else:
            if incl_toys_fit == True:
                plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_wtoys.pdf'))
            else:
                plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF.pdf'))

    print(f"95% BFs = {BFs[closest_index_95]}" )
    print(f"90% BFs = {BFs[closest_index_90]}" )

    print(f"95% BFs incl error = {BFs[inclerr_closest_index_95]}" )
    print(f"90% BFs incl error = {BFs[inclerr_closest_index_90]}" )

    return max_FOM, light_cut, heavy_cut, BFs, CL
    

if __name__=="__main__":

    
    #Load dataframe with bdtlh version applied
    data={}
    dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
    folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut' 
    for sample in cfg.samples:
        data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))

    # Concatenate all DataFrames
    full_data = pd.concat(data.values(), ignore_index=True)


    #add extra rows so can work in range0.9-1
    full_data['P_signal'] = full_data['bdt_score_2']
    full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
    full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

    #add any extra cuts need here###########################
    full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus
    
    '''
    N_dict, interp_N_dict, s_values_dict, eff_dict, err_dict = create_N_map(full_data,lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, smoothing=False, kx=2, ky=2, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/')
    plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,slice=True, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/')
    '''

    save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/'
    
    with open(os.path.join(set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
        interp_N_dict = dill.load(dill_file)

    with open(os.path.join(set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)
    
    
    plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation/'
    #plot_interpolted_effs(interp_N_dict, N_dict ,nlh=20, slice=True, save_path=plotpath)
    #FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BF=1e-7)
    #plot_2d_optimisation(FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, vmax=20,SB_plots = True, save_path=plotpath)
    
    #per_sample_n_expect_dict, S, B, S_err, B_err, signal_BF  = make_final_binning_plot(full_data, interp_N_dict, signal_BF=1e-6, histbins=(2,2), nMC_plots_path=None, final_plot_path = None)
    #likelihood_model_builder_max_err(S, B, S_err, B_err, signal_BF=signal_BF, # these need to be binned
    #                         ntoys = 5000, fit_plotpath=f'{plotpath}final_binning/1e-6/', spread_plotpath=f'{plotpath}final_binning/1e-6/')

    plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(-9,-5,250), incl_ZqqBFerror=True, incl_toys_fit=True, full_df=full_data, ntoys=5000,plot=True,savepath = plotpath)
    