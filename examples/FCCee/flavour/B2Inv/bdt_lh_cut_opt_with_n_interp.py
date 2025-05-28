import os
import glob
import sys
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from yaml import safe_load, YAMLError, dump
import pickle
from tabulate import tabulate
from scipy.interpolate import RectBivariateSpline
from scipy.stats import poisson#, norm
from scipy.stats import norm as snorm
from iminuit import Minuit
from numba_stats import norm
import joblib
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar

import config as cfg 
import efficiency_finder
import post_bdtlh_efficiency_finder as post_bdt_eff_finder
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
    percentage = snorm.cdf(sigma) * 100
    return percentage

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

#round to given number of sig figs
def round_sig(x, sig=1):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def latex_form_exp(signal_BF):
        #turn BF into title worthy version
    num = f"{signal_BF:.1e}".split('e')[0]  # '1.0'
    exponent = int(f"{signal_BF:.1e}".split('e')[1])  # -6
    latex_BF = f"${num} \\times 10^{{{exponent}}}$"

    return latex_BF

def find_x_for_y(y_target, f, x_bounds):
    def g(x): return f(x) - y_target
    sol = root_scalar(g, bracket=x_bounds)
    return sol.root if sol.converged else None


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

                eff_lh, err_lh, N_remaining = post_bdt_eff_finder.get_total_eff_post_bdt(subf, cut= lh_cut, verbose = False, eventsProcessed_dict = cfg.eventsProcessed)
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



def plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,normalised=False, separate_cbar = False,slice=False, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/'):

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
        plt.xlabel('$1-P(h)$')
        plt.ylabel('$1-P(l)$')
        plt.colorbar(label='MC events')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearch), round(len(lsearch)/5))
        xtick_indices = np.arange(0, len(hsearch), round(len(hsearch)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearch[i],4) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearch[i],4) for i in xtick_indices])#, rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(set_outputpath(save_path),f'N_{decay}.pdf'))

        #plotting interpolated N map

        plt.figure()
        gridded_interp = [[interp_N_dict[decay](lsearchinterp[l], hsearchinterp[h]).item() for h in range(len(hsearchinterp))] for l in range(len(lsearchinterp))]
        if normalised==True:
            im = plt.imshow([[gridded_interp[l][h]/interp_N_dict[decay](lsearchinterp[0], hsearchinterp[0]).item() for h in range(len(hsearchinterp))] for l in range(len(lsearchinterp))], origin='lower')
            if separate_cbar == False:
                plt.colorbar(label='Interpolated Density')

        else:
            im = plt.imshow(gridded_interp, origin='lower')
            if separate_cbar == False:
                plt.colorbar(label='Interpolated MC Counts')

        plt.xlabel('$1-P(h)$', fontsize=16)
        plt.ylabel('$1-P(l)$', fontsize=16)
        
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],4) for i in ytick_indices], fontsize=12)
        plt.xticks(xtick_indices, [round(hsearchinterp[i],4) for i in xtick_indices], fontsize=12)#, rotation=90)
  
        if normalised == True:
            plt.tight_layout()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_interp_{decay}_normalised.pdf'), bbox_inches='tight')
            if separate_cbar ==True:
                fig2, ax2 = plt.subplots()
                plt.colorbar(im, ax=ax2, label='Interpolated Density')
                fig2.tight_layout()
                ax2.remove()
                plt.savefig(os.path.join(set_outputpath(save_path),'N_interp_colorbar_density.pdf'), bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_interp_{decay}.pdf'), bbox_inches='tight')

            if separate_cbar ==True:
                fig2, ax2 = plt.subplots()
                plt.colorbar(im, ax=ax2, label='Interpolated MC Counts')
                fig2.tight_layout()
                ax2.remove()
                plt.savefig(os.path.join(set_outputpath(save_path),f'N_interp_colorbar_{decay}.pdf'), bbox_inches='tight')

        plt.close('all')

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
                
                #compute efficiency and wilson error
                total_efficiency, error = efficiency_finder.efficiency_calc(eventsProcessed, N_post)

                fine_grid_splined_eff[l,h] = total_efficiency 
                fine_grid_splined_eff_err[l,h] = error 
        
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
                #calculate efficiency and wilson error
                total_efficiency, error = efficiency_finder.efficiency_calc(eventsProcessed, N_post)
        
                eff[l,h] = total_efficiency
                eff_err[l,h] = error
        
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




  
def run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200, sig_BF=1e-7,incl_other_syst = True):
    
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



    print('Calculating S and B')
    
    '''
    #defining constants needed
    N_z = cfg.N_z
    k = 2 * N_z * cfg.branching_fractions["p8_ee_Zbb_ecm91"][0] * sig_BF # common part of signal expectation
    '''
    for l in np.arange(0,len(lsearch), 1):
        for h in np.arange(0,len(hsearch), 1):
            
            interp_eff_dict_lh = {decay: interp_eff_dict[decay][l, h] for decay in interp_eff_dict.keys()}
            interp_eff_err_dict_lh = {decay: interp_eff_err_dict[decay][l, h] for decay in interp_eff_err_dict.keys()}
            

            per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err =post_bdt_eff_finder.get_n_expected_components(interp_eff_dict_lh, interp_eff_err_dict_lh, signal_bf=sig_BF)
            S, B, S_err, B_err = post_bdt_eff_finder.get_total_SB(per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err, incl_other_syst=incl_other_syst)
            '''
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
            '''
            S_arr[l,h]=S
            B_arr[l,h]=B
            S_error_arr[l,h]=S_err
            B_error_arr[l,h]=B_err

            if S+B>0:
                FOM[l,h] = S/np.sqrt(S+B)
                #also calculate error in FOM itself from S,B error
                err_FOM[l,h] = np.sqrt(1/(4*(S+B)**3)*((2*B+S)**2*S_err**2 + S**2*B_err**2))
            else:
                FOM[l,h] = 0
                err_FOM[l,h] = 0

            '''
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
                '''


    ## finding maximum so can plot slices
    max_sigma =FOM.max()
    indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
    l = lsearch[indices[0]]
    h = hsearch[indices[1]]

    # Prepare the data for the table
    table_data = [["Optimal FOM", "Optimal 1-P(l) cut", "Optimal 1-P(h) cut"],
                  [f"{max_sigma:.6f}", f"{l:.6f}", f"{h:.6f}"]]
    
    # Print the table
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

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


def make_final_binning_plot(df, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 200, signal_BF=1e-6, eventsProcessed_dict = cfg.eventsProcessed , histbins=(2,2), components =  ['hadronic_background','combined_signal'], binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                            plot_signal_components=False,  nMC_plots_path=None, final_plot_path = None, pull_type_plot=False, lcut=None, hcut=None, logpath= None):
    #turn BF into title worthy version
    latex_BF = latex_form_exp(signal_BF)


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
    if lcut is not None and hcut is not None:
        l_cut = lcut
        h_cut = hcut
    else:
        # find optimum cut
        FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_interp_N_dict ,hrange_plot=hrange_interp_N_dict, nlh=nlh , sig_BF=signal_BF, incl_other_syst = False)
        
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
            plt.ylabel('$1-P(l)$')
            plt.xlabel('$1-P(h)$') 
            plt.title(cfg.titles[decay])
            plt.colorbar(h[3])
            N_dict_MC[decay] =  h[0] 
            plt.savefig(os.path.join(set_outputpath(nMC_plots_path),f'NMC_remaining_{decay}_at_BF={signal_BF}_optcut.pdf'))

        else:
            h=np.histogram2d(cut_data[cut_data['decay']==decay]["P_not_heavy"], cut_data[cut_data['decay']==decay]["P_not_light"], bins=histbins,density=False,range = [[h_cut, 1], [l_cut, 1]])
            N_dict_MC[decay] =  h[0] #take counts per bin rather than bin edges
        
        xedges = h[1]
        yedges = h[2]

        
        
    if final_plot_path is not None:
        filename = os.path.join(set_outputpath(final_plot_path),'bin_edges_for_BF.txt')
        mode = 'a' if os.path.exists(filename) else 'w'  # append if exists, else write

        with open(filename, mode) as log_file:
            log_file.write(f"BF: {signal_BF}\n")
            log_file.write(f"1-P(h): {xedges}\n")
            log_file.write(f"1-P(l): {yedges}\n")
            log_file.write(f"\n")

    elif logpath is not None:
        filename = os.path.join(set_outputpath(logpath),'bin_edges_for_BF.txt')
        mode = 'a' if os.path.exists(filename) else 'w'  # append if exists, else write

        with open(filename, mode) as log_file:
            log_file.write(f"BF: {signal_BF}\n")
            log_file.write(f"1-P(h): {xedges}\n")
            log_file.write(f"1-P(l): {yedges}\n")
            log_file.write(f"\n")

    #calculating per bin efficiencies from N MC remaining and convert into per bin S, B and errors (systematics include S and B from efficiency (finite MC size) and BF(Z--> qq) error [based on current measurements - would improve with FCCee])
    efficienies, efficiencies_err, N_dict_MC = post_bdt_eff_finder.get_eff_from_nMC_list(N_dict_MC)
    per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err  =post_bdt_eff_finder.get_n_expected_components(efficienies, efficiencies_err,signal_bf=signal_BF)
    S, B, S_err, B_err = post_bdt_eff_finder.get_total_SB(per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err, incl_other_syst=True)

    if final_plot_path:
        x = binned_x_axis
        if histbins==(2,2):
            plt.figure()
            tot_arr=[0,0,0,0]
            tot_signal = [0,0,0,0]

            if pull_type_plot==True:
                frac_sub=3
                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(2, 1,height_ratios=[frac_sub, 1])  # 2 rows: 3:1 height ratio
                # Main plot (top)
                ax_main = fig.add_subplot(gs[0])
                plt.sca(ax_main)# Set current axis so existing plotting code works unchanged


            #return to plotting script
            for allocation in cfg.sample_allocations:
                i=0
        
                if allocation not in components:
                    continue
                
                for sample in cfg.sample_allocations[allocation]:

                    if sample in cfg.sample_allocations['combined_signal']: #plot combined signal
                        h = per_sample_n_expect_dict[sample]
                        tot_signal = np.add(tot_signal,[h[1,0],h[0,0],h[0,1],h[1,1]])

                        if plot_signal_components == True:
                            hist_opts = histogram_settings()[allocation]
                            plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[sample], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        tot_arr = np.add(tot_arr,[h[1,0],h[0,0],h[0,1],h[1,1]])
                        i+=1

                    else:
                        h = per_sample_n_expect_dict[sample]
                        hist_opts = histogram_settings()[allocation]
                        plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[sample], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        i+=1
                        tot_arr = np.add(tot_arr, [h[1,0],h[0,0],h[0,1],h[1,1]])

            if plot_signal_components == False:
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal,label=r'$\mathcal{B}(B^0_{(s)}\rightarrow{}$invisibles$)=$ '+ f'{latex_BF}', bottom=np.subtract(tot_arr,tot_signal), width=1.0, lw=1.5,edgecolor = plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')              
                
            # sorting ticks so at edges but name still at centre
            bars = plt.gca().patches
            #hide central ticks
            plt.tick_params(axis='x', which='major', length=0)  # hide tick marks at centres
            
            # Edge ticks (visible, no labels)
            edges = [b.get_x() for b in bars] + \
                    [b.get_x() + b.get_width() for b in bars]
            plt.gca().set_xticks(edges, minor=True)     # use gca just for minor ticks
            plt.tick_params(axis='x', which='minor', length=4)  # show edge ticks
            plt.legend()
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.ylabel('Expected Counts')

            if pull_type_plot==True:
                # Bottom axis 
                ax_sub = fig.add_subplot(gs[1], sharex=ax_main)
                
                if plot_signal_components == False:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')
                else:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) , facecolor= 'none', hatch='\\\\\\',label=r'$S$ for $\mathcal{B}(B^0_{(s)}\rightarrow{}$invisibles$)=$ '+ f'{latex_BF}')

                ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =[-val for val in [B_err[1,0], B_err[0,0], B_err[0,1], B_err[1,1]]], color='black', alpha=0.45, width=1, label='$\sigma_B$')#r'$Z \to q \bar{q}$ background systematic')


                # Clean up sub axis
                ax_sub.set_xticks([x[1,0],x[0,0],x[0,1],x[1,1]])
                ax_sub.tick_params(axis='x', which='major', length=0) 
                ax_sub.set_ylabel('Backgrond Subtracted \n Counts')  
                ax_sub.tick_params(axis='x', which='minor', length=4)  # show edge ticks
                ax_sub.legend()

                ax_main.tick_params(axis='x', which='major', bottom=False, labelbottom=False)


                #determine axes and their limits 
                main_height = ax_main.get_ylim()

                #determine sub_height
                sub_height = np.diff(main_height)/frac_sub
                ax_sub.set_ylim(-1.3*B_err[1,1],sub_height -1.3*B_err[1,1])

                plt.legend()
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}_with_second_axis.pdf'), bbox_inches='tight')
            else: 
                
                #add systematic error to B - error bar
                #plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', color='black', alpha=0.4, width=1)
                #plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', facecolor='none',  width=1, edgecolor='black')            
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', facecolor='none',  width=1, edgecolor='none', hatch='///')            
                
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
                    plt.hlines(top_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=1.5)                                       
                    if i ==3:
                        plt.hlines(bottom_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=1.5,label=r'$Z \to q \bar{q}$ background systematic')
                    else:
                        plt.hlines(bottom_error, x_val - 0.5, x_val + 0.5, color='black', linewidth=1.5)
                '''
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}.pdf'))
                
        else:
            print('Warning: currently only set up to plot 2x2 binning')

    return per_sample_n_expect_dict, S, B, S_err, B_err, signal_BF,l_cut,h_cut


def likelihood_model_builder(df, interp_N_dict, signal_BF=1e-6,
                             lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh=200,bins = (2,2),
                             ntoys = 250,
                             fit_plotpath=None, x_values = np.array([['A','B'],['C','D']]), spread_plotpath=None, logpath=None, lcut=None, hcut=None):

    """ 
    likelihood_model_builder(**opts ) will return optimum point from minimising signal error on fit to toys

    """

    _, S, B, S_err, B_err, signal_BF,opt_l_cut,opt_h_cut = make_final_binning_plot(df, interp_N_dict, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=bins, components =  ['hadronic_background','combined_signal'], binned_x_axis = x_values,nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=lcut, hcut=hcut)
    _, onebin_S, onebin_B, onebin_S_err, onebin_B_err,_,_,_ = make_final_binning_plot(df, interp_N_dict, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=1, components =  ['hadronic_background','combined_signal'], binned_x_axis = x_values, nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=opt_l_cut, hcut=opt_h_cut)
    poisson_expectation = B + S
    overall_background_error = onebin_B_err.item()/onebin_B.item()  #need fractional error as it propagates through on scale factor                            

    
    ## define fit to toy (this is the negative log likelihood to minimize)
    def poisson_likelihood(sc_b, sc_s): #scale S and B separately, assuming know shape perfectly
        expectation = sc_b * B + sc_s * S # assumes know shape perfectly
        poiss_term = -np.sum( poisson.logpmf(toy_data, expectation)) #logpmf = Log of the probability mass function
        bkg_constraint_term = -snorm.logpdf( sc_b, 1, overall_background_error)
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
                print(f'fit sc_s = {sc_s}')
                print(f'fit sc_b = {sc_b}')
                x_strings =[x_values[1,0],x_values[0,0],x_values[0,1],x_values[1,1]] 
                x = np.arange(len(x_strings))
                plt.figure() 
                plt.bar(x,[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]],label='Fit B', width=1.0, edgecolor='red', facecolor='none',hatch='///')
                plt.bar(x,[sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]],label='Fit S', bottom=[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], width=1.0, edgecolor=plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) ,hatch='\\\\\\', facecolor='none')
                plt.bar(x,[2*overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], bottom =np.subtract(np.add([sc_b *i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], [sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]]),[overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]), label=r'$\sigma_B$', color='black', alpha=0.4, width=1)
                plt.errorbar(x, [toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]],yerr=[np.sqrt(i) for i in [toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]]],xerr=0.5, fmt='.',label='Toy data', color='k')
                plt.legend()
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.ylabel('Counts')
                plt.xticks(x, x_strings)
                # sorting ticks so at edges but name still at centre
                bars = plt.gca().patches
                #hide central ticks
                plt.tick_params(axis='x', which='major', length=0)  # hide tick marks at centres
                # Edge ticks (visible, no labels)
                edges = [b.get_x() for b in bars] + \
                    [b.get_x() + b.get_width() for b in bars]
                plt.gca().set_xticks(edges, minor=True)     # use gca just for minor ticks
                plt.tick_params(axis='x', which='minor', length=4)  # show edge ticks

                plt.savefig(os.path.join(set_outputpath(os.path.join(fit_plotpath,'toy_fits')),f'toy_fit_for_first_toy_BF{signal_BF}.pdf'))
    
    
    data = np.array(significance_arr)

    # Truncation limits
    lower, upper = data.min(), data.max()

    # Truncated Gaussian log-likelihood
    def truncated_logpdf(x, mu, sigma):
        norm_const = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
        return norm.logpdf(x, mu, sigma) - np.log(norm_const)

    # Negative log-likelihood
    def nll(mu, sigma):
        if sigma <= 0:
            return np.inf
        return -np.sum(truncated_logpdf(data, mu, sigma))

    # Initial parameter guesses
    mu_init = np.mean(data)
    sigma_init = np.std(data)

    # Fit using iminuit
    m = Minuit(nll, mu=mu_init, sigma=sigma_init)
    m.limits["sigma"] = (1e-3, None)
    m.migrad()
    m.hesse()

    fitted_mu = m.values['mu']
    fitted_sigma = m.values['sigma']

    # Plot
    if spread_plotpath:
        fig, ax = plt.subplots()
        counts, bins, _ = ax.hist(data, bins=30, density=True, label="Data")

        # Evaluate PDF for plotting
        x_vals = np.linspace(lower, upper, 1000)
        norm_const = norm.cdf(upper, fitted_mu, fitted_sigma) - norm.cdf(lower, fitted_mu, fitted_sigma)
        pdf_vals = norm.pdf(x_vals, fitted_mu, fitted_sigma) / norm_const

        ax.plot(x_vals, pdf_vals, label=f'Gaussian fit\n$\mu$ = {fitted_mu:.2f}, $\sigma$ = {fitted_sigma:.2f}')
        ax.set_xlabel("Significance")
        ax.set_ylabel("Density")
        plt.title(f'Histogram of significance values over {ntoys} toys with truncated Gaussian fit')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(set_outputpath(os.path.join(fit_plotpath,'toy_histograms')),f'histogram_of_all_toys_BF{signal_BF}.pdf'))

    
     
    significance_mu = fitted_mu
    significance_sigma = fitted_sigma    
    av_significance_for_bf.append(significance_mu)
    stdev_significance_for_bf.append(significance_sigma)
    

    print(f'BF ={signal_BF}') 
    print(f'Mean toy significance from fit ={av_significance_for_bf}')   

    return av_significance_for_bf, stdev_significance_for_bf



def plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=False, full_df=None, ntoys=200,plot=True,saveplotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995',toyplotpath = None, dict_path = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995'):
    
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


    i=0
    toy_BFs=[]
    toy_significance=[]
    toy_sig_spread = []

    for BF in sig_BFs:
        
        FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF   = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF, incl_other_syst=True)

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


        if incl_toys_fit == True:
            if i%2==0:
                if full_df is None:
                    print('Warning: no data provided for toys fit')
                    continue
                else:
                    if i % 20 == 0:
                        av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder(full_df, interp_N_dict, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=toyplotpath, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=toyplotpath)
                    else:
                        av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder(full_df, interp_N_dict, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=None, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=None)
                    
                    toy_BFs.append(sig_BFs[i])
                    toy_significance.append(av_significance_for_bf[0])
                    toy_sig_spread.append(stdev_significance_for_bf[0])


        i+=1

    if incl_toys_fit == True:
        #convert toy lists to numpy
        toys_BFs = np.array(toy_BFs)
        toys_significance = np.array(toy_significance)
        toys_sig_spread = np.array(toy_sig_spread)

    
    ##########################
    #printing some #s to check!
    ##########################

    #find 3 sigma and 5 sigma points
    #with interpolation 
    five_sigma_BF = np.interp(5, max_FOM, BFs)
    three_sigma_BF = np.interp(3, max_FOM, BFs)
    print(f"5 sigma BFs = {five_sigma_BF}" )
    print(f"3 sigma BFs = {three_sigma_BF}" )


    #calulating S/sqrt(S+B+varS+varB)
    #error includes whatever specified in 2d optimisation 

    significance_incl_error = S_exp/np.sqrt(S_exp+B_exp+S_err**2+B_err**2)
    five_sigma_BF_inclerr = np.interp(5, significance_incl_error, BFs)
    three_sigma_BF_inclerr = np.interp(3, significance_incl_error, BFs)
    print(f"5 sigma BFs incl error = {five_sigma_BF_inclerr}" )
    print(f"3 sigma BFs incl error= {three_sigma_BF_inclerr}" )


    CL = sigma_to_percentage(max_FOM)


    naive_dict= {'BFs': BFs,
                'significance': max_FOM, 
                'error':max_FOM_err}
    
    toys_dict= {'BFs': toys_BFs,
                'significance': toys_significance, 
                'error':toys_sig_spread,
                'ntoys':ntoys}
    
    incl_syst_dict= {'BFs': BFs,
                'significance': significance_incl_error,
                'incl_other_syst':incl_other_syst}
    
    BDT_cuts_dict= {'BFs':BFs,
                    'light':light_cut,
                    'heavy':heavy_cut}

    #save dictionaries
    with open(os.path.join(set_outputpath(dict_path),'naive_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(naive_dict, fp)
 
    with open(os.path.join(set_outputpath(dict_path),'toys_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(toys_dict, fp)

    with open(os.path.join(set_outputpath(dict_path),'incl_syst_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(incl_syst_dict, fp)

    with open(os.path.join(set_outputpath(dict_path),'optimal_bdt_cuts_dict.pkl'), 'wb') as fp:
        pickle.dump(BDT_cuts_dict, fp)


    if plot == True: 
        sensitivity_CL_plotter(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=saveplotpath)

    return max_FOM, light_cut, heavy_cut, BFs, CL



def sensitivity_CL_plotter(naive_dict, incl_syst_dict, toys_dict = None, savepath=None):

    if toys_dict is None:
        incl_toys_fit=False
    else:
        incl_toys_fit=True

    #calculate 3 and 5 sigma for naive
    
    naive_five_sigma_BF = np.interp(5, naive_dict['significance'], naive_dict['BFs'])
    naive_three_sigma_BF = np.interp(3, naive_dict['significance'], naive_dict['BFs'])

    print(f'3sigma BF= {naive_three_sigma_BF}')
    print(f'5sigma BF= {naive_five_sigma_BF}')


    #plotting S/sqrt(S+B) line
    plt.figure()
    plt.plot(naive_dict['BFs'],naive_dict['significance'], label = 'Naive Counting \nExperiment')#r'$S/\sqrt{S+B}$') #
    plt.fill_between(naive_dict['BFs'], naive_dict['significance']-naive_dict['error'], naive_dict['significance']+naive_dict['error'],alpha=0.4)
    plt.vlines(x=naive_three_sigma_BF, ymin=0, ymax=3, linestyle='--')#,label=r'3$\sigma$ BF = '+f'{latex_form_exp(round_sig(naive_three_sigma_BF,sig=2))}')
    plt.vlines(x=naive_five_sigma_BF, ymin=0, ymax=5, linestyle='--')#,label=r'5$\sigma$ BF = '+f'{latex_form_exp(round_sig(naive_five_sigma_BF,sig=2))}')
   
    if incl_toys_fit == True:

        #interpolate toy fit
        interp_toys_significance = UnivariateSpline(list(toys_dict['BFs'])[::4], list(toys_dict['significance'])[::4], k = 2,s=0.1)
        #interp_BF_for_significance = UnivariateSpline(list(toys_dict['significance'])[::4],list(toys_dict['BFs'])[::4], k = 2,s=0.1)
        interp_toys_significance_upper = UnivariateSpline(list(toys_dict['BFs'])[::4], list(np.add(toys_dict['significance'],toys_dict['error']))[::4], k = 2,s=0.1)
        interp_toys_significance_lower = UnivariateSpline(list(toys_dict['BFs'])[::4], list(np.subtract(toys_dict['significance'],toys_dict['error']))[::4], k = 2,s=0.1)
        #five_sigma_toys = interp_BF_for_significance(5)
        #three_sigma_toys = interp_BF_for_significance(3)
        five_sigma_toys = find_x_for_y(5, interp_toys_significance, x_bounds = (min(list(toys_dict['BFs'])[::4]), max(list(toys_dict['BFs'])[::4])))
        three_sigma_toys = find_x_for_y(3, interp_toys_significance, x_bounds = (min(list(toys_dict['BFs'])[::4]), max(list(toys_dict['BFs'])[::4])))
        print(three_sigma_toys)
        print(f"5 sigma BFs toys = {five_sigma_toys}" )
        print(f"3 sigma toys= {three_sigma_toys}" )
        plot_BFs = np.logspace(np.log10(min(list(toys_dict['BFs'])[::4])), np.log10(max(list(toys_dict['BFs'])[::4])),len(naive_dict['BFs']))
        plt.plot(plot_BFs,interp_toys_significance(plot_BFs), color='green',label='Binned Fits \n to Toys')# ,label = r'Mean Toy $\sqrt{-2\Delta\ln{\mathcal{L}}}$')#
        plt.fill_between(plot_BFs, interp_toys_significance_lower(plot_BFs), interp_toys_significance_upper(plot_BFs),alpha=0.4, color='green')
        plt.vlines(x=three_sigma_toys, ymin=0, ymax=3, linestyle='--', color='green')#, label=r'3$\sigma$ BF = '+f'{latex_form_exp(round_sig(three_sigma_toys,sig=2))}')
        plt.vlines(x=five_sigma_toys, ymin=0, ymax=5, linestyle='--',color='green')#, label=r'5$\sigma$ BF = '+f'{latex_form_exp(round_sig(five_sigma_toys,sig=2))}', color='green')
        
        plt.xlim(min(toys_dict['BFs']), 4e-6)#max(toys_dict['BFs']))

    incl_other_syst = incl_syst_dict['incl_other_syst']

    five_sigma_BF_inclerr = np.interp(5, incl_syst_dict['significance'], incl_syst_dict['BFs'])
    three_sigma_BF_inclerr = np.interp(3, incl_syst_dict['significance'], incl_syst_dict['BFs'])

    print(f'3sigma incl err BF= {three_sigma_BF_inclerr}')
    print(f'5sigma incl err BF= {five_sigma_BF_inclerr}')


    if incl_other_syst == True:
        plt.plot(incl_syst_dict['BFs'],incl_syst_dict['significance'], color='orange',label='Including MC \nSample Size \nUncertainty')#,label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$')##+ '\n including '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})}$, $\sigma_{\varepsilon_i}$
        plt.fill_between(incl_syst_dict['BFs'], incl_syst_dict['significance'], incl_syst_dict['significance'],alpha=0, color='orange')
        plt.vlines(x=three_sigma_BF_inclerr, ymin=0, ymax=3, linestyle='--', color='orange')#, label=r'3$\sigma$ BF = '+f'{latex_form_exp(round_sig(three_sigma_BF_inclerr,sig=2))}')
        plt.vlines(x=five_sigma_BF_inclerr, ymin=0, ymax=5, linestyle='--', color='orange')#, label=r'5$\sigma$ BF = '+f'{latex_form_exp(round_sig(five_sigma_BF_inclerr,sig=2))}')
        plt.hlines(y = 3,xmin = min(toys_dict['BFs']),xmax = three_sigma_BF_inclerr, linestyle='--',color='k',alpha = 0.3)
        plt.hlines(y = 5,xmin = min(toys_dict['BFs']),xmax = five_sigma_BF_inclerr, linestyle='--',color='k',alpha = 0.3)
    else:
        plt.plot(incl_syst_dict['BFs'],incl_syst_dict['significance'],label='Including only MC Sample \nSize Uncertainty')#, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n neglecting '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})} $')
        plt.fill_between(incl_syst_dict['BFs'], incl_syst_dict['significance'], incl_syst_dict['significance'],alpha=0)
    
    plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.ylabel(r'Significance')
    plt.xscale('log')
    plt.ylim(0,8)
    plt.xlim(2e-9,4e-6)

    # reordering the labels 
    handles, labels = plt.gca().get_legend_handles_labels() 
    # specify order 
    order = [0,1,2]
    # pass handle & labels lists along with order as below 
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize = 11)

    #plt.title(r'Optimum FOM as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    if incl_other_syst == True:
        if incl_toys_fit == True:
            plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_inclfullerr_wtoys.pdf'))
        else:
            plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_inclfullerr.pdf'))

    else:
        if incl_toys_fit == True:
            plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_wtoys.pdf'))
        else:
            plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF.pdf'))



    ##################################
    # Now plotting CL 
    ##################################
        #convert sigma to CL - for now one sided
    
    CL = sigma_to_percentage(naive_dict['significance'])
    CL_incl_error = sigma_to_percentage(incl_syst_dict['significance']) # this dependent on if have full error or not above
    #CL_toys = sigma_to_percentage(toys_significance)

    # using interpolation
    CL95BF= np.interp(95, CL, naive_dict['BFs'])
    CL90BF = np.interp(90, CL, naive_dict['BFs'])
    print(f"95% CL BFs = {CL95BF}" )
    print(f"90% CL BFs = {CL90BF}" )

    CL95BF_inclerr= np.interp(95, CL_incl_error, naive_dict['BFs'])
    CL90BF_inclerr = np.interp(90, CL_incl_error, naive_dict['BFs'])
    print(f"95% CL BFs incl error = {CL95BF_inclerr}" )
    print(f"90% CL BFs incle error = {CL90BF_inclerr}" )


    plt.figure()
    plt.plot(naive_dict['BFs'],CL,label='Naive Counting \nExperiment')#,label='$S/\sqrt{S+B}$' )
    plt.fill_between(naive_dict['BFs'], sigma_to_percentage(naive_dict['significance']-naive_dict['error']),  sigma_to_percentage(naive_dict['significance']+naive_dict['error']),alpha=0.4)
    plt.vlines(x=CL90BF, ymin=70, ymax=90, linestyle='--')#, label=r'90$\%$ BF = '+f'{latex_form_exp(round_sig(CL90BF,sig=2))}')
    plt.vlines(x=CL95BF, ymin=70, ymax=95, linestyle='--')#, label=r'95$\%$ BF = '+f'{latex_form_exp(round_sig(CL95BF,sig=2))}')
    
    if incl_toys_fit == True:

        CL_toys = sigma_to_percentage(toys_dict['significance'])

        #interpolate toy fit
        interp_CL_toys = UnivariateSpline(list(toys_dict['BFs'])[::4], list(CL_toys)[::4], k = 2,s=1.3)

        #interp_BF_for_CL_toys = UnivariateSpline(CL_toys, toys_BFs, k = 2)
        interp_CL_toys_upper = UnivariateSpline(list(toys_dict['BFs'])[::4], list(sigma_to_percentage(np.add(toys_dict['significance'],toys_dict['error'])))[::4], k = 2,s=1.3)
        interp_CL_toys_lower = UnivariateSpline(list(toys_dict['BFs'])[::4], list(sigma_to_percentage(np.subtract(toys_dict['significance'],toys_dict['error'])))[::4], k = 2,s=1.3)

        #CL95_toys = interp_BF_for_CL_toys(95)
        #CL90_toys = interp_BF_for_CL_toys(90)
        #CL95_toys = np.interp(95, CL_toys, toys_BFs) - if use this need .item() in lable
        #CL90_toys = np.interp(90, CL_toys, toys_BFs)
        CL95_toys = find_x_for_y(95, interp_CL_toys, x_bounds = (min(list(toys_dict['BFs'])[::4]), max(list(toys_dict['BFs'])[::4])))
        CL90_toys = find_x_for_y(90, interp_CL_toys, x_bounds = (min(list(toys_dict['BFs'])[::4]), max(list(toys_dict['BFs'])[::4])))
        print(f"95% BF from toys = {CL95_toys}" )
        print(f"90% BF from toys= {CL90_toys}" )

        plt.plot(plot_BFs,interp_CL_toys(plot_BFs), color='green',label='Binned Fits \nto Toys')#,label = r' Mean Toy $\sqrt{-2\Delta\ln{\mathcal{L}}}$')#
        plt.fill_between(plot_BFs, interp_CL_toys_lower(plot_BFs), interp_CL_toys_upper(plot_BFs),alpha=0.4, color='green')
        plt.vlines(x=CL90_toys, ymin=70, ymax=90, linestyle='--', color='green')#, label=r'90$\%$ BF = '+f'{latex_form_exp(round_sig(CL90_toys,sig=2))}')
        plt.vlines(x=CL95_toys, ymin=70, ymax=95, linestyle='--', color='green')#, label=r'95$\%$ BF = '+f'{latex_form_exp(round_sig(CL95_toys,sig=2))}')
        plt.xlim(min(toys_dict['BFs']),4e-6)# max(toys_dict['BFs']))


    if incl_other_syst == True:
        plt.plot(naive_dict['BFs'],CL_incl_error, color='orange',label='Including MC \n Sample Size \nUncertainty')#, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$')##+ '\n including '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})}$, $\sigma_{\varepsilon_i}$
        plt.fill_between(naive_dict['BFs'], CL_incl_error, CL_incl_error,alpha=0, color='orange')
        plt.vlines(x=CL90BF_inclerr, ymin=70, ymax=90, linestyle='--', color='orange')#, label=r'90$\%$ BF = '+f'{latex_form_exp(round_sig(CL90BF_inclerr,sig=2))}')
        plt.vlines(x=CL95BF_inclerr, ymin=70, ymax=95, linestyle='--', color='orange')#, label=r'95$\%$ BF = '+f'{latex_form_exp(round_sig(CL95BF_inclerr,sig=2))}')
        plt.hlines(y = 90,xmin = min(toys_dict['BFs']),xmax = CL90BF_inclerr, linestyle='--',color='k',alpha = 0.3)
        plt.hlines(y = 95,xmin = min(toys_dict['BFs']),xmax = CL95BF_inclerr, linestyle='--',color='k',alpha = 0.3)
    else:
        plt.plot(naive_dict['BFs'],CL_incl_error, label = r'$S/\sqrt{S+B+\sigma_S^2+\sigma_B^2}$'+ '\n neglecting '+r'$\sigma_{\mathcal{B}(Z \rightarrow q\bar{q})} $')
        plt.fill_between(naive_dict['BFs'], CL_incl_error, CL_incl_error,alpha=0)
        
    plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.ylabel(r'Rejection CL')
    plt.xscale('log')
    plt.ylim(70,102)
    plt.xlim(2e-9,4e-6)
    # reordering the labels 
    handles, labels = plt.gca().get_legend_handles_labels() 
    # specify order 
    order = [0,1,2] 
    # pass handle & labels lists along with order as below 
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize = 11)

    #plt.title(r'1-CL as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    if incl_other_syst == True:
        if incl_toys_fit == True:
            plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_inclfullerr_wtoys.pdf'))
        else:
            plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_inclfullerr.pdf'))

    else:
        if incl_toys_fit == True:
            plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_wtoys.pdf'))
        else:
            plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF.pdf'))


def return_fullselneff_for_BF(interp_N_dict, signal_BF,lrange_plot= (0.995,1),hrange_plot = (0.995,1), nlh=200):
    
    FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF  = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=signal_BF, incl_other_syst=True)

    ## finding optimum cut to then calc eff.
    indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
    l = lsearch[indices[0]]
    h = hsearch[indices[1]]

    #create efficiency dictionary from interpolated N which can then print!
    interp_eff_dict, interp_eff_err_dict = interp_N_to_eff_err(interp_N_dict, lrange=(l,l) ,hrange=(h,h),nlh_plot=1,eventsProcessed_dict = cfg.eventsProcessed,savepath=None)

    return interp_eff_dict,interp_eff_err_dict



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
    
    
    plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation'
    
    #plot_interpolted_effs(interp_N_dict, N_dict ,nlh=20, slice=True, save_path=plotpath)
    #FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BF=1e-7)
    #plot_2d_optimisation(FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, vmax=20,SB_plots = True, save_path=plotpath)
    #per_sample_n_expect_dict, S, B, S_err, B_err, signal_BF  = make_final_binning_plot(full_data, interp_N_dict, signal_BF=1e-6, histbins=(2,2), nMC_plots_path=f'{plotpath}final_binning/1e-6/', final_plot_path = f'{plotpath}final_binning/1e-6/')

    
    plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(np.log10(1e-9),np.log10(4e-6),300), incl_other_syst=True, incl_toys_fit=True, full_df=full_data, ntoys=10000,plot=True,saveplotpath = plotpath ,toyplotpath = plotpath)
    
    