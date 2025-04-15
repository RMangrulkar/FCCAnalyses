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
from scipy.stats import norm

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

            
        if slice == True:
            i=6

            for decay in interp_eff_dict.keys():
                        #to remove ee
                if decay =='p8_ee_Zee_ecm91':
                    continue

        
                plt.figure()
                plt.plot(hsearch,eff_dict[decay][round(nlh*i/10),:], label=r'raw efficiency slice 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
                eff_interp_slice = interp_eff_dict[decay][round((nlh*i/10)*10),:]
                plt.plot(hsearchinterp,eff_interp_slice, label=r'interpolated efficiency slice 1-P(light) $>$'+f'{round(lsearchinterp[round((nlh*i/10)*10)],5)}')
                plt.xlabel('BDT_lh 1-P(heavy)')
                plt.ylabel(r'Efficiency')
                plt.legend()
                plt.savefig(os.path.join(set_outputpath(save_path),f'efficinecy_{decay}_heavy.pdf'))


                plt.figure()
                plt.plot(lsearch,eff_dict[decay][:,round(nlh*i/10)], label=r'raw efficiency slice 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
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


  
def run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=1000 , sig_BF=1e-7):
    
    lsearch = np.linspace(*lrange_plot,nlh) #need to be the same lrange and hrange as efficiency map was generated with 
    hsearch = np.linspace(*hrange_plot,nlh)

    #create efficiency deictionary from interpolated N
    interp_eff_dict, interp_eff_err_dict = interp_N_to_eff_err(interp_N_dict, lrange=lrange_plot ,hrange=hrange_plot,nlh_plot=nlh,eventsProcessed_dict = cfg.eventsProcessed,savepath=None)


    S_arr = np.zeros((len(lsearch), len(hsearch)))
    B_arr = np.zeros((len(lsearch), len(hsearch)))
    FOM = np.zeros((len(lsearch), len(hsearch)))

    print('Calculating S and B')

    #defining constants needed
    N_z = cfg.N_z
    k = 2 * N_z * cfg.branching_fractions["p8_ee_Zbb_ecm91"][0] * sig_BF # common part of signal expectation
    
    for l in np.arange(0,len(lsearch), 1):
        for h in np.arange(0,len(hsearch), 1): 
            S = k * sum([cfg.prod_frac[decay][0]*interp_eff_dict[decay][l,h] for decay in cfg.sample_allocations["combined_signal"]])
            B = N_z* sum([cfg.branching_fractions[decay][0]*interp_eff_dict[decay][l,h] for decay in cfg.sample_allocations["hadronic_background"]])
            S_arr[l,h]=S
            B_arr[l,h]=B
            if S+B>0:
                FOM[l,h] = S/np.sqrt(S+B)
            else:
                FOM[l,h] = 0

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
     
    return FOM, S_arr, B_arr, lsearch, hsearch, sig_BF


def plot_2d_optimisation(FOM, S_arr, B_arr, lsearch, hsearch, sigBF,vmax=5,SB_plots = False, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/'):
    
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
    plt.plot(hsearch,FOM[indices[0],:], label=r'FOM slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
    plt.xlabel('BDT_lh 1-P(heavy)')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.legend()
    plt.title('FOM slice with for common signal BF = '+f'{sigBF:.2e}')
    plt.savefig(os.path.join(save_path,f'FOM_slice_heavy.pdf'))

    plt.figure()
    plt.plot(lsearch,FOM[:,indices[1]], label=r'FOM slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
    plt.xlabel('BDT_lh 1-P(light)')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.legend()
    plt.title('FOM slice with for common signal BF = '+f'{sigBF:.2e}')
    plt.savefig(os.path.join(save_path,f'FOM_slice_light.pdf'))



    if SB_plots == True:
        #plot S
        plt.figure()
        plt.plot(lsearch,S_arr[:,indices[1]], label=r'S slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'S')
        plt.legend()
        plt.title('Signal expectation slice with for common signal BF = '+f'{sigBF:.2e}')
        plt.savefig(os.path.join(save_path,f'S_slice_light.pdf'))

        plt.figure()
        plt.plot(hsearch,S_arr[indices[0],:], label=r'S slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'S')
        plt.legend()
        plt.title('Signal expectation slice with for common signal BF = '+f'{sigBF:.2e}')
        plt.savefig(os.path.join(save_path,f'S_slice_heavy.pdf'))
        
        #plot B
        plt.figure()
        plt.plot(lsearch,B_arr[:,indices[1]], label=r'B slice at optimum cut in 1-P(heavy) $>$'+f'{round(h,5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'B')
        plt.legend()
        plt.savefig(os.path.join(save_path,f'B_slice_light.pdf'))

        plt.figure()
        plt.plot(hsearch,B_arr[indices[0],:], label=r'B slice at optimum cut in 1-P(light) $>$'+f'{round(l,5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'B')
        plt.legend()
        plt.savefig(os.path.join(save_path,f'B_slice_heavy.pdf'))


def plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BFs=np.logspace(-9,-4,250), plot=True,savepath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995'):
    
    #create dictionaries to store results
    max_FOM=np.zeros(len(sig_BFs))
    BFs=np.zeros(len(sig_BFs))
    light_cut=np.zeros(len(sig_BFs))
    heavy_cut=np.zeros(len(sig_BFs))
    i=0

    for BF in sig_BFs:
        
        FOM, S_arr, B_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_N_dict,lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF)

        ## finding maximum so can plot slices
        indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
        l = lsearch[indices[0]]
        h = hsearch[indices[1]]

        max_FOM[i] = FOM.max()
        light_cut[i] = l
        heavy_cut[i] = h    
        BFs[i] = sig_BF
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


    if plot==True:

        #plotting
        plt.figure()
        plt.plot(BFs,max_FOM)
        plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.ylabel(r'$S/\sqrt{S+B}$')
        plt.xscale('log')
        plt.ylim(0,8)

        # Add vertical and horizontal lines stopping at the points
        plt.vlines(x=highlight_x_5, ymin=0, ymax=highlight_y_5, color='deeppink', linestyle='--', label=r'5$\sigma$')
        plt.hlines(y=highlight_y_5, xmin=min(BFs), xmax=highlight_x_5, color='deeppink', linestyle='--')

        plt.vlines(x=highlight_x_3, ymin=0, ymax=highlight_y_3, color='purple', linestyle='--', label=r'3$\sigma$')
        plt.hlines(y=highlight_y_3, xmin=min(BFs), xmax=highlight_x_3, color='purple', linestyle='--')

        plt.legend()
        plt.title(r'Optimum FOM as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.savefig(os.path.join(savepath,f'FOMvsBF.pdf'))


    
    #convert sigma to CL - for now one sided

    CL = sigma_to_percentage(max_FOM)

    #find 90 sigma and 95 % points

    closest_index_90 = np.argmin(np.abs(np.array(CL) - 90))
    highlight_x_90 = BFs[closest_index_90]
    highlight_y_90 = CL[closest_index_90]

    closest_index_95 = np.argmin(np.abs(np.array(CL) - 95))
    highlight_x_95 = BFs[closest_index_95]
    highlight_y_95 = CL[closest_index_95]


    if plot == True: 
        plt.figure()
        plt.plot(BFs,CL)
        plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.ylabel(r'1-CL (1-sided test)')
        plt.xscale('log')

        # Add vertical and horizontal lines stopping at the points
        plt.vlines(x=highlight_x_95, ymin=52, ymax=highlight_y_95, color='deeppink', linestyle='--', label=r'95$\%$')
        plt.hlines(y=highlight_y_95, xmin=min(BFs), xmax=highlight_x_95, color='deeppink', linestyle='--')

        plt.vlines(x=highlight_x_90, ymin=52, ymax=highlight_y_90, color='purple', linestyle='--', label=r'90$\%$')
        plt.hlines(y=highlight_y_90, xmin=min(BFs), xmax=highlight_x_90, color='purple', linestyle='--')
        plt.ylim(52,102)
        plt.legend()
        plt.title(r'1-CL as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
        plt.savefig(os.path.join(savepath,f'CLvsBF.pdf'))

    print(f"95% BFs = {BFs[closest_index_95]}" )
    print(f"90% BFs = {BFs[closest_index_90]}" )

    return max_FOM, light_cut, heavy_cut, BFs, CL
    





if __name__=="__main__":

    '''
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

    N_dict, interp_N_dict, s_values_dict, eff_dict, err_dict = create_N_map(full_data,lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, smoothing=False, kx=2, ky=2, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/')
    plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,slice=True, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/')
    '''

    save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/'
    
    with open(os.path.join(set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
        interp_N_dict = dill.load(dill_file)

    with open(os.path.join(set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)
    
    
    #plot_interpolted_effs(interp_N_dict, N_dict ,nlh=20, slice=True, save_path=plotpath)
    plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation'
    
    #FOM, S_arr, B_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BF=1e-7)
    #plot_2d_optimisation(FOM, S_arr, B_arr, lsearch, hsearch, sig_BF, vmax=20,SB_plots = True, save_path=plotpath)
    
    plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=500 , sig_BFs=np.logspace(-9,-4,250), plot=True,savepath = plotpath)
    
