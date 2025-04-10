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

def make_n_remaining_plots(df,lrange=(0.99,1) ,hrange=(0.99,1),nlh=40, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/N_MC/', N_plot=True,eff_plot=False): 
    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)
    
    #define lists need etc
    eff_dict={}
    eff_err_dict={}
    N_remaining_dict={}
    print('Creating efficiency map for decay:')
    
    for decay in df["decay"].unique():
        #to remove when want to include Bd too#########################
        if decay =='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu':
            continue
        elif decay =='p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu':
            continue
        ###############################################################
        print(decay)
        
        eff_df=np.zeros((len(lsearch), len(hsearch)))
        err_df=np.zeros((len(lsearch), len(hsearch)))
        N_df=np.zeros((len(lsearch), len(hsearch)))
        
        subf = df[df["decay"]==decay]
        l_arr=[]
        h_arr=[]
    
        for l in np.arange(0,len(lsearch),1):
            for h in np.arange(0,len(hsearch),1):
                l_arr.append(lsearch[l])
                h_arr.append(hsearch[h])
                lh_cut = f"(P_not_heavy>{hsearch[h]})&(P_not_light>{lsearch[l]})"

                eff_lh, err_lh, N_remaining = eff_finder.get_total_eff_post_bdt(subf, cut= lh_cut, verbose = False, eventsProcessed_dict = cfg.eventsProcessed)
                eff_df[l,h] = eff_lh[decay]
                err_df[l,h] = err_lh[decay] 
                N_df[l,h] = N_remaining[decay]   
    
        eff_dict[decay] = eff_df
        eff_err_dict[decay] = err_df
        N_remaining_dict[decay] = N_df

    if N_plot == True:
        for decay in N_remaining_dict.keys():
            plt.figure()
            for i in range(1,11,2):
                plt.plot(hsearch,N_remaining_dict[decay][round(nlh*i/10),:], label=r'fixed cut on 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(heavy)')
            plt.ylabel(r'Number of MC events remaining')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_{decay}_heavy.pdf'))


            plt.figure()
            for i in range(1,11,2):
                plt.plot(lsearch,N_remaining_dict[decay][:,round(nlh*i/10)], label=r'fixed cut on 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(light)')
            plt.ylabel(r'Number of MC events remaining')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_{decay}_light.pdf'))


        plt.figure()
        for decay in N_remaining_dict.keys():
            plt.plot(hsearch,N_remaining_dict[decay][round(nlh*9/10),:], label=decay+r'fixed cut on 1-P(light) $>$'+f'{round(lsearch[round(nlh*9/10)],5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'Number of MC events remaining')
        plt.legend()
        plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_allbkg_heavy.pdf'))


        plt.figure()
        for decay in N_remaining_dict.keys():
            plt.plot(lsearch,N_remaining_dict[decay][:,round(nlh*9/10)], label=decay+r'fixed cut on 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*9/10)],5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'Number of MC events remaining')
        plt.legend()
        plt.savefig(os.path.join(set_outputpath(save_path),f'N_MC_events_remaining_allbkg_light.pdf'))

    if eff_plot == True:

        for decay in eff_dict.keys():
            plt.figure()
            for i in range(1,11,2):
                plt.errorbar(hsearch,eff_dict[decay][round(nlh*i/10),:], eff_err_dict[decay][round(nlh*i/10),:],label=r'fixed cut on 1-P(light) $>$'+f'{round(lsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(heavy)')
            plt.ylabel(r'Total efficiency')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'Total_efficiency_{decay}_heavy.pdf'))


            plt.figure()
            for i in range(1,11,2):
                plt.errorbar(lsearch,eff_dict[decay][:,round(nlh*i/10)], eff_err_dict[decay][:,round(nlh*i/10)],eff_err_dict[decay][:,round(nlh*i/10)], label=r'fixed cut on 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*i/10)],5)}')
            plt.xlabel('BDT_lh 1-P(light)')
            plt.ylabel(r'Total efficiency')
            plt.legend()
            plt.savefig(os.path.join(set_outputpath(save_path),f'Total_efficiency_{decay}_light.pdf'))


        plt.figure()
        for decay in eff_dict.keys():
            plt.errorbar(hsearch,eff_dict[decay][round(nlh*9/10),:], eff_err_dict[decay][round(nlh*9/10),:], label=decay+r'fixed cut on 1-P(light) $>$'+f'{round(lsearch[round(nlh*9/10)],5)}')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel(r'Total efficiency')
        plt.legend()
        plt.savefig(os.path.join(set_outputpath(save_path),f'efficiency_allbkg_heavy.pdf'))


        plt.figure()
        for decay in eff_dict.keys():
            plt.errorbar(lsearch,eff_dict[decay][:,round(nlh*9/10)], eff_err_dict[decay][:,round(nlh*9/10)], label=decay+r'fixed cut on 1-P(heavy) $>$'+f'{round(hsearch[round(nlh*9/10)],5)}')
        plt.xlabel('BDT_lh 1-P(light)')
        plt.ylabel(r'Number of MC events remaining')
        plt.legend()
        plt.savefig(os.path.join(set_outputpath(save_path),f'efficiency_allbkg_light.pdf'))




    return eff_dict, eff_err_dict, N_remaining_dict




def make_interpolated_eff_map(df,lrange=(0.99,1) ,hrange=(0.99,1),nlh=40, smoothing=False, kx=3,ky=3, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/'): #df should be full data for final result

    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)
    
    #define lists need etc
    eff_dict={}
    eff_err_dict={}
    interp_eff_dict={}
    s_values_dict={}
    N_dict={}
    print('Creating efficiency map for decay:')
    
    for decay in df["decay"].unique():
        #to remove when want to include Bd too#########################
        #if decay =='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu':
        #    continue
        ###############################################################
        print(decay)
        
        eff_df=np.zeros((len(lsearch), len(hsearch)))
        err_df=np.zeros((len(lsearch), len(hsearch)))
        N_df=np.zeros((len(lsearch), len(hsearch)))
        
        subf = df[df["decay"]==decay]
        l_arr=[]
        h_arr=[]
    
        for l in np.arange(0,len(lsearch),1):
            for h in np.arange(0,len(hsearch),1):
                l_arr.append(lsearch[l])
                h_arr.append(hsearch[h])
                lh_cut = f"(P_not_heavy>{hsearch[h]})&(P_not_light>{lsearch[l]})"

                eff_lh, err_lh, N_remaining = eff_finder.get_total_eff_post_bdt(subf, cut= lh_cut, verbose = False, eventsProcessed_dict = cfg.eventsProcessed)
                eff_df[l,h] = eff_lh[decay]
                err_df[l,h] = err_lh[decay]
                N_df[l,h] = N_remaining[decay]    
    
        eff_dict[decay] = eff_df
        eff_err_dict[decay] = err_df
        N_dict[decay] = N_df

        if smoothing == True:
            s_value = np.sum(np.square(err_df))
            s_values_dict[decay] = s_value
        else:
            s_value = 0
            s_values_dict[decay] = s_value

        interp_eff = RectBivariateSpline(lsearch, hsearch, eff_df,kx=kx, ky=ky, s=s_value)
        interp_eff_dict[decay] = interp_eff

    print(s_values_dict)

    with open(os.path.join(set_outputpath(save_path),"efficiencies_dictionary"), "wb") as dill_file:
        dill.dump(eff_dict, dill_file)

    with open(os.path.join(set_outputpath(save_path),"interpolated_efficiencies_dictionary"), "wb") as dill_file2:
        dill.dump(interp_eff_dict, dill_file2)

    with open(os.path.join(set_outputpath(save_path),"efficiency_errors_dictionary"), "wb") as dill_file3:
        dill.dump(eff_err_dict, dill_file3)

    with open(os.path.join(set_outputpath(save_path),"N_remaining_dictionary"), "wb") as dill_file4:
        dill.dump(N_dict, dill_file4)

    return eff_dict, interp_eff_dict, eff_err_dict, s_values_dict



def make_eff_plots(eff_dict, interp_eff_dict,eff_err_dict, lrange=(0.99,1) ,  hrange=(0.99,1),  nlh=40, lrangeplot=(0.99,1) ,hrangeplot=(0.99,1),slice = True, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/',nlh_plot=40, vmin=-5,vmax=5):
    

    lsearch = np.linspace(*lrange,nlh) 
    hsearch = np.linspace(*hrange,nlh)

    lplot = lsearch[-nlh_plot:] #np.linspace(*lrangeplot,nlh_plot) 
    hplot = hsearch[-nlh_plot:] #np.linspace(*hrangeplot,nlh_plot)       

    for decay in interp_eff_dict.keys():
        
        #to remove ee
        if decay =='p8_ee_Zee_ecm91':
            continue
       
        lsearchinterp = np.linspace(*lrangeplot,nlh_plot*10) 
        hsearchinterp = np.linspace(*hrangeplot,nlh_plot*10)

        fine_grid_splined =np.zeros((len(lsearchinterp), len(hsearchinterp)))
        splined=np.zeros((len(lplot), len(hplot)))
        pull=np.zeros((len(lplot), len(hplot)))

        error = eff_err_dict[decay][-nlh_plot:, -nlh_plot:]
        E = eff_dict[decay][-(nlh_plot):, -(nlh_plot):]
        #print(E[0,0])
        #print(eff_dict[f"{decay}"][20,20])
    
        for l in np.arange(0,len(lplot),1):
            for h in np.arange(0,len(hplot),1):
                splined[l,h] = interp_eff_dict[f"{decay}"](lplot[l],hplot[h],grid=False)
                if error[l,h]>0:
                    pull[l,h] = (E[l,h] - splined[l,h])/error[l,h]
                else:
                    pull[l,h]=0


        for l in np.arange(0,len(lsearchinterp),1):
            for h in np.arange(0,len(hsearchinterp),1):
                fine_grid_splined[l,h] = interp_eff_dict[f"{decay}"](lsearchinterp[l],hsearchinterp[h],grid=False)
                

        #plotting 2D pulls

        plt.figure()
        plt.imshow(pull, origin='lower',vmin=vmin,vmax=vmax)
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='pulls')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lplot), round(len(lplot)/5))
        xtick_indices = np.arange(0, len(hplot), round(len(hplot)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lplot[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hplot[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'Efficiency spline pulls for {decay} decay')
        plt.savefig(os.path.join(set_outputpath(save_path),f'2D_pulls_{decay}.pdf'))


        #plotting splined efficiencies themselves (finely gridded)

        plt.figure()
        plt.imshow(fine_grid_splined, origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='efficiency')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lsearchinterp), round(len(lsearchinterp)/5))
        xtick_indices = np.arange(0, len(hsearchinterp), round(len(hsearchinterp)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lsearchinterp[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hsearchinterp[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'Efficiency spline pulls for {decay} decay')
        plt.savefig(os.path.join(set_outputpath(save_path),f'splined_efficiency_{decay}.pdf'))

        # plotting raw efficiencies
    

        plt.figure()
        plt.imshow(E, origin='lower')
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='efficiency')
        
        # Set tick labels for every 10th bin
        ytick_indices = np.arange(0,len(lplot), round(len(lplot)/5))
        xtick_indices = np.arange(0, len(hplot), round(len(hplot)/5))
        
        # Use ytick_indices and xtick_indices to set the ticks
        plt.yticks(ytick_indices, [round(lplot[i],5) for i in ytick_indices])
        plt.xticks(xtick_indices, [round(hplot[i],5) for i in xtick_indices], rotation=90)
        plt.title(f'Efficiency spline pulls for {decay} decay')
        plt.savefig(os.path.join(set_outputpath(save_path),f'raw_efficiency_{decay}.pdf'))

        #plotting in region care about
        """

        # Assuming pull, lsearch, and hsearch are already defined
        num_bins = 20  # Number of bins to keep

        # Select the last 20x20 bins from the pull array
        pull_subset = pull[-num_bins:, -num_bins:]
        lsearch_subset = lsearch[-num_bins:]  # Last 20 values of lsearch
        hsearch_subset = hsearch[-num_bins:]  # Last 20 values of hsearch

        plt.figure()
        plt.imshow(pull_subset, origin='lower', vmin=-5, vmax=5, extent=[
            hsearch_subset[0], hsearch_subset[-1],  # X-axis range
            lsearch_subset[0], lsearch_subset[-1]   # Y-axis range
        ])
        plt.xlabel('BDT_lh 1-P(heavy)')
        plt.ylabel('BDT_lh 1-P(light)')
        plt.colorbar(label='pulls')

        # Set tick labels appropriately
        plt.xticks(np.linspace(hsearch_subset[0], hsearch_subset[-1], 5),
           [round(val, 5) for val in np.linspace(hsearch_subset[0], hsearch_subset[-1], 5)],
           rotation=90)
        plt.yticks(np.linspace(lsearch_subset[0], lsearch_subset[-1], 5),
           [round(val, 5) for val in np.linspace(lsearch_subset[0], lsearch_subset[-1], 5)])
        plt.savefig(os.path.join(set_outputpath(save_path),f'2D_pulls_very_tight_{decay}.pdf'))
        """


        if slice == True:
            n = 10
            fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6,6))
            #ax[0].plot(hsearch,eff_dict[f"{decay}"][n,:], label=r'efficiency slice at 1-P(light) $>$'+ f'{lsearch[n]}',color='b')
            ax[0].errorbar(hplot[:],E[n, :],error[n,:],label=r'efficiency slice at 1-P(light) $>$'+ f'{lplot[n]}')
            ax[0].errorbar(hsearchinterp,interp_eff_dict[f"{decay}"](lplot[n],hsearchinterp,grid=False), label=r'interpolated efficiency slice a 1-P(light) $>$'+ f'{lplot[n]}')
        
            ax[1].errorbar( hplot[:], pull[n, :], np.ones_like(pull[n, :]) )
            ax[1].axhline(0, c='k', ls='--' )   
            ax[1].set_ylabel('Pull')
            ax[1].set_ylim(-3,3)
            ax[0].set_xlabel('BDT_lh 1-P(heavy)')
            ax[0].set_ylabel(f'{decay} efficiency')
            ax[0].legend()
            fig.tight_layout()
            plt.savefig(os.path.join(save_path,f'heavy_slice_{decay}.pdf'))

            
            fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6,6))
            #ax[0].plot(lsearch,eff_dict[f"{decay}"][:,n], label=r'efficiency slice at 1-P(heavy) $>$'+ f'{hsearch[n]}',color='b')
            ax[0].errorbar(lplot[:],E[:,n],error[ :,n],label=r'efficiency slice at 1-P(heavy) $>$'+ f'{hplot[n]}')
            ax[0].errorbar(lsearchinterp,interp_eff_dict[f"{decay}"](lsearchinterp,hplot[n],grid=False), label=r'interpolated efficiency slice a 1-P(heavy) $>$'+ f'{hplot[n]}')
        
            ax[1].errorbar( lplot[:], pull[ :,n], np.ones_like(pull[ :,n]) )
            ax[1].axhline(0, c='k', ls='--' )   
            ax[1].set_ylabel('Pull')
            ax[1].set_ylim(-3,3)
            ax[0].set_xlabel('BDT_lh 1-P(light)')
            ax[0].set_ylabel(f'{decay} efficiency')
            ax[0].legend()
            fig.tight_layout()
            plt.savefig(os.path.join(save_path,f'light_slice_{decay}.pdf'))


def run_2d_optimisation(interp_eff_dict,err_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nl=1000,nh=1000 , sig_BF=1e-7):
    
    lsearch = np.linspace(*lrange_plot,nl) #need to be the same lrange and hrange as efficiency map was generated with 
    hsearch = np.linspace(*hrange_plot,nh)

    S_arr = np.zeros((len(lsearch), len(hsearch)))
    B_arr = np.zeros((len(lsearch), len(hsearch)))
    FOM = np.zeros((len(lsearch), len(hsearch)))

    print('Calculating S and B')

    #defining constants needed
    N_z = cfg.N_z
    k = 2 * N_z * cfg.branching_fractions["p8_ee_Zbb_ecm91"][0] * sig_BF # common part of signal expectation
    
    for l in np.arange(0,len(lsearch), 1):
        for h in np.arange(0,len(hsearch), 1): 
            S = k * sum([cfg.prod_frac[decay][0]*interp_eff_dict[decay](lsearch[l],hsearch[h]).item() for decay in cfg.sample_allocations["combined_signal"]])
            B = N_z* sum([cfg.branching_fractions[decay][0]*interp_eff_dict[decay](lsearch[l],hsearch[h]).item() for decay in cfg.sample_allocations["hadronic_background"]])
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


def plot_BF_sensitivitise(interp_eff_dict,err_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nl=500,nh=500 , sig_BFs=np.logspace(1e-9,1e-4,250), savepath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995'):
    
    #create dictionaries to store results
    max_FOM=np.zeros(len(sig_BFs))
    BFs=np.zeros(len(sig_BFs))
    light_cut=np.zeros(len(sig_BFs))
    heavy_cut=np.zeros(len(sig_BFs))
    i=0

    for BF in sig_BFs:
        
        FOM, S_arr, B_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_eff_dict,err_dict,lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nl=nl,nh=nh , sig_BF=BF)

        ## finding maximum so can plot slices
        indices = np.unravel_index(np.argmax(FOM), np.shape(FOM)) #nb argmax returns indices of the max value
        l = lsearch[indices[0]]
        h = hsearch[indices[1]]

        max_FOM[i] = FOM.max()
        light_cut[i] = l
        heavy_cut[i] = h    
        BFs[i] = sig_BF
        i+=1


    #plotting

    plt.figure()
    plt.plot(BFs,max_FOM)
    plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.xscale('log')
    plt.ylim(0,8)
    plt.legend()
    plt.title(r'Optimum FOM as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.savefig(os.path.join(savepath,f'FOMvsBF.pdf'))


    print(f"5 sigma BFs =  {[BFs[k] for k in (np.where(np.array([round(max_FOM[i], 1) for i in range(len(max_FOM))]) == round(5.0, 1)))[0]]}")

    print(f"3 sigma BF =  {[BFs[k] for k in (np.where(np.array([round(max_FOM[i], 1) for i in range(len(max_FOM))]) == round(3.0, 1)))[0]]}")

    
    #convert sigma to CL - for now one sided

    CL = sigma_to_percentage(max_FOM)

    plt.figure()
    plt.plot(BFs,CL)
    plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.ylabel(r'1-CL (1-sided test)')
    plt.xscale('log')
    plt.legend()
    plt.title(r'1-CL as a function of $\mathcal{B}(B_{(s)}^0 \rightarrow$ invisibles$)$')
    plt.savefig(os.path.join(savepath,f'CLvsBF.pdf'))


    
    


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

    eff_dict, interp_eff_dict, eff_err_dict, s_values_dict = make_interpolated_eff_map(full_data,lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, smoothing=True, kx=2, ky=2, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/smoothing/0995/')
    '''
    
    
    save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995'

    with open(os.path.join(set_outputpath(save_path), "efficiencies_dictionary"), "rb") as dill_file:
        eff_dict = dill.load(dill_file)

    with open(os.path.join(set_outputpath(save_path), "interpolated_efficiencies_dictionary"), "rb") as dill_file:
        interp_eff_dict = dill.load(dill_file)

    with open(os.path.join(set_outputpath(save_path), "efficiency_errors_dictionary"), "rb") as dill_file:
        eff_err_dict = dill.load(dill_file)
    
    '''
    make_eff_plots(eff_dict, interp_eff_dict,eff_err_dict,lrangeplot=(0.995,1) ,hrangeplot=(0.995,1), nlh_plot = 20, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,slice = True, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/smoothing/0995/',vmin=-3,vmax=3)
    FOM, S_arr, B_arr, lsearch, hsearch, sig_BF = run_2d_optimisation(interp_eff_dict,eff_err_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nl=500,nh=500 , sig_BF=1e-7)
    plot_2d_optimisation(FOM, S_arr, B_arr, lsearch, hsearch, sig_BF, vmax=20,SB_plots = True, save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995')
    '''
    
    plot_BF_sensitivitise(interp_eff_dict,eff_err_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nl=1000,nh=1000 , sig_BFs=np.logspace(-9,-5,250), savepath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995')





#possible method fro adding extra weight to edges
'''
# Original grid
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)
z = some_function(x[:, None], y[None, :])  # shape (50, 50)

# Duplicate edge rows and columns to reinforce them
z_aug = np.vstack([
    z[0:1, :],       # top edge
    z,               # original
    z[-1:, :]        # bottom edge
])

x_aug = np.concatenate([
    [x[0] - 1e-5],   # very slightly outside to avoid singular knots
    x,
    [x[-1] + 1e-5]
])

# Now the spline will be more faithful to the top and bottom edges
spline = RectBivariateSpline(x_aug, y, z_aug, s=1.0)
'''