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
import textwrap

import config as cfg
from  basic_functions import set_outputpath
from efficiency_tools import efficiency_finder
from efficiency_tools import post_bdtlh_efficiency_finder as post_bdt_eff_finder
from basic_functions import flatten_list

# import basic functions
from bdt_lh_cut_opt_significance import sigma_to_percentage,round_sig, latex_form_exp, find_x_for_y

#import plotters and related functions
from bdt_lh_cut_opt_significance import sensitivity_CL_plotter, histogram_settings

#import optimisation and SB calculator
from bdt_lh_cut_opt_significance import run_2d_optimisation, calc_SB_from_opt_cut # note that this has been updated to allow cut_opt_samples input (ie. tell it which samples to run optimisation on)

plt.style.use('fcc.mplstyle')


def make_final_binning_plot_extra_bkgs(df, interp_N_dict,
                                       cut_opt_samples=None,  
                                       lrange_interp_N_dict=(0.999,1),
                                       hrange_interp_N_dict=(0.999,1),
                                       nlh = 200, signal_BF=1e-6, 
                                       eventsProcessed_dict = cfg.eventsProcessed, 
                                       histbins=(2,2), 
                                       components_to_plot =  ['hadronic_background','combined_signal'],
                                       model_all_1prong_leptonic_tau = True, 
                                       binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                       plot_signal_components=False,  
                                       nMC_plots_path=None, 
                                       final_plot_path = None, 
                                       pull_type_plot=False, 
                                       lcut=None, 
                                       hcut=None, 
                                       logpath= None):
    
    '''
    Function that creates and plots  the "final binning plot" for a given signal BF. 
    Imputs to function
    df: dataframe of all samples want to plot (used to make plot)
        dataframe used to calculate number of events per bin
    interp_N_dict: dictionary (used to find cut point)
        Dictionary of interpolated distributions for the number of events for a given BDT cut - output of create_N_map.
    lrange_interp_N_dict: tuple, optional
        Range of BDT_lh cut values for 1-P(l) interp_N_dict is produced over
    hrange_interp_N_dict:tuple, optional
        Range of BDT_lh cut values for 1-P(h) interp_N_dict is produced over
    nlh: integer, optional
        Number of bins (in each direction ie. 1-P(l) and 1-P(h)), want to run optimisation to find optimum cut over 
    cut_opt_samples: list, optional
        List of samples want to include in S and B values used to calculate optimal cut. Default:None - ie. use all samples in interp_N_dict keys list
    components_to_plot: list, optiona;
        List of Sample Allocations to use in plot. Note this can be different from those used in optimisation.
    signal_BF:float, optional
        Signal BF want to calculate optimisation of cut for and then also use in plot
    eventsProcessed_dict:dictionary, optional
        Dictionary of number of eventsProcessed by tupling, stored in config. Used to calculate selection efficiencies.
    histbins: tuple, optional
        Binning want to make final bin plot over in BDT space
    binned_x_axis: list,optional
        Names for bins that are flattened to give on xaxis
    plot_signal_components:bool, optional
        Flag as to whether to plot the signal components separeately or as a single combined signal
    nMC_plots_path: str, optional
        Path to save histograms of number of events per bin (in 2D BDT space) - if None, plots not saved
    final_plot_path: str, optional
        Path to save final binning plot. If None, not saved
    pull_type_plot:bool, optional
        Flag as to whether plot with signal-bkg benethe plot
    lcut: float, optional
        Optional cut point for 1-P(l), if None cut optimisation will be run
    hcut: float, optional
        Optional cut point for 1-P(h), if None cut optimisation will be run
    Note: need to specify both lcut and hcut for the input cut point to be used
    logpath:str, optional
        Path to save bin edges
    '''
    #turn BF into title worthy version
    latex_BF = latex_form_exp(signal_BF)
   
    print('--> Finding optimum cut')
    if lcut is not None and hcut is not None:
        l_cut = lcut
        h_cut = hcut
    else:
        # find optimum cut
        FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, = run_2d_optimisation(interp_N_dict, cut_opt_samples=cut_opt_samples, lrange_plot=lrange_interp_N_dict ,hrange_plot=hrange_interp_N_dict, nlh=nlh , sig_BF=signal_BF, incl_other_syst = False)
        
        indices = np.unravel_index(np.argmax(FOM), np.shape(FOM))
        l_cut = lsearch[indices[0]]
        h_cut = hsearch[indices[1]]

    #cut df on optimal BDT cuts
    cut_data = df.copy().query(f'(P_not_light>{l_cut})&(P_not_heavy>{h_cut})')

    # define samples want from components input 
    samples = flatten_list([cfg.sample_allocations[component] for component in components_to_plot])

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
    efficienies, efficiencies_err, N_dict_MC = post_bdt_eff_finder.get_eff_from_nMC_list(N_dict_MC, eventsProcessed_dict = eventsProcessed_dict)
    per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err  =post_bdt_eff_finder.get_n_expected_components(efficienies, efficiencies_err,signal_bf=signal_BF, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau)

    #check if have additional backgrounds to inclusive ones (ie. compare samples to inclusive backgrounds list in config)
    if len(list(set(samples).intersection(cfg.exclusive_backgrounds)))>0:
        excl_bkgs = list(set(samples).intersection(cfg.exclusive_backgrounds))
    else:
        excl_bkgs = None

    S, B, S_err, B_err = post_bdt_eff_finder.get_total_SB(per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err, incl_other_syst=True, exclusive_background_samples = excl_bkgs) # just used in final return efficiencies therefore want to include any other backgrounds here

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
        
                if allocation not in components_to_plot:
                    continue
                
                for sample in cfg.sample_allocations[allocation]:

                    if sample in cfg.sample_allocations['combined_signal']: #plot combined signal
                        h = per_sample_n_expect_dict[sample]
                        tot_signal = np.add(tot_signal,[h[1,0],h[0,0],h[0,1],h[1,1]])

                        if plot_signal_components == True:
                            hist_opts = histogram_settings()[allocation]
                            plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[sample+'_invis'], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        tot_arr = np.add(tot_arr,[h[1,0],h[0,0],h[0,1],h[1,1]])
                        i+=1

                    else:
                        h = per_sample_n_expect_dict[sample]
                        hist_opts = histogram_settings()[allocation]
                        
                        if model_all_1prong_leptonic_tau==True:
                            if sample in ("p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu"):
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}_1prong_leptonic_tau"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                            else:
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        else:
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])   
                        
                        i+=1
                        tot_arr = np.add(tot_arr, [h[1,0],h[0,0],h[0,1],h[1,1]])

            if plot_signal_components == False:
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal,label=r'$\mathcal{B}(B^0_{(s)}\rightarrow{}$invisible$)=$ '+ f'{latex_BF}', bottom=np.subtract(tot_arr,tot_signal), width=1.0, lw=1.5,edgecolor = plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')              
                
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
            plt.ylabel('Expected counts')

            if pull_type_plot==True:
                # Bottom axis 
                ax_sub = fig.add_subplot(gs[1], sharex=ax_main)
                
                if plot_signal_components == False:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')
                else:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) , facecolor= 'none', hatch='\\\\\\',label=r'$S$ for $\mathcal{B}(B^0_{(s)}\rightarrow{}$invisible$)=$ '+ f'{latex_BF}')

                ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =[-val for val in [B_err[1,0], B_err[0,0], B_err[0,1], B_err[1,1]]], color='black', alpha=0.45, width=1, label='$\sigma_B$')#r'$Z \to q \bar{q}$ background systematic')


                # Clean up sub axis
                ax_sub.set_xticks([x[1,0],x[0,0],x[0,1],x[1,1]])
                ax_sub.tick_params(axis='x', which='major', length=0) 
                ax_sub.set_ylabel('Background-subtracted \n counts')  
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
                plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}_with_second_axis_comp{len(components_to_plot)}_{components_to_plot[-1]}.pdf'), bbox_inches='tight')
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


def make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,
                                       cut_opt_samples=None,  
                                       lrange_interp_N_dict=(0.999,1),
                                       hrange_interp_N_dict=(0.999,1),
                                       nlh = 200, signal_BF=1e-6, 
                                       eventsProcessed_dict = cfg.eventsProcessed, 
                                       histbins=(2,2), 
                                       components_to_plot =  ['hadronic_background','combined_signal'], 
                                       model_all_1prong_leptonic_tau = True,
                                       binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                       plot_signal_components=False,  
                                       nMC_plots_path=None, 
                                       final_plot_path = None, 
                                       pull_type_plot=False, 
                                       lcut=None, 
                                       hcut=None, 
                                       logpath= None):
    
    #Note that all plots only set up for (2x2) binnning - but can still do i bin calculation
    '''
    Function that creates and plots  the "final binning plot" for a given signal BF, all using interpolated maps 
    Inputs to function
    interp_N_dict: dictionary (used to find cut point)
        Dictionary of interpolated distributions for the number of events for a given BDT cut - output of create_N_map.
    lrange_interp_N_dict: tuple, optional
        Range of BDT_lh cut values for 1-P(l) interp_N_dict is produced over
    hrange_interp_N_dict:tuple, optional
        Range of BDT_lh cut values for 1-P(h) interp_N_dict is produced over
    nlh: integer, optional
        Number of bins (in each direction ie. 1-P(l) and 1-P(h)), want to run optimisation to find optimum cut over 
    cut_opt_samples: list, optional
        List of samples want to include in S and B values used to calculate optimal cut. Default:None - ie. use all samples in interp_N_dict keys list
    components_to_plot: list, optiona;
        List of Sample Allocations to use in plot. Note this can be different from those used in optimisation.
    signal_BF:float, optional
        Signal BF want to calculate optimisation of cut for and then also use in plot
    eventsProcessed_dict:dictionary, optional
        Dictionary of number of eventsProcessed by tupling, stored in config. Used to calculate selection efficiencies.
    histbins: tuple, optional
        Binning want to make final bin plot over in BDT space
    binned_x_axis: list,optional
        Names for bins that are flattened to give on xaxis
    plot_signal_components:bool, optional
        Flag as to whether to plot the signal components separeately or as a single combined signal
    nMC_plots_path: str, optional
        Path to save histograms of number of events per bin (in 2D BDT space) - if None, plots not saved
    final_plot_path: str, optional
        Path to save final binning plot. If None, not saved
    pull_type_plot:bool, optional
        Flag as to whether plot with signal-bkg benethe plot
    lcut: float, optional
        Optional cut point for 1-P(l), if None cut optimisation will be run
    hcut: float, optional
        Optional cut point for 1-P(h), if None cut optimisation will be run
    Note: need to specify both lcut and hcut for the input cut point to be used
    logpath:str, optional
        Path to save bin edges
    '''
    #turn BF into title worthy version
    latex_BF = latex_form_exp(signal_BF)
   
    print('--> Finding optimum cut')
    if lcut is not None and hcut is not None:
        l_cut = lcut
        h_cut = hcut
    else:
        # find optimum cut
        FOM, err_FOM, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF, = run_2d_optimisation(interp_N_dict, cut_opt_samples=cut_opt_samples, lrange_plot=lrange_interp_N_dict ,hrange_plot=hrange_interp_N_dict, nlh=nlh , sig_BF=signal_BF, incl_other_syst = False)
        
        indices = np.unravel_index(np.argmax(FOM), np.shape(FOM))
        l_cut = lsearch[indices[0]]
        h_cut = hsearch[indices[1]]

    
    # define samples want from components input 
    samples = flatten_list([cfg.sample_allocations[component] for component in components_to_plot])


    # Number of events per bin in MC from interpolation 

    if histbins != (2,2):
        
        if histbins != 1:
            raise ValueError("Currently only set up for histbins=(2,2) or 1")
        
        elif histbins == 1:
            h_bins_edges = np.linspace(h_cut, 1, histbins + 1) #x axis
            l_bins_edges = np.linspace(l_cut, 1, histbins + 1)

            N_dict_MC = {}

            print('--> Finding number of events per bin')
            #find number of MC events per bin after interpolation
            for sample in interp_N_dict.keys():

                # finite difference
                N_MC = np.maximum(interp_N_dict[sample](l_cut,h_cut,grid=False),0)

                N_dict_MC[sample] =  np.array([N_MC])
                    
    else:   
        #bin edges - h on x axis, l on y axis
        h_bins_edges = np.linspace(h_cut, 1, histbins[0] + 1) #x axis
        l_bins_edges = np.linspace(l_cut, 1, histbins[1] + 1)

        N_dict_MC = {}

        print('--> Finding number of events per bin')
        #find number of MC events per bin after interpolation
        for sample in interp_N_dict.keys():
            # Grid shape: (histbins[0]+1, histbins[1]+1)
            F = np.zeros((histbins[0] + 1, histbins[1] + 1))

            for ih in range(histbins[0] + 1):
                for il in range(histbins[1] + 1):
                    F[ih, il] = interp_N_dict[sample](l_bins_edges[il], h_bins_edges[ih], grid=False)
                # number of events with H > h_cut and L > l_cut

            N_MC = np.zeros((histbins[0], histbins[1]))
            
            for ih in range(histbins[0]):
                for il in range(histbins[1]):

                    #print(F[ih+1, il+1])

                    # finite difference
                    N_MC[ih, il] = np.maximum((
                        F[ih, il]     # bottom-left corner
                        - F[ih+1, il]     # bottom-right
                        - F[ih,   il+1]   # top-left
                        + F[ih+1, il+1]   # top-right (restore overlap)
                        ),0) 
                    #ie. if <0 due to spline dip set to 0

            N_dict_MC[sample] =  N_MC 
                 

            if nMC_plots_path is not None:

                # Create a figure
                plt.figure(figsize=(8,6))

       
                # N_MC should match the "grid" of bin edges, so h_bins_edges on x, l_bins_edges on y
                # pcolormesh expects shape (len(y_edges)-1, len(x_edges)-1), so may need to transpose
                plt.pcolormesh(h_bins_edges, l_bins_edges, N_MC.T, shading='auto', cmap=plt.cm.Blues)

                plt.colorbar(label='Number of MC events')
                plt.xlabel('$1-P(h)$')
                plt.ylabel('$1-P(l)$')
                plt.title(cfg.titles[sample])

                plt.show()
                plt.savefig(os.path.join(set_outputpath(nMC_plots_path),f'NMC_remaining_{sample}_at_BF={signal_BF}_optcut_Ninterp_only.pdf'))
            
            
        if final_plot_path is not None:
            filename = os.path.join(set_outputpath(final_plot_path),'bin_edges_for_BF.txt')
            mode = 'a' if os.path.exists(filename) else 'w'  # append if exists, else write

            with open(filename, mode) as log_file:
                log_file.write(f"BF: {signal_BF}\n")
                log_file.write(f"1-P(h): {h_bins_edges}\n")
                log_file.write(f"1-P(l): {l_bins_edges}\n")
                log_file.write(f"\n")

        elif logpath is not None:
            filename = os.path.join(set_outputpath(logpath),'bin_edges_for_BF.txt')
            mode = 'a' if os.path.exists(filename) else 'w'  # append if exists, else write

            with open(filename, mode) as log_file:
                log_file.write(f"BF: {signal_BF}\n")
                log_file.write(f"1-P(h): {h_bins_edges}\n")
                log_file.write(f"1-P(l): {l_bins_edges}\n")
                log_file.write(f"\n")


    #calculating per bin efficiencies from N MC remaining and convert into per bin S, B and errors (systematics include S and B from efficiency (finite MC size) and BF(Z--> qq) error [based on current measurements - would improve with FCCee])
    efficienies, efficiencies_err, N_dict_MC = post_bdt_eff_finder.get_eff_from_nMC_list(N_dict_MC, eventsProcessed_dict = eventsProcessed_dict)
    per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err  =post_bdt_eff_finder.get_n_expected_components(efficienies, efficiencies_err,signal_bf=signal_BF, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau)

    #check if have additional backgrounds to inclusive ones (ie. compare samples to inclusive backgrounds list in config)
    if len(list(set(samples).intersection(cfg.exclusive_backgrounds)))>0:
        excl_bkgs = list(set(samples).intersection(cfg.exclusive_backgrounds))
    else:
        excl_bkgs = None
    

    S, B, S_err, B_err = post_bdt_eff_finder.get_total_SB(per_sample_n_expect_dict, per_sample_novereff, per_sample_eff_err, per_sample_frac_BFZbb_err, per_sample_frac_fk_err, incl_other_syst=True, exclusive_background_samples = excl_bkgs) # just used in final return efficiencies therefore want to include any other backgrounds here


    if final_plot_path:
        x = binned_x_axis
        if histbins==(2,2):
            plt.figure()
            tot_arr=[0,0,0,0]
            tot_signal = [0,0,0,0]
            tot_B2lnu = [0,0,0,0]

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
        
                if allocation not in components_to_plot:
                    continue
                
                for sample in cfg.sample_allocations[allocation]:


                    if sample in cfg.sample_allocations['combined_signal']: #plot combined signal
                        h = per_sample_n_expect_dict[sample]
                        tot_signal = np.add(tot_signal,[h[1,0],h[0,0],h[0,1],h[1,1]])

                        if plot_signal_components == True:
                            hist_opts = histogram_settings()[allocation]
                            plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[sample+'_invis'], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        tot_arr = np.add(tot_arr,[h[1,0],h[0,0],h[0,1],h[1,1]])
                        i+=1

                    elif allocation =='B2lnu_background_combined': #plot combined B2lnu backgrounds
                        h = per_sample_n_expect_dict[sample]
                        tot_B2lnu = np.add(tot_B2lnu,[h[1,0],h[0,0],h[0,1],h[1,1]])
                        
                        if i == (len(cfg.sample_allocations[allocation])-1):
                            hist_opts = histogram_settings()[allocation]
                            plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_B2lnu,label=cfg.titles[allocation], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'], facecolor= hist_opts['facecolor'], hatch=hist_opts['hatch'])
                            tot_arr = np.add(tot_arr,tot_B2lnu)
                        i+=1

                    else:
                        h = per_sample_n_expect_dict[sample]
                        hist_opts = histogram_settings()[allocation]

                        if model_all_1prong_leptonic_tau==True:
                            if sample in ("p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu","p8_ee_Zbb_ecm91_EvtGen_Bc2TauNuTau2MuNuNu"):
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}_1prong_leptonic_tau"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                            else:
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])
                        else:
                                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[h[1,0],h[0,0],h[0,1],h[1,1]],label=cfg.titles[f"{sample}"], bottom=tot_arr, width=1.0, lw=1.5,edgecolor =hist_opts['edgecolor'][i] , facecolor= hist_opts['facecolor'][i], hatch=hist_opts['hatch'][i])   

                        i+=1
                        tot_arr = np.add(tot_arr, [h[1,0],h[0,0],h[0,1],h[1,1]])

            if plot_signal_components == False:
                plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal,label=r'$\mathcal{B}(B^0_{(s)}\rightarrow{}$invisible$)=$ '+ f'{latex_BF}', bottom=np.subtract(tot_arr,tot_signal), width=1.0, lw=1.5,edgecolor = plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')              
                
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
            plt.ylabel('Expected counts')

            if pull_type_plot==True:
                # Bottom axis 
                ax_sub = fig.add_subplot(gs[1], sharex=ax_main)
                
                if plot_signal_components == False:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] )  , facecolor= 'none', hatch='\\\\\\')
                else:
                    ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],tot_signal, bottom=0, width=1.0, lw=1.5,edgecolor =plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) , facecolor= 'none', hatch='\\\\\\',label=r'$S$ for $\mathcal{B}(B^0_{(s)}\rightarrow{}$invisible$)=$ '+ f'{latex_BF}')

                ax_sub.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =[-val for val in [B_err[1,0], B_err[0,0], B_err[0,1], B_err[1,1]]], color='black', alpha=0.45, width=1, label='$\sigma_B$')#r'$Z \to q \bar{q}$ background systematic')


                # Clean up sub axis
                ax_sub.set_xticks([x[1,0],x[0,0],x[0,1],x[1,1]])
                ax_sub.tick_params(axis='x', which='major', length=0) 
                ax_sub.set_ylabel('Background-subtracted \n counts')  
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
                plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}_with_second_axis_comp{len(components_to_plot)}_{components_to_plot[-1]}.pdf'), bbox_inches='tight')
            else: 
                
                #add systematic error to B - error bar
                #plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', color='black', alpha=0.4, width=1)
                #plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', facecolor='none',  width=1, edgecolor='black')            
                #plt.bar([x[1,0],x[0,0],x[0,1],x[1,1]],[2*i for i in [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]], bottom =np.subtract([B[1,0],B[0,0],B[0,1],B[1,1]], [B_err[1,0],B_err[0,0],B_err[0,1],B_err[1,1]]), label=r'$Z \to q \bar{q}$ background systematic', facecolor='none',  width=1, edgecolor='none', hatch='///')            
                

                plt.legend()
                plt.savefig(os.path.join(set_outputpath(final_plot_path),f'final_binning_plot_BF={signal_BF}.pdf'))
                
        else:
            raise ValueError('Currently only set up to plot 2x2 binning, ie. histbins==(2,2)')

    return per_sample_n_expect_dict, S, B, S_err, B_err, signal_BF,l_cut,h_cut



def likelihood_model_builder_extra_bkgs(df, interp_N_dict, cut_opt_samples=None, componenets_in_FOM=['hadronic_background','combined_signal'], model_all_1prong_leptonic_tau = True, signal_BF=1e-6,
                             lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh=200,bins = (2,2),
                             ntoys = 250,
                             fit_plotpath=None, x_values = np.array([['A','B'],['C','D']]), spread_plotpath=None, logpath=None, lcut=None, hcut=None):

    """ 
    likelihood_model_builder(**opts ) will return optimum point from minimising signal error on fit to toys

    """

    _, S, B, S_err, B_err, signal_BF,opt_l_cut,opt_h_cut = make_final_binning_plot_extra_bkgs(df, interp_N_dict,cut_opt_samples=cut_opt_samples, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=bins, components_to_plot = componenets_in_FOM ,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, binned_x_axis = x_values,nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=lcut, hcut=hcut)
    _, onebin_S, onebin_B, onebin_S_err, onebin_B_err,_,_,_ = make_final_binning_plot_extra_bkgs(df, interp_N_dict,cut_opt_samples=cut_opt_samples, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=1, components_to_plot =  componenets_in_FOM, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, binned_x_axis = x_values, nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=opt_l_cut, hcut=opt_h_cut)
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

        # Set error definition to ensure minuit is expecting NLL not 2*NLL (since it is 2NLL which is chi2 dist)
        mi.errordef = Minuit.LIKELIHOOD

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

                #define labels
                x_strings =[x_values[1,0],x_values[0,0],x_values[0,1],x_values[1,1]] 
                x = np.arange(len(x_strings))

                #define fit quantities
                sig_fit = np.array([sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]])
                bkg_fit = np.array([sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]])
                total_fit = sig_fit + bkg_fit
                toy_data_np = np.array([toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]])
                residuals =  (toy_data_np-total_fit)/np.sqrt(toy_data_np)
                

                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

                # --- Main plot (stacked bars) ---
                ax0 = fig.add_subplot(gs[0])
                ''' MAYBE CHANGE SO THAT USE bkg_fit etc more'''
                ax0.bar(x,bkg_fit,label=r'Fitted $B$', width=1.0, edgecolor='red', facecolor='none',hatch='///')
                ax0.bar(x,sig_fit,label=r'Fitted $S$', bottom=[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], width=1.0, edgecolor=plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) ,hatch='\\\\\\', facecolor='none')
                # Error on bkg expectation used as gaussain constraint (ie. fractionalB error * B expected)
                ax0.bar(x,[2*overall_background_error*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], bottom =np.subtract(np.add([sc_b *i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], [sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]]),[overall_background_error*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]), label=r'$\sigma_B$', color='black', alpha=0.4, width=1)
                # error bar just contains stat error on toy data
                ax0.errorbar(x, toy_data_np,yerr=[np.sqrt(i) for i in [toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]]],xerr=0.5, fmt='.',label='Pseudoexperiment data', color='k')

                ax0.set_ylabel('Counts')
                ax0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax0.set_xticks(x)
                ax0.set_xticklabels(x_strings)
                ax0.tick_params(axis='x', which='major', length=0)
                ax0.legend()

                # Edge ticks
                bars = ax0.patches
                edges = [b.get_x() for b in bars] + [b.get_x() + b.get_width() for b in bars]
                ax0.set_xticks(edges, minor=True)
                ax0.tick_params(axis='x', which='minor', length=4)

                # --- Residual plot ---
                
                #BKG SUBTRACTED SIGNAL
                ax1 = fig.add_subplot(gs[1], sharex=ax0)

                ax1.errorbar(x, toy_data_np - bkg_fit,yerr=np.sqrt(toy_data_np),xerr=0.5, fmt='.',color='k')
                ax1.set_ylabel('Background-subtracted \n counts') 
                ax1.bar(x,sig_fit,label=r'Fit $S$', width=1.0, edgecolor=plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) ,hatch='\\\\\\', facecolor='none')
                #ax1.bar(x,[2*overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]],bottom=[-overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], label=r'$\sigma_B$', color='black', alpha=0.4, width=1)
                ax1.set_xticks(x)
                ax1.set_xticklabels(x_strings)
                ax1.tick_params(axis='x', which='minor', length=4)

                #ensure remove middle ticks
                ax1.tick_params(axis='x', which='major', length=0)

                #determine sub_height
                #berr = [overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]
                #ax1.set_ylim(-1.3*berr[3],max((sig_fit[3] + 0.3*berr[3]),1.3*berr[3]))
                ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

                # Remove x tick labels on top plot
                plt.setp(ax0.get_xticklabels(), visible=False)

                # --- Save ---
                plt.tight_layout()
                plt.savefig(os.path.join(set_outputpath(os.path.join(fit_plotpath, 'toy_fits')), f'toy_fit_for_first_toy_BF{signal_BF}_with_bkg_sub.pdf'))
             
    
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


# Note the ONLY changes compated to likelihood_model_builder_extra_bkgs is that make_final_binning_plot_extra_bkgs is replaced by make_final_binning_plot_extra_bkgs_nodata
def likelihood_model_builder_extra_bkgs_nodata(interp_N_dict, cut_opt_samples=None, componenets_in_FOM=['hadronic_background','combined_signal'], model_all_1prong_leptonic_tau = True, signal_BF=1e-6,
                             lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh=200,bins = (2,2),
                             ntoys = 250,
                             fit_plotpath=None, x_values = np.array([['A','B'],['C','D']]), spread_plotpath=None, logpath=None, lcut=None, hcut=None):

    """ 
    likelihood_model_builder(**opts ) will return optimum point from minimising signal error on fit to toys

    """


    _, S, B, S_err, B_err, signal_BF,opt_l_cut,opt_h_cut = make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=cut_opt_samples, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=bins, components_to_plot = componenets_in_FOM ,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, binned_x_axis = x_values,nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=lcut, hcut=hcut)
    _, onebin_S, onebin_B, onebin_S_err, onebin_B_err,_,_,_ = make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=cut_opt_samples, lrange_interp_N_dict=lrange_interp_N_dict ,hrange_interp_N_dict=hrange_interp_N_dict,signal_BF=signal_BF, nlh=nlh,eventsProcessed_dict = cfg.eventsProcessed , histbins=1, components_to_plot =  componenets_in_FOM,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, binned_x_axis = x_values, nMC_plots_path=None, final_plot_path = None, logpath = logpath, lcut=opt_l_cut, hcut=opt_h_cut)
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

        # Set error definition to ensure minuit is expecting NLL not 2*NLL (since it is 2NLL which is chi2 dist)
        mi.errordef = Minuit.LIKELIHOOD

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

                #define labels
                x_strings =[x_values[1,0],x_values[0,0],x_values[0,1],x_values[1,1]] 
                x = np.arange(len(x_strings))

                #define fit quantities
                sig_fit = np.array([sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]])
                bkg_fit = np.array([sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]])
                total_fit = sig_fit + bkg_fit
                toy_data_np = np.array([toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]])
                residuals =  (toy_data_np-total_fit)/np.sqrt(toy_data_np)
                

                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

                # --- Main plot (stacked bars) ---
                ax0 = fig.add_subplot(gs[0])
                ''' MAYBE CHANGE SO THAT USE bkg_fit etc more'''
                ax0.bar(x,bkg_fit,label=r'Fitted $B$', width=1.0, edgecolor='red', facecolor='none',hatch='///')
                ax0.bar(x,sig_fit,label=r'Fitted $S$', bottom=[sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], width=1.0, edgecolor=plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) ,hatch='\\\\\\', facecolor='none')
                # Error on bkg expectation used as gaussain constraint (ie. fractionalB error * B expected)
                ax0.bar(x,[2*overall_background_error*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], bottom =np.subtract(np.add([sc_b *i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], [sc_s *i for i in [S[1,0],S[0,0],S[0,1],S[1,1]]]),[overall_background_error*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]), label=r'$\sigma_B$', color='black', alpha=0.4, width=1)
                # error bar just contains stat error on toy data
                ax0.errorbar(x, toy_data_np,yerr=[np.sqrt(i) for i in [toy_data[1,0],toy_data[0,0],toy_data[0,1],toy_data[1,1]]],xerr=0.5, fmt='.',label='Pseudoexperiment data', color='k')

                ax0.set_ylabel('Counts')
                ax0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax0.set_xticks(x)
                ax0.set_xticklabels(x_strings)
                ax0.tick_params(axis='x', which='major', length=0)
                ax0.legend()

                # Edge ticks
                bars = ax0.patches
                edges = [b.get_x() for b in bars] + [b.get_x() + b.get_width() for b in bars]
                ax0.set_xticks(edges, minor=True)
                ax0.tick_params(axis='x', which='minor', length=4)

                # --- Residual plot ---
                
                #BKG SUBTRACTED SIGNAL
                ax1 = fig.add_subplot(gs[1], sharex=ax0)

                ax1.errorbar(x, toy_data_np - bkg_fit,yerr=np.sqrt(toy_data_np),xerr=0.5, fmt='.',color='k')
                ax1.set_ylabel('Background-subtracted \n counts') 
                ax1.bar(x,sig_fit,label=r'Fit $S$', width=1.0, edgecolor=plt.cm.Blues( np.linspace(0, 1, 12)[-4] ) ,hatch='\\\\\\', facecolor='none')
                #ax1.bar(x,[2*overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]],bottom=[-overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]], label=r'$\sigma_B$', color='black', alpha=0.4, width=1)
                ax1.set_xticks(x)
                ax1.set_xticklabels(x_strings)
                ax1.tick_params(axis='x', which='minor', length=4)

                #ensure remove middle ticks
                ax1.tick_params(axis='x', which='major', length=0)

                #determine sub_height
                #berr = [overall_background_error*sc_b*i for i in [B[1,0],B[0,0],B[0,1],B[1,1]]]
                #ax1.set_ylim(-1.3*berr[3],max((sig_fit[3] + 0.3*berr[3]),1.3*berr[3]))
                ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

                # Remove x tick labels on top plot
                plt.setp(ax0.get_xticklabels(), visible=False)

                # --- Save ---
                plt.tight_layout()
                plt.savefig(os.path.join(set_outputpath(os.path.join(fit_plotpath, 'toy_fits')), f'toy_fit_for_first_toy_BF{signal_BF}_with_bkg_sub.pdf'))
             
    
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

def calculate_BF_sensitivities_extra_bkgs(interp_N_dict, cut_opt_samples=None, componenets_in_FOM=None,model_all_1prong_leptonic_tau=True,  lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=False, full_df=None, ntoys=200,plot=True,saveplotpath = 'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999',toyplotpath = None, dict_path = 'outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999'):
    
    #create dictionaries to store results
    max_FOM_arr = np.zeros(len(sig_BFs))
    max_FOM_err_arr = np.zeros(len(sig_BFs))
    BFs_arr=np.zeros(len(sig_BFs))
    light_cut_arr=np.zeros(len(sig_BFs))
    heavy_cut_arr=np.zeros(len(sig_BFs))
    S_exp_arr = np.zeros(len(sig_BFs))
    B_exp_arr = np.zeros(len(sig_BFs))
    S_err_arr = np.zeros(len(sig_BFs))
    B_err_arr = np.zeros(len(sig_BFs))


    i=0
    toy_BFs=[]
    toy_significance=[]
    toy_sig_spread = []

    #loop through BFs and calculate significance from toys
    for BF in sig_BFs:
        
        #Run optimisation using inclusive backgrounds and signal to find optimum point for BDT cut
        FOM_arr, err_FOM_arr, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF   = run_2d_optimisation(interp_N_dict, cut_opt_samples=cut_opt_samples, lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF, incl_other_syst=True)
        
        #define samples in FOM from input components
        if componenets_in_FOM is not None:
            samples_in_FOM = flatten_list([cfg.sample_allocations[component] for component in componenets_in_FOM])
        else:
            samples_in_FOM = None 

        FOM, err_FOM, S, B, S_err, B_err, sig_BF, l_optcut, h_optcut = calc_SB_from_opt_cut(interp_N_dict, FOM_arr, lsearch, hsearch, sig_BF, SB_samples=samples_in_FOM, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau,incl_other_syst = True)
        

        max_FOM_arr[i] = FOM
        max_FOM_err_arr[i] = err_FOM
        light_cut_arr[i] = l_optcut
        heavy_cut_arr[i] = h_optcut 
        BFs_arr[i] = sig_BF
        S_exp_arr[i] = S
        B_exp_arr[i] = B
        S_err_arr[i] = S_err
        B_err_arr[i] = B_err


        if incl_toys_fit == True:
            if i%2==0:
                if full_df is None:
                    print('Warning: no data provided for toys fit')
                    continue
                else:
                    #only produce plots for toys every 20th BF
                    if i % 20 == 0:
                        av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder_extra_bkgs(full_df,  interp_N_dict, cut_opt_samples =cut_opt_samples,  componenets_in_FOM=componenets_in_FOM, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=toyplotpath, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=toyplotpath)
                    else:
                        av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder_extra_bkgs(full_df, interp_N_dict,cut_opt_samples =cut_opt_samples,   componenets_in_FOM=componenets_in_FOM,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=None, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=None)
                    
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
    five_sigma_BF = np.interp(5, max_FOM_arr, BFs_arr)
    three_sigma_BF = np.interp(3, max_FOM_arr, BFs_arr)
    print(f"5 sigma BFs = {five_sigma_BF}" )
    print(f"3 sigma BFs = {three_sigma_BF}" )


    #calulating S/sqrt(S+B+varS+varB)
    #error includes whatever specified in 2d optimisation 

    significance_incl_error = S_exp_arr/np.sqrt(S_exp_arr+B_exp_arr+S_err_arr**2+B_err_arr**2)
    five_sigma_BF_inclerr = np.interp(5, significance_incl_error, BFs_arr)
    three_sigma_BF_inclerr = np.interp(3, significance_incl_error, BFs_arr)
    print(f"5 sigma BFs incl error = {five_sigma_BF_inclerr}" )
    print(f"3 sigma BFs incl error= {three_sigma_BF_inclerr}" )


    CL = sigma_to_percentage(max_FOM_arr)


    naive_dict= {'BFs': BFs_arr,
                'significance': max_FOM_arr, 
                'error':max_FOM_err_arr}
    
    incl_syst_dict= {'BFs': BFs_arr,
                'significance': significance_incl_error,
                'incl_other_syst':incl_other_syst}
    
    BDT_cuts_dict= {'BFs':BFs_arr,
                    'light':light_cut_arr,
                    'heavy':heavy_cut_arr}
    
    if incl_toys_fit == True:
        toys_dict= {'BFs': toys_BFs,
                    'significance': toys_significance, 
                    'error':toys_sig_spread,
                    'ntoys':ntoys}
        with open(os.path.join(set_outputpath(dict_path),'toys_sensitivity_dict.pkl'), 'wb') as fp:
            pickle.dump(toys_dict, fp)
    else:
        toys_dict= None
    

    #save dictionaries
    with open(os.path.join(set_outputpath(dict_path),'naive_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(naive_dict, fp)
 
    with open(os.path.join(set_outputpath(dict_path),'incl_syst_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(incl_syst_dict, fp)

    with open(os.path.join(set_outputpath(dict_path),'optimal_bdt_cuts_dict.pkl'), 'wb') as fp:
        pickle.dump(BDT_cuts_dict, fp)


    if plot == True: 
        sensitivity_CL_plotter(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=saveplotpath)

    return max_FOM_arr, light_cut_arr, heavy_cut_arr, BFs_arr, CL


# Again only changes here are replacing likelihood_model_builder_extra_bkgs with likelihood_model_builder_extra_bkgs_nodata and removing associated df requreiements
def calculate_BF_sensitivities_extra_bkgs_nodata(interp_N_dict, cut_opt_samples=None, componenets_in_FOM=None, model_all_1prong_leptonic_tau = True,  lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=False, ntoys=200,plot=True,saveplotpath = 'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999',toyplotpath = None, dict_path = 'outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999'):
    
    #create dictionaries to store results
    max_FOM_arr = np.zeros(len(sig_BFs))
    max_FOM_err_arr = np.zeros(len(sig_BFs))
    BFs_arr=np.zeros(len(sig_BFs))
    light_cut_arr=np.zeros(len(sig_BFs))
    heavy_cut_arr=np.zeros(len(sig_BFs))
    S_exp_arr = np.zeros(len(sig_BFs))
    B_exp_arr = np.zeros(len(sig_BFs))
    S_err_arr = np.zeros(len(sig_BFs))
    B_err_arr = np.zeros(len(sig_BFs))


    i=0
    toy_BFs=[]
    toy_significance=[]
    toy_sig_spread = []

    #loop through BFs and calculate significance from toys
    for BF in sig_BFs:
        
        #Run optimisation using inclusive backgrounds and signal to find optimum point for BDT cut
        FOM_arr, err_FOM_arr, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF   = run_2d_optimisation(interp_N_dict, cut_opt_samples=cut_opt_samples, lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF, incl_other_syst=True)
        
        #define samples in FOM from input components
        if componenets_in_FOM is not None:
            samples_in_FOM = flatten_list([cfg.sample_allocations[component] for component in componenets_in_FOM])
        else:
            samples_in_FOM = None 

        FOM, err_FOM, S, B, S_err, B_err, sig_BF, l_optcut, h_optcut = calc_SB_from_opt_cut(interp_N_dict, FOM_arr, lsearch, hsearch, sig_BF, SB_samples=samples_in_FOM,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau,incl_other_syst = True)
        

        max_FOM_arr[i] = FOM
        max_FOM_err_arr[i] = err_FOM
        light_cut_arr[i] = l_optcut
        heavy_cut_arr[i] = h_optcut 
        BFs_arr[i] = sig_BF
        S_exp_arr[i] = S
        B_exp_arr[i] = B
        S_err_arr[i] = S_err
        B_err_arr[i] = B_err


        if incl_toys_fit == True:

            #only produce plots for toys every 20th BF
            if i % 20 == 0:
                av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder_extra_bkgs_nodata(interp_N_dict, cut_opt_samples =cut_opt_samples,  componenets_in_FOM=componenets_in_FOM, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=toyplotpath, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=toyplotpath)
            else:
                av_significance_for_bf, stdev_significance_for_bf =likelihood_model_builder_extra_bkgs_nodata(interp_N_dict,cut_opt_samples =cut_opt_samples,   componenets_in_FOM=componenets_in_FOM, model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau, signal_BF=BF, lrange_interp_N_dict=lrange_plot ,hrange_interp_N_dict=hrange_plot,nlh=nlh,bins = (2,2), ntoys = ntoys,fit_plotpath=None, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=None)
            
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
    five_sigma_BF = np.interp(5, max_FOM_arr, BFs_arr)
    three_sigma_BF = np.interp(3, max_FOM_arr, BFs_arr)
    print(f"5 sigma BFs = {five_sigma_BF}" )
    print(f"3 sigma BFs = {three_sigma_BF}" )


    #calulating S/sqrt(S+B+varS+varB)
    #error includes whatever specified in 2d optimisation 

    significance_incl_error = S_exp_arr/np.sqrt(S_exp_arr+B_exp_arr+S_err_arr**2+B_err_arr**2)
    five_sigma_BF_inclerr = np.interp(5, significance_incl_error, BFs_arr)
    three_sigma_BF_inclerr = np.interp(3, significance_incl_error, BFs_arr)
    print(f"5 sigma BFs incl error = {five_sigma_BF_inclerr}" )
    print(f"3 sigma BFs incl error= {three_sigma_BF_inclerr}" )


    CL = sigma_to_percentage(max_FOM_arr)


    naive_dict= {'BFs': BFs_arr,
                'significance': max_FOM_arr, 
                'error':max_FOM_err_arr}
    
    incl_syst_dict= {'BFs': BFs_arr,
                'significance': significance_incl_error,
                'incl_other_syst':incl_other_syst}
    
    BDT_cuts_dict= {'BFs':BFs_arr,
                    'light':light_cut_arr,
                    'heavy':heavy_cut_arr}
    
    if incl_toys_fit == True:
        toys_dict= {'BFs': toys_BFs,
                    'significance': toys_significance, 
                    'error':toys_sig_spread,
                    'ntoys':ntoys}
        with open(os.path.join(set_outputpath(dict_path),'toys_sensitivity_dict.pkl'), 'wb') as fp:
            pickle.dump(toys_dict, fp)
    else:
        toys_dict= None
    

    #save dictionaries
    with open(os.path.join(set_outputpath(dict_path),'naive_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(naive_dict, fp)
 
    with open(os.path.join(set_outputpath(dict_path),'incl_syst_sensitivity_dict.pkl'), 'wb') as fp:
        pickle.dump(incl_syst_dict, fp)

    with open(os.path.join(set_outputpath(dict_path),'optimal_bdt_cuts_dict.pkl'), 'wb') as fp:
        pickle.dump(BDT_cuts_dict, fp)


    if plot == True: 
        sensitivity_CL_plotter(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=saveplotpath)

    return max_FOM_arr, light_cut_arr, heavy_cut_arr, BFs_arr, CL


def calc_BF_sig_stderrs_nodata(interp_N_dict, cut_opt_samples=None, componenets_in_FOM=None, model_all_1prong_leptonic_tau=True, lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),plot=True,saveplotpath = 'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999',dict_path = 'outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0999', plot_dotted_line_list =  ["noBsyst","5percent_Bsyst" ,"10percent_Bsyst"]):
    
    #create dictionaries to store results
    max_FOM_arr = np.zeros(len(sig_BFs))
    max_FOM_err_arr = np.zeros(len(sig_BFs))
    BFs_arr=np.zeros(len(sig_BFs))
    light_cut_arr=np.zeros(len(sig_BFs))
    heavy_cut_arr=np.zeros(len(sig_BFs))
    S_exp_arr = np.zeros(len(sig_BFs))
    B_exp_arr = np.zeros(len(sig_BFs))
    S_err_arr = np.zeros(len(sig_BFs))
    B_err_arr = np.zeros(len(sig_BFs))


    i=0

    #loop through BFs and calculate significance from toys
    for BF in sig_BFs:
        
        #Run optimisation using inclusive backgrounds and signal to find optimum point for BDT cut
        FOM_arr, err_FOM_arr, S_arr, B_arr, S_error_arr, B_error_arr, lsearch, hsearch, sig_BF   = run_2d_optimisation(interp_N_dict, cut_opt_samples=cut_opt_samples, lrange_plot=lrange_plot ,hrange_plot=hrange_plot, nlh=nlh , sig_BF=BF, incl_other_syst=True)
        
        #define samples in FOM from input components
        if componenets_in_FOM is not None:
            samples_in_FOM = flatten_list([cfg.sample_allocations[component] for component in componenets_in_FOM])
        else:
            samples_in_FOM = None 

        FOM, err_FOM, S, B, S_err, B_err, sig_BF, l_optcut, h_optcut = calc_SB_from_opt_cut(interp_N_dict, FOM_arr, lsearch, hsearch, sig_BF, SB_samples=samples_in_FOM,model_all_1prong_leptonic_tau = model_all_1prong_leptonic_tau,incl_other_syst = True)
        
        light_cut_arr[i] = l_optcut
        heavy_cut_arr[i] = h_optcut 
        BFs_arr[i] = sig_BF
        S_exp_arr[i] = S
        B_exp_arr[i] = B
        S_err_arr[i] = S_err
        B_err_arr[i] = B_err

        i+=1


    dict_to_save= {'BFs': BFs_arr,
            'S': S_exp_arr, 
            'B':B_exp_arr,
            'S_err':S_err_arr,
            'B_err':B_err_arr,
            'l_cut':light_cut_arr,
            'h_cut':heavy_cut_arr,
            'cut_opt_samples':cut_opt_samples,
            'componenets_in_FOM':componenets_in_FOM
            }
    print("saving dictionary of S and B")
    #save dictionary
    with open(os.path.join(set_outputpath(dict_path),'SB_dict.pkl'), 'wb') as fp:
        pickle.dump(dict_to_save, fp)

    print("plotting")
    if plot==True:
        plot_BF_sig_stderrs(S_exp_arr,B_exp_arr,BFs_arr,saveplotpath, plot_dotted_line_list=plot_dotted_line_list)
 



def plot_BF_sig_stderrs(S,B,sig_BFs,savepath,plot_dotted_line_list= ["noBsyst","2percent_Bsyst" ,"10percent_Bsyst"]):

    S_arr = np.array(S)
    B_arr = np.array(B)
    BFs_arr = np.array(sig_BFs)
    max_FOM_arr = S_arr/np.sqrt(S_arr+B_arr)

    colors = {} 
    significances = {}
    perc_syst = {}

    significances["noBsyst"] = max_FOM_arr
    significances["1percent_Bsyst"] = S_arr/np.sqrt(S_arr+B_arr+(0.01*B_arr)**2)
    significances["2percent_Bsyst"]  = S_arr/np.sqrt(S_arr+B_arr+(0.02*B_arr)**2)
    significances["5percent_Bsyst"]  = S_arr/np.sqrt(S_arr+B_arr+(0.05*B_arr)**2)
    significances["10percent_Bsyst"]  = S_arr/np.sqrt(S_arr+B_arr+(0.1*B_arr)**2)
    #significances["20percent_Bsyst"]  = S_arr/np.sqrt(S_arr+B_arr+(0.2*B_arr)**2)

    perc_syst["noBsyst"] = 0
    perc_syst["1percent_Bsyst"] = 1
    perc_syst["2percent_Bsyst"] = 2
    perc_syst["5percent_Bsyst"] = 5
    perc_syst["10percent_Bsyst"] = 10
    #perc_syst["20percent_Bsyst"] = 20

    colors["noBsyst"] = "tab:blue"
    colors["1percent_Bsyst"] = "tab:green"
    colors["2percent_Bsyst"] = "hotpink"
    colors["5percent_Bsyst"] = "tab:purple"
    colors["10percent_Bsyst"] = "tab:orange"
    #"tab:cyan"
    
    ##########################
    #printing some #s to check!
    ##########################

    #find 3 sigma and 5 sigma points
    five_sigma_BF = {}
    three_sigma_BF = {}
    significances_to_plot = {}
    CL = {}
    CL_90 = {}
    CL_95 = {}

    x_label = r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisible$)$'
    
    plt.figure()

    for key in significances.keys():

        five_sigma_BF[key] = np.interp(5, significances[key], BFs_arr)
        three_sigma_BF[key] = np.interp(3, significances[key], BFs_arr)
        print(f"5 sigma BFs {key} = {five_sigma_BF[key]}" )
        print(f"3 sigma BFs {key} = {three_sigma_BF[key]}" )
        

        #plotting S/sqrt(S+B) line

        #since significance plot only up to 6sigma, replace anything from BF with > 6 with 8 so that dont get dip from cuts changing coming back in
        if key == "10percent_Bsyst":
            mask = [1 if i<=5e-6 else 0 for i in BFs_arr]
            significances_to_plot[key] = np.array([significances[key][i] if mask[i]==1 else 8 for i in range(len(BFs_arr))])
        else:
            significances_to_plot[key] = significances[key]

        plt.plot(BFs_arr,significances_to_plot[key], label =  r'$\sigma_B/B=$'+f'$ {perc_syst[key]}\%$', color= colors[key])
        
        if key in plot_dotted_line_list:    
            plt.vlines(x=three_sigma_BF[key], ymin=0, ymax=3, linestyle='--', color= colors[key])
            plt.vlines(x=five_sigma_BF[key], ymin=0, ymax=5, linestyle='--', color= colors[key])

    plt.hlines(y = 3,xmin = min(BFs_arr),xmax = three_sigma_BF["10percent_Bsyst"], linestyle='--',color='k',alpha = 0.3)
    plt.hlines(y = 5,xmin = 1e-8 ,xmax = five_sigma_BF["10percent_Bsyst"], linestyle='--',color='k',alpha = 0.3)
    #nb xmin changed to make space for legend min(BFs_arr)
    plt.xlabel(x_label)
    plt.ylabel(r'Significance')
    plt.xscale('log')
    plt.ylim(0,6) 
    plt.xlim(1e-9,2e-5) #cut optimisation not really valid beyond this due to spline range
    plt.legend(fontsize = 11)
    plt.savefig(os.path.join(set_outputpath(savepath),f'FOMvsBF_StandardSysts.pdf'))



    ##################################
    # Now plotting CL 
    ##################################
        #convert sigma to CL - for now one sided
    
   
    plt.figure()

    for key in significances.keys():
        CL[key] = sigma_to_percentage(significances[key])
        CL_90[key] = np.interp(90, CL[key], BFs_arr)
        CL_95[key] = np.interp(95, CL[key], BFs_arr)
        print(f"90%CL BFs {key} = {CL_90[key]}" )
        print(f"95%CL BFs {key} = {CL_95[key]}" )
    
        plt.plot(BFs_arr,CL[key],label =  r'$\sigma_B/B=$'+f' ${perc_syst[key]}\%$', color = colors[key])
    
        if key in plot_dotted_line_list:    
            plt.vlines(x=CL_90[key], ymin=0, ymax=90, linestyle='--',  color =colors[key])
            plt.vlines(x=CL_95[key], ymin=0, ymax=95, linestyle='--', color =colors[key])
    
    plt.hlines(y = 90,xmin = min(BFs_arr),xmax = CL_90["10percent_Bsyst"], linestyle='--',color='k',alpha = 0.3)
    plt.hlines(y = 95,xmin = min(BFs_arr),xmax = CL_95["10percent_Bsyst"], linestyle='--',color='k',alpha = 0.3)

    plt.xlabel(x_label)
    plt.ylabel(r'Rejection CL')
    plt.xscale('log')
    plt.ylim(70,102)
    plt.xlim(1e-9,2e-5) 

    plt.legend(fontsize = 11)
    plt.savefig(os.path.join(set_outputpath(savepath),f'CLvsBF_StandardSysts.pdf'))











