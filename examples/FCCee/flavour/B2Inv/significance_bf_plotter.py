# efficiency_map.py
# Creates a 2D map of BDT1 and BDT2 efficiencies for each sample,
#     as well as a 1D map of BDTComb efficiency for each sample

import os

import pandas as pd
from time import time
from datetime import timedelta

from glob import glob
import numpy as np
import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak  # Needed if using awkward arrays
plt.style.use('fcc.mplstyle')
import efficiency_finder
import config as cfg
from efficiency_finder import get_efficiencies
from scipy.stats import poisson, norm
from iminuit import Minuit
import re
import sys
from scipy import interpolate
import math

    
##############################
## Plotting 
##############################


def plot(graph,
        multiple_types_significance,
        signal_bf=np.logspace(-7,-4, num=50),
        errorbar=False,
        errorband=False,
        cuts='wp1',
        interactive=True, 
        save=None, 
        xtitle=None,
        title=None, 
        logx=True,
        logy=False,
        ylim=None,
        components=["signal", "background"],
        verbose=False,
        nbins=5,
        ntoys=250,
        toyinterpolation=False):

    """ 
    plot( BDTcuts, **opts ) will plot a variable

    Parameters
    ----------
    graph: str
        'significance'  or 'significance_incl_error' or 's/full_berr' or 'toy_fit' (if multiple_types_significance==True). If multiple_types_significance==True, then multiple types can be quoted as a list.
    multiple_types_significance: bool
        Whether graph is including multiple types of significance figure (in which case it'll probably be only one cut). If True, graph must be a list, otherwise graph must be  str.
    signal_bf : float, optional
        The assumed signal branching fraction to be plotted against. Default = np.logspace(1e-7,1e-4)
    errorbar: bool, optional
        Plot error bars from total error in significance due to S and B (assuming independent). Default=False. Defaults = False
    errorband: bool, optional
        Plot error band from total error in significance due to S and B (assuming independent)
    cuts : list, optional
        List containing strings. These must be'wp1' or cut branch varname according to a (valid) UPROOT expression.
        Which cuts to use on data. List enables multiple lines with different cuts to be plotted. Default: wp1
    interactive : bool, optional
        Show the plot interactively after its made. Default: True
    save : str, optional
        Save file for the plot. If None then no plot is saved. Default: None 
    xtitle : str, optional
        Provide a custom title for the x axis. Default : `Siganl BF`
    title : str, optional
        Provide a custom title. Default : `BDTcuts`
        Use this if LaTeX complains about the cut expression.
    logx : bool, optional
        Use log scale for the x axis. Default: True
    logy : bool, optional
        Use log scale for the y axis. Default: False
    ylim: : tuple or list, optional
        The lower and upper limits to use in the plot. 
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['signal', 'background']
    verbose : bool, optional
        Print out some useful stuff. Default: False
    nbins: int, optional
        Specifies number of bins for toy fit and significance calculation. Default=5
    ntoys: int, optional
        Specifies number of toys for toy fit. Default=250
    toyinterpolation: bool, optional
        Specifies whether to use cubic interpolation on toy data plot. Default=False
    
    """
    #check 'graph' is a list if 'multiple_types_significance' is True
    if multiple_types_significance:
        if type(graph) != list:
            sys.exit('Error: for multiple_types_significance=True, graph must be a list')


    xmin = min(signal_bf) 
    xmax = max(signal_bf)

    if ylim:
        ymin=ylim[0]
        ymax=ylim[1]


    fig, ax = plt.subplots()

    ##############################
    ## List of cut expressions to pass to uproot and get efficiencies, expected number of events and errors
    ##############################

    if type(cuts)==str:
        number_of_lines=1
        cuts=[cuts]
    else:
        number_of_lines=len(cuts)
    
    for N in np.arange(0,number_of_lines,1):

        if cuts[N] =='wp1':   
            #cuts from config at current working point
            cutforms = [f'{key}>={value}' for key, value in cfg.wp1_cuts.items()]
            #putting cuts in correct format
            eff_cut=str(f'({cutforms[0]})')
            for j in np.arange(1,len(cutforms),1):
                eff_cut += str(f' & ({cutforms[j]})')
            label="wp1: (EVT_MVA1$\geq$0.994), (EVT_MVA2$\geq$0.95)"
        else:
            eff_cut = cuts[N]
            label1=cuts[N].replace(">=","$\geq$")
            label=label1.replace("&",",")

        if multiple_types_significance == False:

            eff = get_efficiencies(inputtype='stage2',  raw=True, cut=f"{eff_cut}", verbose=verbose) #raw=True means includes preselection and BDT efficiencies

            sig_arr=[]
            sig_err_arr=[]
            sig_incl_error_arr=[]
            B_frac_err_arr=[]
            S_frac_err_arr=[]
            frac_sig_err_arr=[]
            sig_incl_Berror_arr=[]
            sig_incl_Berror_errorband_arr=[]

            #getting values for plots
            for n in signal_bf:
                S=0
                B=0
                S_var=0
                B_var=0

                n_expect = efficiency_finder.get_sample_expectations(eff, n, save=None, verbose=False, cut=eff_cut)

                for allocation in cfg.sample_allocations:
                    if allocation not in components:
                        continue
                    samples = cfg.sample_allocations[allocation]


                    if allocation=='signal':
                        S_arr = np.array([n_expect[f"{sample}_num"] for sample in samples]) 
                        Serr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples]) 

                        for j in np.arange(len(samples)):
                            S += S_arr[j]

                            S_var += Serr_arr[j]**2
                        
                    elif allocation=='background':
                        B_arr = np.array([n_expect[f"{sample}_num"] for sample in samples])
                        Berr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples])
                        
                        for j in np.arange(len(samples)):
                            B += B_arr[j]
                            B_var += Berr_arr[j]**2
                    

                significance = S / np.sqrt(S + B)
                sig_arr.append(significance)

                significance_error = np.sqrt((S_var*(2*B+S)**2+B_var*S**2)/(4*(S+B)**3))
                sig_err_arr.append(significance_error)
                frac_sig_err_arr.append(significance_error/significance)

                sig_incl_error = S / np.sqrt(S + B +B_var +S_var)
                sig_incl_error_arr.append(sig_incl_error)

                sig_incl_Berror = S / np.sqrt( B +B_var)
                sig_incl_Berror_arr.append(sig_incl_Berror)

                sig_incl_Berror_errorband = np.sqrt( S +S_var) / np.sqrt( B +B_var)
                sig_incl_Berror_errorband_arr.append(sig_incl_Berror_errorband)

                B_frac_err = np.sqrt(B_var)/B
                B_frac_err_arr.append(B_frac_err)

                S_frac_err = np.sqrt(S_var)/S
                S_frac_err_arr.append(S_frac_err)


            if graph =='significance':
                ax.set_ylabel(r'$\frac{S}{\sqrt{S+B}}$')
                
                if errorbar:
                    ax.errorbar(np.array(signal_bf),sig_arr,yerr=sig_err_arr, label= label)
                else: 
                    ax.plot( np.array(signal_bf),sig_arr, label= label)
                if errorband: 
                    ax.fill_between(np.array(signal_bf), np.array(sig_arr)-np.array(sig_err_arr), np.array(sig_arr)+np.array(sig_err_arr),alpha=0.55)

    
            if graph =='significance_incl_error':
                ax.set_ylabel(r'$\frac{S}{\sqrt{S+B+\sigma_S^2+\sigma_B^2}}$')
                ax.plot( np.array(signal_bf),sig_incl_error_arr, label= label)

            if graph =='s/full_berr': #this motivated by definition of p-value,ie. porbability in case of b only hypothesis that a statistical fluctuation gives the observed excess (s) ie. if saw s of size corresponding to BF=x, significace would be
                ax.set_ylabel(r'$\frac{S}{\sqrt{B+\sigma_B^2}}$') 
                ax.plot( np.array(signal_bf),sig_incl_Berror_arr, label= label)
                if errorband: #errorband due to uncertainty in s
                    ax.fill_between(np.array(signal_bf), np.array(sig_incl_Berror_arr)-np.array(sig_incl_Berror_errorband_arr), np.array(sig_incl_Berror_arr)+np.array(sig_incl_Berror_errorband_arr),alpha=0.55)

            #if graph =='B_error':
            #    ax.set_ylabel(r'$\frac{\sigma_B}{B}$')
            #    ax.plot( np.array(signal_bf),B_frac_err_arr, label= label)
            #if graph =='S_error':
            #    ax.set_ylabel(r'$\frac{\sigma_S}{S}$')
            #    ax.plot( np.array(signal_bf),S_frac_err_arr, label= label)
            #if graph =='Sig_err':
            #    ax.set_ylabel(r'Fractional Uncertainty in FoM')
            #    ax.plot( np.array(signal_bf),frac_sig_err_arr, label= label)
        

        if multiple_types_significance==True: #if list of cuts only first one used
            ax.set_ylabel(r'significance-like figure')
 


            # Regular expression to match the cut values for MVA1 and MVA2 in any order
            pattern = r'EVT_MVA1>=(\d+\.\d+)|EVT_MVA2>=(\d+\.\d+)'
            cut_string=eff_cut
            matches = re.findall(pattern, cut_string)
                
            # Initialize the cut values
            MVA1_cut, MVA2_cut = None, None
                
            # Loop through matches and assign the appropriate values
            for match in matches:
                if match:
                    if 'EVT_MVA1>=' in cut_string[cut_string.index(match[0])-10:cut_string.index(match[0])+10]:
                        MVA1_cut = float(match[0])
                    elif 'EVT_MVA2>=' in cut_string[cut_string.index(match[1])-10:cut_string.index(match[1])+10]:
                        MVA2_cut = float(match[1])

            xrange = (MVA2_cut, 1.00)

            bins = np.linspace(xrange[0],xrange[1],nbins+1)

            # Creating MVA2 cut expressions per bin
            cut_expr = []
            for b in range(len(bins)):
                if (b+1) == len(bins):
                    break
                elif (b+2) == len(bins):
                    cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut}) & (EVT_MVA2 >= {bins[b]}) & (EVT_MVA2 <= {bins[b+1]})')
                else:
                    cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut}) & (EVT_MVA2 >= {bins[b]}) & (EVT_MVA2 < {bins[b+1]})')

            effs = get_efficiencies('stage2', cut=cut_expr, raw=True, verbose=False)

            #defining empty arrays to fill
            
            av_significance_for_bf = []
            stdev_significance_for_bf=[]
            sig_arr=[]
            sig_err_arr=[]
            sig_incl_error_arr=[]
            sig_incl_Berror_arr=[]
            sig_incl_Berror_errorband_arr=[]

                

            for bf in signal_bf:
                # getting sample expectations and their error
                n_expect = efficiency_finder.get_sample_expectations(effs, bf, verbose=False, cut=cut_expr)

    
                S_perbin=np.zeros_like(n_expect[f"{cfg.samples[0]}_num"])
                B_perbin=np.zeros_like(n_expect[f"{cfg.samples[0]}_num"])
                S_var_perbin=np.zeros_like(n_expect[f"{cfg.samples[0]}_num"])
                B_var_perbin=np.zeros_like(n_expect[f"{cfg.samples[0]}_num"])

                for allocation in cfg.sample_allocations:
                    if allocation not in components:
                        continue
                    samples = cfg.sample_allocations[allocation]

                    if allocation=='signal':
                        S_arr = np.array([n_expect[f"{sample}_num"] for sample in samples]) 
                        Serr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples]) 

                        for j in np.arange(len(samples)):
                            S_perbin += S_arr[j]
                            S_var_perbin+= Serr_arr[j]**2
     
                        
                    elif allocation=='background':
                        B_arr = np.array([n_expect[f"{sample}_num"] for sample in samples])
                        Berr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples])
    
                        
                        for j in np.arange(len(samples)):
                            B_perbin += B_arr[j]
                            B_var_perbin += Berr_arr[j]**2


                #defining errors and poisson expectation
                sig_S_perbin = np.sqrt(S_var_perbin)
                sig_B_perbin = np.sqrt(B_var_perbin)


                if any(i=='significance' for i in graph):
                    significance_perbin = S_perbin / np.sqrt(S_perbin + B_perbin)
                    significance = np.sqrt(np.sum(np.square(significance_perbin)))
                    sig_arr.append(significance)
                    
                    significance_error_perbin = np.sqrt((S_var_perbin*(2*B_perbin+S_perbin)**2+B_var_perbin*S_perbin**2)/(4*(S_perbin+B_perbin)**3))
                    significance_error = np.sqrt(np.sum(np.square(2*significance_perbin*significance_error_perbin)))/(2*significance)
                    sig_err_arr.append(significance_error) 


                if any(i=='significance_incl_error'for i in graph):
                    sig_incl_error_perbin = S_perbin / np.sqrt(S_perbin + B_perbin +B_var_perbin +S_var_perbin)
                    sig_incl_error = np.sqrt(np.sum(np.square(sig_incl_error_perbin)))
                    sig_incl_error_arr.append(sig_incl_error)

                    #doesnt make sense to have an error band on this

                if any(i=='s/full_berr'for i in graph):
                    sig_incl_Berror_perbin = S_perbin / np.sqrt( B_perbin +B_var_perbin)
                    sig_incl_Berror = np.sqrt(np.sum(np.square(sig_incl_Berror_perbin)))
                    sig_incl_Berror_arr.append(sig_incl_Berror)

                    sig_incl_Berror_errorband_perbin = np.sqrt( S_perbin +S_var_perbin) / np.sqrt( B_perbin +B_var_perbin)
                    sig_incl_Berror_errorband = np.sqrt(np.sum(np.square(2*sig_incl_Berror_perbin*sig_incl_Berror_errorband_perbin)))
                    sig_incl_Berror_errorband_arr.append(sig_incl_Berror_errorband)


                if any(i=='toy_fit'for i in graph):
                    #defining variables for toys
                    poisson_expectation = B_perbin + S_perbin
                    average_background_error = np.mean( sig_B_perbin / B_perbin)


                    ## define fit to toy (this is the negative log likelihood to minimize)
                    def poisson_likelihood(sc_b, sc_s):
                        expectation = sc_b * B_perbin + sc_s * S_perbin
                        poiss_term = -np.sum( poisson.logpmf(toy_data, expectation)) #logpmf = Log of the probability mass function
                        bkg_constraint_term = -norm.logpdf( sc_b, 1, average_background_error )
                        return poiss_term + bkg_constraint_term
                    

                    significance_arr=[]

                    # throw and refit toys
                    for n in range(ntoys):
                        toy_data = np.random.poisson(poisson_expectation) #throw toys
                        mi = Minuit(poisson_likelihood, sc_b=1, sc_s=1 ) #fit toy
                        mi.migrad()
                        mi.hesse()

                        # refit with S=0 fixed for significance
                        mi0 = Minuit(poisson_likelihood, sc_b=1, sc_s=0 )
                        mi0.fixed['sc_s'] = True
                        mi0.migrad()
                        mi0.hesse()

                        significance = np.sqrt(abs(2*(mi.fval-mi0.fval)))
                        significance_arr.append(significance)

                    significance_av = np.average(significance_arr)
                    significance_stdev = np.std(significance_arr)
                    av_significance_for_bf.append(significance_av)
                    stdev_significance_for_bf.append(significance_stdev)

            #############################################
            #Plotting
            #############################################


            if any(i=='significance' for i in graph):
                ax.plot( np.array(signal_bf),sig_arr, label= r'$S/\sqrt{S+B}$')
                if errorband: 
                    ax.fill_between(np.array(signal_bf), np.array(sig_arr)-np.array(sig_err_arr), np.array(sig_arr)+np.array(sig_err_arr),alpha=0.55)


            if any(i=='s/full_berr'for i in graph):
                ax.plot( np.array(signal_bf),sig_incl_Berror_arr, label= r'$\frac{S}{\sqrt{B+\sigma_B^2}}$')
                if errorband: #errorband due to uncertainty in s
                    ax.fill_between(np.array(signal_bf), np.array(sig_incl_Berror_arr)-np.array(sig_incl_Berror_errorband_arr), np.array(sig_incl_Berror_arr)+np.array(sig_incl_Berror_errorband_arr),alpha=0.55)
  

            if any(i=='toy_fit'for i in graph):
                if toyinterpolation==True:
                    cubinterpolate_significance = interpolate.interp1d(np.array(signal_bf), av_significance_for_bf, kind = 'cubic')
                    cubinterpolate_sig_stdev = interpolate.interp1d(np.array(signal_bf), stdev_significance_for_bf, kind = 'cubic')
                    xnew = np.logspace(math.log10(signal_bf[0]),math.log10(signal_bf[-1]),80)
                    ax.plot(xnew, cubinterpolate_significance(xnew), label= r'$\sqrt{2\Delta\ln{\mathcal{L}}}$ mean'+' \nover '+f'{ntoys} toys')
                    if errorband: #errorband due to uncertainty in s
                        ax.fill_between(np.array(xnew), np.array(cubinterpolate_significance(xnew))-np.array(cubinterpolate_sig_stdev(xnew)), np.array( cubinterpolate_significance(xnew))+np.array(cubinterpolate_sig_stdev(xnew)),alpha=0.55)

                else:
                    ax.plot( np.array(signal_bf),av_significance_for_bf, label= r'$\sqrt{2\Delta\ln{\mathcal{L}}}$ mean'+' \nover '+f'{ntoys} toys')
                    if errorband: #errorband due to uncertainty in s
                        ax.fill_between(np.array(signal_bf), np.array(av_significance_for_bf)-np.array(stdev_significance_for_bf), np.array(av_significance_for_bf)+np.array(stdev_significance_for_bf),alpha=0.55)

            if any(i=='significance_incl_error'for i in graph):
                ax.plot( np.array(signal_bf),sig_incl_error_arr, label= r'$\frac{S}{\sqrt{S+B+\sigma_S^2+\sigma_B^2}}$')
                if errorband: 
                    ax.fill_between(np.array(signal_bf), np.array(sig_incl_error_arr), np.array(sig_incl_error_arr),alpha=0.55)



    if title:
        ax.set_title(title)
    else:
        cut_label1=eff_cut.replace(">=","$\geq$")
        cut_label=cut_label1.replace("&",",")
        if nbins==1:
            title_bin='bin'
        else:
            title_bin='equal width bins'
        if multiple_types_significance==True:
            ax.set_title(f'Tight BDT cuts {cut_label},\nbinning MVA2 with {nbins} {title_bin}')
        else:
            ax.set_title(f'Tight cuts on both BDTs: {cut_label},\nsingle bin counting experiment')


    
    ax.set_xlim(xmin,xmax)
    ax.legend()
    


    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        ax.set_xlabel(r'$\mathcal{B}(B_s^0 \rightarrow{} \nu \bar{\nu})$')

    if logx:
        ax.set_xscale('log')


    if logy:
        ax.set_yscale('log')

    if ylim:
        ax.set_ylim([ymin,ymax])

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)




#plot(['significance','significance_incl_error','toy_fit'],multiple_types_significance=True,errorband=True,cuts='wp1',ylim=(0,8),signal_bf=np.logspace(-8,-4, num=80)) #save='significance_vs_bf_combined_plot_wp1_0to8.pdf'
#plot(['significance','significance_incl_error','toy_fit'],multiple_types_significance=True,errorband=True,cuts='(EVT_MVA1>=0.9955)&(EVT_MVA2>=0.985)',ylim=(0,8),signal_bf=np.logspace(-8,-4, num=80),save='significance_vs_bf_combined_plot_onebinoptimum_MVA20985_Z0to8_5bins_toyinterpolation.pdf',toyinterpolation=True)
#plot(['significance','significance_incl_error','toy_fit'],multiple_types_significance=True,errorband=True,cuts='(EVT_MVA1>=0.9955)&(EVT_MVA2>=0.985)',ylim=(0,8),signal_bf=np.logspace(-8,-4, num=80),save='significance_vs_bf_combined_plot_onebinoptimum_MVA20985_Z0to8_1bin_toyinterpolation.pdf',nbins=1,toyinterpolation=True)  
#plot(['significance','significance_incl_error','toy_fit'],multiple_types_significance=True,errorband=True,cuts='(EVT_MVA1>=0.9955)&(EVT_MVA2>=0.985)',ylim=(0,8),signal_bf=np.logspace(-8,-4, num=50),ntoys=200,nbins=5,toyinterpolation=False,save='significancebf_combined_onebinoptimumMVA20985_errorband_Z0to8_5bin_notoyinterpolation_200toys.pdf')  
