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


    
##############################
## Plotting 
##############################


def plot(type,
        signal_bf=np.logspace(-7,-4, num=50),
        errorband=False,
        cut='wp1',
        interactive=True, 
        save=None, 
        xtitle=None,
        range=None, 
        logx=True,
        components=["signal", "background"],
        verbose=False):

    """ 
    plot( BDTcuts, **opts ) will plot a variable

    Parameters
    ----------
    type: str
        'significance' or 'uncertainty'
    signal_bf : float, optional
        The assumed signal branching fraction to be plotted against. Default = np.logspace(1e-7,1e-4)
    errorband : bool, optional
        Plot band showing effect of B+/- deltaB. Default=False
    cut : str, optional
        'wp1' or cut branch varname according to a (valid) UPROOT expression.
        Which cuts to use on data. Default: wp1
    interactive : bool, optional
        Show the plot interactively after its made. Default: True
    save : str, optional
        Save file for the plot. If None then no plot is saved. Default: None 
    xtitle : str, optional
        Provide a custom title for the x axis. Default : `BDTcuts`
        Use this if LaTeX complains about the cut expression.
    range : tuple or list, optional
        The lower and upper limits to use in the plot. 
    logx : bool, optional
        Use log scale for the x axis. Default: True
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['signal', 'background']
    verbose : bool, optional
        Print out some useful stuff. Default: False
    """
  


    if range is None:
        xmin = min(signal_bf) 
        xmax = max(signal_bf)
    else:
        xmin = range[0]
        xmax = range[1]


    ##############################
    ## List of cut expressions to pass to uproot and get efficiencies, expected number of events and errors
    ##############################

    if cut =='wp1':   
        #cuts from config at current working point
        cutforms = [f'{key}>{value}' for key, value in cfg.wp1_cuts.items()]
        #putting cuts in correct format
        eff_cut=str(f'({cutforms[0]})')
        for i in np.arange(1,len(cutforms),1):
            eff_cut += str(f' & ({cutforms[i]})')

    else:
        eff_cut = cut
   

    eff = get_efficiencies(inputtype='stage2',  raw=True, cut=f"{eff_cut}", verbose=verbose) #raw=True means includes preselection and BDT efficiencies

    #################################
        ## Plotting
    ###############################


    fig, ax = plt.subplots()

    sig_arr=[]
    sig_l_arr=[]
    sig_u_arr=[]
    uncertainty_arr=[]
    unc_l_arr=[]
    unc_u_arr=[]


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

                for i in np.arange(len(samples)):
                    S += S_arr[i]
                    S_var += Serr_arr[i]**2
                
            elif allocation=='background':
                B_arr = np.array([n_expect[f"{sample}_num"] for sample in samples])
                Berr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples])
                
                for i in np.arange(len(samples)):
                    B += B_arr[i]
                    B_var += Berr_arr[i]**2
            

        significance = S / np.sqrt(S + B)
        sig_arr.append(significance)
        uncertainty_arr.append(1/significance)

        sig_l= S / np.sqrt(S + (B+np.sqrt(B_var)))
        sig_l_arr.append(sig_l)
        sig_u= S / np.sqrt(S + (B-np.sqrt(B_var)))
        sig_u_arr.append(sig_u)
        unc_l_arr.append(1/sig_u)
        unc_u_arr.append(1/sig_l)

        print(np.sqrt(B_var)/B)

    if type =='significance':
        ax.plot( np.array(signal_bf),sig_arr, label= r'$S/\sqrt(S+B)$')
        ax.set_ylabel('Significance')
        if errorband:
            ax.plot(np.array(signal_bf),sig_u_arr, color='r',linestyle='dashed',label= r'$S/\sqrt(S+(B \pm \sigma_B))$')
            ax.plot(np.array(signal_bf),sig_l_arr, color='r',linestyle='dashed')

    if type =='uncertainty':
        ax.plot( np.array(signal_bf),uncertainty_arr, label= r'$\sqrt(S+B)/S$')
        ax.set_ylabel(r'Uncertainty, $\sqrt(S+B)/S$')
        if errorband:
            ax.plot(np.array(signal_bf),unc_u_arr, color='r',linestyle='dashed',label= r'$\sqrt(S+(B \pm \sigma_B))/S$')
            ax.plot(np.array(signal_bf),unc_l_arr, color='r',linestyle='dashed')
   
    if cut=='wp1':
        ax.set_title(r"Cuts = MVA1$>0.994$, MVA2$>0.95$")
    else:
        ax.set_title(f"Cut={cut}")

    
    ax.set_xlim(xmin,xmax)
    ax.legend()
    


    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        ax.set_xlabel(f'Signal BF')

    if logx:
        ax.set_xscale('log')

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)

