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

import config as cfg
from efficiency_finder import get_efficiencies


    
##############################
## Plotting 
##############################


def plot(BDTcuts,
        cutvalues=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1],
        othercut=None,
        raw = False,
        errorbars=True,
        interactive=True, 
        save=None, 
        xtitle=None,
        range=None, 
        logy=False,
        components=["signal", "background"],
        verbose=False):

    """ 
    plot( BDTcuts, **opts ) will plot a variable

    Parameters
    ----------
    BDTcuts : str
        EVT_MVA1, EVT_MVA2 or EVT_MVAComb
        If not available those that are will be listed.
    cutvalues : list, optional
        List of values want to calculate efficiency at. Default: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1]
    othercut : str, optional
        Cut branch varname according to a (valid) UPROOT expression (requires barckets as if doing & ....). Default: None
    raw: bool, optional
        If True and cut is provided, use the eventsProcessed TParameter, otherwise use eventsSelected (ie. if true also includes preselection efficiency). Default = False. 
    errorbars: bool,optional
        If True plots error bars on efficiency. Default = True
    interactive : bool, optional
        Show the plot interactively after its made. Default: True
    save : str, optional
        Save file for the plot. If None then no plot is saved. Default: None 
    xtitle : str, optional
        Provide a custom title for the x axis. Default : `BDTcuts`
        Use this if LaTeX complains about the cut expression.
    range : tuple or list, optional
        The lower and upper limits to use in the plot. 
    logy : bool, optional
        Use log scale for the y axis. Default: False
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['signal', 'background']
    verbose : bool, optional
        Print out some useful stuff. Default: False
    """
    # If nchunks is a list, use corresponding elements


    if range is None:
        xmin = min(cutvalues) 
        xmax = max(cutvalues)
    else:
        xmin = range[0]
        xmax = range[1]


    ##############################
    ## List of cut expressions to pass to uproot and get efficiencies and errors
    ##############################
    if othercut==None:
        eff_cut = [f"({BDTcuts} > {i})" for i in cutvalues]
    else:
        eff_cut = [f"({BDTcuts} > {i}) & {othercut}" for i in cutvalues]

    eff = get_efficiencies(inputtype='stage2', cut=eff_cut, raw=raw, verbose=verbose)#, save=os.path.join(outputpath, f'{BDTcuts}_efficiency_dict.csv')


    #################################
    ## Plotting
    ###############################


    fig, ax = plt.subplots()


    for allocation in cfg.sample_allocations:
        if allocation not in components:
            continue
        samples = cfg.sample_allocations[allocation]
        y = [ eff[f"{sample}_eff"] for sample in samples ]
        y_err =[ eff[f"{sample}_err"] for sample in samples ]
        label = [ cfg.titles[sample] for sample in samples ]
        

        if allocation=='signal':
            color = 'cornflowerblue'

        elif allocation=='background':
            reds = mpl.colormaps['Reds_r']
            color = reds( np.linspace(0, 1, len(samples)+2)[1:-1] )


        for i in np.arange(0, len(samples),1):
            ax.plot( np.array(cutvalues),np.array(y)[i],color = color[i], label=label[i])
            ax.set_xlim(xmin,xmax)
            
            if errorbars:
                ax.errorbar( np.array(cutvalues),np.array(y)[i],yerr = np.array(y_err)[i],color = color[i], fmt='none')




    ax.legend()
    ax.set_ylabel("MVA cut efficiency")

    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        ax.set_xlabel(f'{BDTcuts} cut value (cut={othercut})')

    if logy:
        ax.set_yscale('log')

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)

