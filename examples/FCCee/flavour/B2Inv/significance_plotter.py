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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

    
##############################
## Plotting 
##############################


def plot(BDTcuts,
        significance_fig = 'S/sqrt(S+B)',
        SandB=False,
        cutvalues=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1],
        signal_bf=1e-5,
        othercut=None,
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
    significance_fig: str
        'S/sqrt(S+B)', 'S/sqrt(B+sigB^2)', 'S/sqrt(B+S+sigB^2+sigS^2)'
        Significance figure to be used. Default: 'S/sqrt(S+B)'
    SandB: bool, optional
        If False, plots significance, otherwise plots weighted number (ie. S and B). Default=False
    cutvalues : list, optional
        List of values want to calculate efficiency at. Default: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1]
    signal_bf : float, optional
        The assumed signal branching fraction to use with the weights. Default = 10^-5
    othercut : str, optional
        Cut branch varname according to a (valid) UPROOT expression. Default: None
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
    ## List of cut expressions to pass to uproot and get efficiencies, expected number of events and errors
    ##############################

    if othercut==None:
        eff_cut = [f"({BDTcuts} >= {i})" for i in cutvalues]
    else:
        eff_cut = [f"({BDTcuts} >= {i}) & {othercut}" for i in cutvalues]

    eff = get_efficiencies(inputtype='stage2',  raw=True, cut=eff_cut, verbose=verbose) #raw=True means includes preselection and BDT efficiencies
    n_expect = efficiency_finder.get_sample_expectations(eff, signal_bf, save=None, verbose=verbose, cut=eff_cut)
    

    #################################
        ## Plotting
    ###############################


    fig, ax = plt.subplots()

    B = np.zeros(len(cutvalues))
    S = np.zeros(len(cutvalues))
    S_var = np.zeros(len(cutvalues))
    B_var = np.zeros(len(cutvalues))

    for allocation in cfg.sample_allocations:
        if allocation not in components:
            continue
        samples = cfg.sample_allocations[allocation]
        label = [ cfg.titles[sample] for sample in samples ]


        if allocation=='signal':
            color = 'cornflowerblue'
            Sarr = np.array([n_expect[f"{sample}_num"] for sample in samples])
            Serr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples]) 

            for i in np.arange(len(samples)):
                S += Sarr[i]
                S_var += Serr_arr[i]**2


            if SandB:
                ax.plot( np.array(cutvalues),np.array(Sarr)[0], label=r"$B_s^0 \to \nu \bar{\nu}$",color='cornflowerblue')
        
        
            
        elif allocation=='background':
            reds = mpl.colormaps['Reds_r']
            color = reds( np.linspace(0, 1, len(samples)+2)[1:-1] )
            Barr = np.array([n_expect[f"{sample}_num"] for sample in samples])
            Berr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples])
            
            for i in np.arange(len(samples)):
                B += Barr[i]
                B_var += Berr_arr[i]**2
            
            if SandB:
                for i in np.arange(0, len(samples),1):
                    ax.plot( np.array(cutvalues),np.array(Barr)[i], color=color[i], label=label[i])
                    ax.set_ylabel("Expected Number")

    if significance_fig=='S/sqrt(S+B)':
        significance = S / np.sqrt(S + B )
        ylabel= r"$\frac{S}{\sqrt{S+B}}$"

    if significance_fig== 'S/sqrt(B+S+sigB^2+sigS^2)':
        significance = S / np.sqrt(S + B +S_var +B_var )
        ylabel= r"$\frac{S}{\sqrt{S+B+\sigma_S^2+\sigma_B^2}}$"

    if significance_fig== 'S/sqrt(B+sigB^2)':
        significance = S / np.sqrt(B +B_var)
        ylabel= r"$\frac{S}{\sqrt{B+\sigma_B^2}}$"

    if SandB==False:
        ax.plot( np.array(cutvalues),significance)
        ax.set_ylabel(ylabel)

    ax.set_xlim(xmin,xmax)
    ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')
    ax.legend()
    

    print('MVA cut for optimum point = '+ str(cutvalues[np.nanargmax(significance)]))


    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        cut_label1=othercut.replace(">=","$\geq$")
        cut_label=cut_label1.replace("&",",")
        ax.set_xlabel(f'{BDTcuts} cut value (cut={cut_label})')

    if logy:
        ax.set_yscale('log')

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)




    
##############################
## Plotting 2D
##############################


def plot2D(
        significance_fig = 'S/sqrt(S+B)',
        MVA1_cutvalues=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1],
        MVA2_cutvalues=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1],
        signal_bf=1e-5,
        othercut=None,
        interactive=True, 
        save=None, 
        xtitle=None,
        components=["signal", "background"],
        verbose=False,
        significance_min=None):

    """ 
    plot( **opts ) will plot a variable

    Parameters
    ----------
    significance_fig: str
        'S/sqrt(S+B)', 'S/sqrt(B+sigB^2)', 'S/sqrt(B+S+sigB^2+sigS^2)'
        Significance figure to be used. Default: 'S/sqrt(S+B)'
    MVA1_cutvalues : list, optional
        List of MVA1 values want to calculate efficiency at. Default: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1]
    MVA2_cutvalues : list, optional
        List of MVA2 values want to calculate efficiency at. Default: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99,1]
    signal_bf : float, optional
        The assumed signal branching fraction to use with the weights. Default = 10^-5
    othercut : str, optional
        Cut branch varname according to a (valid) UPROOT expression. Default: None
    interactive : bool, optional
        Show the plot interactively after its made. Default: True
    save : str, optional
        Save file for the plot. If None then no plot is saved. Default: None 
    xtitle : str, optional
        Provide a custom title for the x axis. Default : `BDTcuts`
        Use this if LaTeX complains about the cut expression.
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['signal', 'background']
    verbose : bool, optional
        Print out some useful stuff. Default: False
    significance_min : float,optional
        minimum significance value to show on the heatmap scale. Default=None
    """
    # If nchunks is a list, use corresponding elements



    xmin = min(MVA1_cutvalues) 
    xmax = max(MVA1_cutvalues)



    ##############################
    ## List of cut expressions to pass to uproot and get efficiencies, expected number of events and errors
    ##############################

    if othercut==None:
        eff_cut = [f"(EVT_MVA1 >= {i})&(EVT_MVA2 >= {j})" for i in MVA1_cutvalues for j in MVA2_cutvalues]
        ycoordinate=[i for i in MVA1_cutvalues for j in MVA2_cutvalues]
        xcoordinate=[j for i in MVA1_cutvalues for j in MVA2_cutvalues]

    else:
        eff_cut = [f"(EVT_MVA1 >= {i})&(EVT_MVA2 >= {j})& {othercut}" for i in MVA1_cutvalues for j in MVA2_cutvalues]
        #coordinate=[(i,j) for i in MVA1_cutvalues for j in MVA2_cutvalues]
        ycoordinate=[i for i in MVA1_cutvalues for j in MVA2_cutvalues]
        xcoordinate=[j for i in MVA1_cutvalues for j in MVA2_cutvalues]
    

    eff = get_efficiencies(inputtype='stage2',  raw=True, cut=eff_cut, verbose=verbose) #raw=True means includes preselection and BDT efficiencies
    #setting all 0 to small value st domt get errors
    #eff[eff == 0] = 1e-40
    n_expect = efficiency_finder.get_sample_expectations(eff, signal_bf, save=None, verbose=verbose, cut=eff_cut)


    #################################
        ## Plotting
    ###############################


    fig, ax = plt.subplots()

    B = np.zeros_like(xcoordinate)
    S = np.zeros_like(xcoordinate)
    S_var = np.zeros_like(xcoordinate)
    B_var = np.zeros_like(xcoordinate)


    for allocation in cfg.sample_allocations:
        if allocation not in components:
            continue
        samples = cfg.sample_allocations[allocation]
        label = [ cfg.titles[sample] for sample in samples ]


        if allocation=='signal':
            Sarr = np.array([n_expect[f"{sample}_num"] for sample in samples])
            Serr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples]) 

            for i in np.arange(len(samples)):
                S += Sarr[i]
                S_var += Serr_arr[i]**2


            
        elif allocation=='background':
            Barr = np.array([n_expect[f"{sample}_num"] for sample in samples])
            Berr_arr = np.array([n_expect[f"{sample}_err"] for sample in samples])
            
            for i in np.arange(len(samples)):
                B += Barr[i]
                B_var += Berr_arr[i]**2


    if significance_fig=='S/sqrt(S+B)':
        significance = S / np.sqrt(S + B )
        zlabel= r"$\frac{S}{\sqrt{S+B}}$"

    if significance_fig== 'S/sqrt(B+S+sigB^2+sigS^2)':
        significance = S / np.sqrt(S + B +S_var +B_var )
        zlabel= r"$\frac{S}{\sqrt{S+B+\sigma_S^2+\sigma_B^2}}$"

    if significance_fig== 'S/sqrt(B+sigB^2)':
        significance = S / np.sqrt(B +B_var)
        zlabel= r"$\frac{S}{\sqrt{B+\sigma_B^2}}$"


    # Determine the size of the grid 
    x_max, y_max = len(set(xcoordinate)), len(set(ycoordinate))
    # Initialize a 2D array with zeros 
    z_grid = np.zeros((y_max, x_max)) 
    #create integrer list of positions
    sorted_xcoordinate = sorted(set(xcoordinate))
    sorted_ycoordinate = sorted(set(ycoordinate))

    # Find the indexes of each element in the original list within the sorted unique list
    xindexes = [sorted_xcoordinate.index(val) for val in xcoordinate]
    yindexes = [sorted_ycoordinate.index(val) for val in ycoordinate]


    # Fill the 2D array with z values 
    for k in range(len(xcoordinate)): 
         z_grid[yindexes[k], xindexes[k]] = significance[k] 
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if significance_min:
        im=ax.imshow(z_grid,cmap='plasma',vmin=significance_min)
    else:
        im=ax.imshow(z_grid,cmap='plasma')
    ax.set_ylabel('EVT_MVA1')
    ax.set_xlabel('EVT_MVA2')



    # Define custom tick positions and labels
    x_ticks = sorted(set(xindexes)) # positions 
    y_ticks = sorted(set(yindexes)) # positions 
    x_labels = sorted(set(xcoordinate)) # new labels for x-axis 
    y_labels = sorted(set(ycoordinate))
            
    ax.set_xticks(x_ticks) 
    ax.set_yticks(y_ticks) 
    ax.set_xticklabels([round(num,4) for num in x_labels],rotation=90)
    ax.set_yticklabels([round(num,4) for num in y_labels])
    # Remove subdivision ticks 
    ax.xaxis.set_tick_params(which='minor', bottom=False, top=False) 
    ax.yaxis.set_tick_params(which='minor', left=False, right=False)
    ax.xaxis.set_tick_params(which='major', top=False) 
    ax.yaxis.set_tick_params(which='major',right=False)
    
    ax.legend()
    
    fig.colorbar(im, cax=cax, orientation='vertical',label=zlabel)



    print('MVA cut for optimum point:'+ str(eff_cut[np.nanargmax(significance)]))


    if xtitle is not None:
        ax.set_title(xtitle)
    else:
        if othercut:
            cut_label1=othercut.replace(">=","$\geq$")
            cut_label=cut_label1.replace("&",",")
            ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}, other cuts ={cut_label}')
        else:
            ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')



    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)



