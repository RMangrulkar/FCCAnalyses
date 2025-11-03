## Matt's new attempt at the plotting module for the paper ##
## So that the hatching works ##
import os
import sys
from glob import glob
import numpy as np
import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from efficiency_tools import efficiency_finder as ef
from plotters import variable_plotter as vp

plt.style.use('fcc.mplstyle')

# path = "../ella_files/no_selection"
#
# signals = cfg.sample_allocations["combined_signal"]
# backgrounds = cfg.sample_allocations["hadronic_background"]
#
# effs = ef.get_efficiencies('custom', cut=None, raw=True, custompath=path, verbose=True, samples=signals+backgrounds) 
# n_expect = ef.get_sample_expectations(effs, placeholder_bf=1e-6, verbose=True, cut=None) 

def make_the_bloody_plot(var, range, bins, inputpath, samples, normalisation='norm', cut=None, signal_bf=1e-6, plot_cutline=None, xtitle=None, save=None, effpath=None, logy=False):
    """ 
    New plotter which has to hack around various bits for paper plots
    It gets passed a list of samples which must be in the keys of config.sample_allocations
    It will stack everything within that sample

    Parameters
    ----------
    var: str
        Variable name (composites not yet allowed)
    range: tuple of float
        Range for the historgams
    bins: int
        Number of bins for the histograms
    inputpath: str
        Path to where the input files live (expects substructure)
    samples: list of str
        List of samples to plot which much be in the keys of config.sample_allocations
    normalisation: str 
        Must be one of 'none', 'norm', 'density'
    cut: str
        Apply a cut(s)
    plot_cutline: float or tuple of (float, str) or tuple of (float, str, str)
        If just a float which draw vertical dashed line at this float value
        If tuple of (float, str) will draw vertical dashed line at this float value and add item to legend with title in str
        If tuple of (float, str, str) will do the above and also add an arrow if the third item is either "<" or ">"
    effpath: str
        Path to where efficiencies are calculated
        
    """

    # check normalisation argument can be one of 'none', 'norm' or 'density'
    # these will do the following:
    #    'none': apply no normalisation just take raw counts
    #    'norm': apply for normalised counts such that each sample in samples will be normalised so that the sum of each bin is unity
    #    'density': apply for probability density such that each sample in samples will be normalised so that the sum of bin area is unity, ie. the integral of the histogram is one (thus y-axis will have inverse units of x-axis
    if normalisation not in ['none', 'norm', 'density']:
        raise RuntimeError(f"normalisation must be one of 'none', 'norm', 'density' and not {normalisation}")

    # expect a list of samples from cfg.sample_allocations
    full_sample_list = []
    for sample in samples:
        allowed_samples = list(cfg.sample_allocations.keys())
        if sample not in allowed_samples:
            raise RuntimeError(f"Sample {sample} not a valid choice. Must be in {allowed_samples}")
        else:
            for subsample in cfg.sample_allocations[sample]:
                full_sample_list.append(subsample)
    
    # shouldn't be any repeats
    full_sample_list = list(set(full_sample_list))

    # get the efficiencies
    if effpath is None:
        effpath = inputpath
    effs = ef.get_efficiencies('custom', cut=cut, raw=True, custompath=effpath, verbose=True, samples=full_sample_list) 
    n_expect = ef.get_sample_expectations(effs, signal_bf, verbose=True, cut=cut) 

    # make the raw histograms
    histograms = {}
    for sample in full_sample_list:
        values = vp.as_array( sample, var, cut=None, nchunks=None, inputpath=inputpath )
        weight = np.ones_like(values)*n_expect[sample+'_num'] / len(values)
        nh, xe = np.histogram( values, range=range, bins=bins, weights=weight )
        histograms[sample] = nh
    
    # bin info
    bin_widths = np.diff(xe)
    bin_centres = 0.5 * ( xe[:-1] + xe[1:] )

    # now normalise the histograms if asked
    if normalisation != 'none':
        
        for sample in samples:
            sample_total = np.sum( [histograms[subsample] for subsample in cfg.sample_allocations[sample] ] )
            
            if normalisation == 'norm': 
                sample_area = np.sum( sample_total )
    
            elif normalisation == 'density':
                sample_area = np.sum( sample_total * bin_widths )

            for subsample in cfg.sample_allocations[sample]:
                histograms[subsample] = histograms[subsample] / sample_area

    # now plot
    fig, ax = plt.subplots()

    legend_entries = []

    # loop over the allocations (this would be e.g. combined_signal, hadronic_background)
    for i, sample in enumerate(samples):

        bottom = np.zeros_like( bin_centres )
        top = np.zeros_like( bin_centres )
        
        colors = cfg.sample_colors[sample]
        hatches = cfg.sample_hatches[sample]
        
        # loop over the subsamples within each allocation (e.g. BsNuNu, BdNuNu etc.)
        for j, subsample in enumerate(cfg.sample_allocations[sample]):
            top += histograms[subsample]

            # if hatched then need to draw a few times 
            if hatches is not None:
                # draw hatches
                ax.bar( bin_centres, histograms[subsample], bottom=bottom, width=bin_widths, align='center',
                        facecolor='none', edgecolor=colors[j], hatch=hatches[j], lw=0, zorder=100*(5-i)+10*j )
                # draw the outline
                ax.step( bin_centres, top, where='mid', 
                        color=colors[j], linewidth=2, zorder=100*(5-i)+20+10*i )
                # and have to add the half steps at each end by hand (sigh)
                ax.plot( [bin_centres[0]-bin_widths[0]/2, bin_centres[0]], [top[0], top[0]], lw=2, c=colors[j], zorder=100*(5-i)+40+10*i )
                ax.plot( [bin_centres[-1], bin_centres[-1]+bin_widths[-1]/2], [top[-1], top[-1]], lw=2, c=colors[i], zorder=100*(5-i)+60+10*i )

                # make the legend entry
                legend_entries.append( mpl.patches.Patch( facecolor='none', edgecolor=colors[j], hatch=hatches[j], linewidth=2, label=cfg.titles[subsample] ) )
            # if not hatched do something else
            else:
                ax.bar( bin_centres, histograms[subsample], bottom=bottom, width=bin_widths, align='center',
                        color=colors[j], edgecolor=colors[j], linewidth=0.5)
        
                # make the legend entry
                legend_entries.append( mpl.patches.Patch( facecolor=colors[j], edgecolor='none', label=cfg.titles[subsample] ) )
            
            bottom += histograms[subsample]
        
        # plot total if asked
        if sample in cfg.sample_total.keys():
            if cfg.sample_total[sample] is not None:
                
                # plot the total histogram
                ax.step( bin_centres, top, where='mid', color=cfg.sample_total[sample], lw=2 )
                # have to add ends of the step by hand (sigh)
                ax.plot( [bin_centres[0]-bin_widths[0]/2, bin_centres[0]], [top[0], top[0]], lw=2, c=cfg.sample_total[sample] )
                ax.plot( [bin_centres[-1], bin_centres[-1]+bin_widths[-1]/2], [top[-1], top[-1]], lw=2, c=cfg.sample_total[sample] )

                # make the legend entry
                legend_entries.append( mpl.patches.Patch( facecolor='none', edgecolor=cfg.sample_total[sample], lw=2, label=cfg.titles[sample] ) )
    
    # plot cut line if asked
    if plot_cutline is not None:
        xval_cutline = None
        if np.isscalar( plot_cutline ):
            xval_cutline = plot_cutline
        else:
            xval_cutline = plot_cutline[0]

        ax.axvline( x=xval_cutline, color='k', linestyle='--', lw=1, zorder=900 )

        if not np.isscalar( plot_cutline ):
            if len(plot_cutline)>1:
                legend_entries.append( mpl.lines.Line2D( [0], [0], color='k', lw=1, ls='--', label=plot_cutline[1] ) )
            if len(plot_cutline)>2:
                yval = ax.get_ylim()[1]*0.9
                xrng = np.diff(ax.get_xlim())[0]*0.05
                arr_start = (plot_cutline[0], yval)
                if plot_cutline[2]==">":
                    sign = 1
                elif plot_cutline[2]=="<":
                    sign = -1
                else:
                    raise RuntimeError(f"Invalid value for plot_cutline 3rd element {plot_cutline[2]}. Must be either '>' or '<'")
                arr_end = (plot_cutline[0] + sign*xrng, yval)
                ax.annotate('', xy=arr_end, xytext=arr_start, arrowprops=dict(arrowstyle='-|>', color='k', lw=2) )
        

    # draw the legend
    ax.legend(handles=legend_entries)

    # set range
    if not logy:
        ax.set_ylim(bottom=0)

    # log?
    if logy:
        ax.set_yscale('log')
    
    # x title
    if xtitle is not None:
        ax.set_xlabel(xtitle)
        if normalisation=='none':
            ax.set_ylabel("Event Counts")
        elif normalisation=='norm':
            ax.set_ylabel("Normalised Counts")
        elif normalisation=='density':
            ax.set_ylabel("Density")

    if save is not None:
        fig.savefig(save)


# make_the_bloody_plot( var="EVT_e", range=(25,95), bins=70, inputpath=path, samples=["combined_signal", "hadronic_background"], xtitle='Total Event Energy [GeV]', save="figs/EVT_e_prelim.pdf" )
# make_the_bloody_plot( var="EVT_e", range=(25,95), bins=70, xtitle='Total Event Energy [GeV]', save="figs/EVT_e_prelim.pdf" )
# make_the_bloody_plot( var="EVT_hemisEmin_nCharged", range=(0,25), bins=25, xtitle='Number of Charged Tracks in the Signal Hemisphere', save="figs/EVT_hemisEmin_nCharged_prelim.pdf" )
# make_the_bloody_plot( var="EVT_hemisEmin_nLept", range=(0,5), bins=5, xtitle='Total Number of $e^\pm$ and $\mu^\pm$ in the Signal Hemisphere', save="figs/EVT_hemisEmin_nLept_prelim.pdf" )
# make_the_bloody_plot( var="PV_Rec_vtx_m", range=(0,90), bins=45, xtitle='Reconstructed Mass of the Primary Vertex [GeV]', save="figs/PV_Rec_vtx_m_prelim.pdf" )

# plt.show()
