import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from efficiency_tools import post_bdtlh_efficiency_finder as eff_finder

plt.style.use('fcc.mplstyle')

def histogram_settings():
    
    hist_settings = { allocation: {} for allocation in cfg.sample_allocations }
    total_color = { allocation: {} for allocation in cfg.sample_allocations }

    for allocation in cfg.sample_allocations:
        samples = cfg.sample_allocations[allocation]
            
        if allocation=='Bssignal':
            hist_settings[allocation]['histtype'] = 'step'
            hist_settings[allocation]['lw'] = 2
            hist_settings[allocation]['color'] = plt.cm.Blues( np.linspace(0, 1, len(cfg.sample_allocations['combined_signal'])+4)[3]) #'cornflowerblue'
            hist_settings[allocation]['hatch'] = '////'
        elif allocation=='Bdsignal':
            hist_settings[allocation]['histtype'] = 'step'
            hist_settings[allocation]['lw'] = 2
            hist_settings[allocation]['color'] = plt.cm.Blues( np.linspace(0, 1, len(cfg.sample_allocations['combined_signal'])+4)[-2]) #'mediumblue'#'royalblue'
            hist_settings[allocation]['hatch'] = r'\\\\'
        elif allocation=='combined_signal':
            hist_settings[allocation]['histtype'] = 'step'
            hist_settings[allocation]['lw'] = 2
            hist_settings[allocation]['color'] =plt.cm.Blues( np.linspace(0, 1, len(samples)+4)[3:-1] ) #['cornflowerblue','mediumblue']#['cornflowerblue', 'dodgerblue',]#['cadetblue','teal']#['cornflowerblue','mediumblue']
            total_color[allocation] = 'midnightblue'
            hist_settings[allocation]['hatch'] = '////'

        elif allocation=='hadronic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.Reds_r( np.linspace(0, 1, len(samples)+2)[1:-1] )
            total_color[allocation] = 'k'
        elif allocation=='light_hadronic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.Reds_r( np.linspace(0, 1, len(cfg.sample_allocations['hadronic_background'])+2)[3:-1] )
            total_color[allocation] = 'indianred'
        elif allocation=='heavy_hadronic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.Reds_r( np.linspace(0, 1, len(cfg.sample_allocations['hadronic_background'])+2)[1:3] )
            total_color[allocation] = 'darkred'
        elif allocation=='bb_only':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['lw'] = 2
            hist_settings[allocation]['alpha'] = 0.6
            hist_settings[allocation]['color'] = plt.cm.Reds_r( np.linspace(0, 1, len(cfg.sample_allocations['hadronic_background'])+2)[1] )

        elif allocation=='tau_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.tab20b((4+ np.linspace(0, 1, len(cfg.sample_allocations['leptonic_background'])+2)[1])/5 )#'mediumvioletred'
            total_color[allocation] = 'indigo'
        elif allocation=='leptonic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.tab20b( (4+np.linspace(0, 1, len(samples)+2)[1:-1]) /5 )# plt.cm.tab20b((4+ np.linspace(0, 1, len(samples)))/5 ) #plt.cm.PuRd_r( np.linspace(0, 1, len(samples)+2)[1:-1] ) 
            total_color[allocation] = 'indigo'#'purple'
        elif allocation=='light_leptonic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.tab20b((4+ np.linspace(0, 1, len(cfg.sample_allocations['leptonic_background'])+2)[2:-1])/5 )
            total_color[allocation] = 'mediumvioletred'

    return hist_settings, total_color

#Function to enable automatic xtitle with > or <
def replace_all(s, old_char, new_char):
    # Replace all occurrences of the old character with the new character
    s = s.replace(old_char, new_char)
    return s

     
def plot_variable(data,variable,
                    cut = None, #must be string of correct format as in vp
                    bdt_name = "BDT_lh",
                    weight=True,
                    density=False,
                    bins=50,
                    xrange=None,
                    outpath=None,
                    stacked=True, 
                    total=["hadronic_background"], 
                    components=["combined_signal", "hadronic_background"], #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue
                    signal_bf=1e-6,
                    interative = True,
                    verbose=False):
    
    
    if outpath:
        savepath=os.path.join(outpath,f'{variable}_with_{bdt_name}_cut_{cut}.pdf')
    else:
        savepath= f'{variable}_with_{bdt_name}_cut_{cut}.pdf'


    decays_list = [cfg.sample_allocations[i] for i in components]
    flat_decays_list = [item for sublist in decays_list for item in sublist]


    if cut:
        df = data.copy().query(cut)
    else:
        df = data.copy()


    values = {sample: df[df['decay']==sample][variable].to_numpy() for sample in flat_decays_list}

    if xrange is None:
        xmin = min( [ min(values[sample]) for sample in values.keys() if len(values[sample])>0] )
        xmax = max( [ max(values[sample]) for sample in values.keys() if len(values[sample])>0] )

    else:
        xmin = xrange[0]
        xmax = xrange[1]

    # If variable is an integer make sure nbins correct
    #for most variables nbins=xmax-xmin
    #unless ChargedRP_fromPV_transformed
    if bins!=50:
        nbins = bins
    elif 'ChargedRP_fromPV_transformed' in variable:
        nbins=3
    elif '_n' in variable and '_norm' not in variable and '_not' not in variable:
        print(variable)
        xmin = 0
        nbins= int(xmax - xmin)
    else:
        nbins=50

    fig, ax = plt.subplots()

    hist_settings,total_colours = histogram_settings()

    if density==False:
        ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')

    for allocation in cfg.sample_allocations:

        if allocation not in components:
            continue

        samples = cfg.sample_allocations[allocation]
        hist_x = [ values[sample] for sample in samples ]
        hist_l = [ cfg.titles[sample] for sample in samples ]
        hist_opts = hist_settings[allocation]
        tot_colour= total_colours[allocation]

        if weight:
            eff, err, n_remaining = eff_finder.get_total_eff_post_bdt(df, cut=None, verbose=verbose) #already filtered df on cut earlier
            n_exp, n_err, BFZbb_err_dict_components = eff_finder.get_n_expected(eff, err, signal_bf=signal_bf, calc_BFZbb_err=True)

            hist_w = [ n_exp[sample]/len(df[df['decay']==sample])* np.ones_like(values[sample]) for sample in samples ] 

            print([f'n_{sample}={n_exp[sample]}'for sample in samples ])

        else:
            hist_w = None

        if stacked:
            hist_opts['stacked'] = True
        else:
            hist_opts['stacked'] = False

        ax.hist( 
            x = hist_x,
            bins = nbins,
            range = (xmin,xmax),
            density = density,
            label = hist_l,
            weights = hist_w,
            **hist_opts
        )

        if allocation in total and stacked:
            ax.hist( 
                np.concatenate( hist_x), 
                bins = nbins,
                range = (xmin,xmax),
                density = density,
                label = f'Total {allocation.replace("_", " ")}',
                weights = np.concatenate( hist_w ) if weight else None,
                histtype = 'step',
                color = tot_colour,#'k',
                lw = 2,
            )

    ax.legend(reverse=True)



    if cut is not None:
        ax.set_xlabel(f"{variable} (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
    else:
        ax.set_xlabel(f"{variable} (cut={cut})")

    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Counts')
   

    fig.tight_layout()

    if interative == True:
        plt.show()

    if outpath is not None:
        fig.savefig(savepath)

    
