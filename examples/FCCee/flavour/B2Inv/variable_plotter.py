# variable_plotter.py
# Script to interactively plot features from a specified inputpath
import os
from glob import glob
import numpy as np
import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak  # Needed if using awkward arrays
plt.style.use('fcc.mplstyle')

# go configure 
import config as cfg
import efficiency_finder

# I'd like to have an argument please
from argparse import ArgumentParser
parser = ArgumentParser(description="Interactively plots features from a specified inputpath")
parser.add_argument("-i","--inputpath", default=f"{cfg.fccana_opts['outputDir']['stage2']}", help="Path to look for files in, default is the stage2 directory in config.py")
parser.add_argument("-e","--efficiencies", default=None, help="Name of efficiency dictionary key, default is None")
args = parser.parse_args()

def get_list_of_files(folder):
    
    path = os.path.join( args.inputpath, folder )
    if not os.path.exists( path ):
        raise RuntimeError( f"No such path {path}" )
    
    files = glob( os.path.join(args.inputpath, folder, "*.root") )
    if len(files)==0:
        raise RuntimeError( f"No root files found in {path}" )

    return files

def get_list_of_branches(folder):
    f0 = get_list_of_files(folder)[0]
    try:
        tr = uproot.open( f0+":events" )
        return [ br.name for br in tr.branches ]
    except:
        raise RuntimeError( f"No tree called events found in file {f0}" )

def check_var(folder, varname):
    branches = get_list_of_branches(folder)
    if varname not in branches:
        print( f"Branches found in files at path {folder}:" )
        for br in branches:
            print('  ', br)
        raise RuntimeError( f"No branch {varname} found in files at path {folder}. Try one from the list above." )
    return True

def as_array(folder, varname, cut, nchunks):
    if nchunks is not None:
        files = glob(os.path.join(os.path.abspath(args.inputpath), folder, "*.root"))[:nchunks]
        path = [ f"{f}:events" for f in files ]
    else:
        path = os.path.join( os.path.abspath(args.inputpath), folder, "*.root:events" )

    try: 
        # awkward array instead of numpy -> allows variable length elements
        #arr = uproot.concatenate( path+":events", expressions=varname, library="np")[varname]
        arr = uproot.concatenate( path, expressions=varname, cut=cut)[varname]
    except:
        branches = get_list_of_branches(folder)
        print( f"Branches found in files at path {folder}:" )
        for br in branches:
            print('  ', br)
        raise RuntimeError( f"Cannot process expression {varname} in files at path {folder}. Try combinations of branches from the list above." )

    # Return awkward array as a flattened ndarray
    return ak.to_numpy(ak.ravel(arr))

# Should work as-is after flattening awkward array `values`
def outlier_removal(values, threshold=7):
    mean = np.mean(values)
    sdev = np.std(values)
    pull = (values - mean)/sdev
    values = values[ np.abs(pull)<=threshold ]

    return values

def histogram_settings():
    
    hist_settings = { sample: {} for sample in cfg.samples }

    for sample in cfg.samples:

        if sample in cfg.sample_allocations["signal"]:
            hist_settings[sample]["histtype"] = "bar"

        elif sample in cfg.sample_allocations["background"]:
            hist_settings[sample]["histtype"] = "step"

    return hist_settings

def get_efficiencies():
    """
    Returns the efficiency dictionary from config.py with the efficiencies argument as key
    """
    if args.efficiencies is None:
        return { sample : 1 for sample in cfg.samples }
    else:
        if args.efficiencies not in cfg.efficiencies:
            raise RuntimeError( f"Tried passed efficiency dictionary key {args.efficiencies} which does not exist in config" )
        return cfg.efficiencies[args.efficiencies]

def get_weights(cut=None):
    """
    Returns a dictionary of weights for each sample
    Assumes a placeholder branching fraction of 1e-6 for Bs2NuNu
    """
    hist_weights = {}
    effs = get_efficiencies()
    for sample in cfg.samples:
        hist_weights[sample] = effs[sample][0] * cfg.branching_fractions[sample][0]
        if sample in cfg.sample_allocations['signal']:
            hist_weights[sample] *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample]*1e-6
    
    if cut is not None:
        cut_efficiency = efficiency_finder.get_efficiencies('custom',
                                                            further_analysis=True,
                                                            cut=cut,
                                                            raw=False,
                                                            custompath=args.inputpath,
                                                            vebose=False)
        
        hist_weights = {sample: hist_weights[sample]*cut_efficiency[sample+'_eff'] for sample in hist_weights}

    return hist_weights

def plot(varname,
         signal_bf=1e-6,
         cut=None,
         nchunks=None,
         stacked=True, 
         weight=True, 
         density=False, 
         remove_outliers=True, 
         interactive=False, 
         save=None, 
         bins=50,
         xtitle=None,
         range=None, 
         logy=False,
         total=["background"], 
         components=["signal", "background"]):
    
    """ 
    plot( varname, **opts ) will plot a variable

    Parameters
    ----------
    varname : str
        The variable to plot (must be branchname in tree). 
        If not available those that are will be listed.
    signal_bf : float, optional
        The assumed signal branching fraction to use with the weights. Default = 10^-6
    cut : str, optional
        Cut branch varname according to a (valid) UPROOT expression. Default: None
    nchunks : int or list of ints, optional
        Provide number of files to use per sample for the plot. Default: None (all files are used)
    stacked : bool, optional
        Stack histograms by their "allocation". Default: True
    weight : bool, optional
        Weight histograms by their expected branching fraction 
        multiplied by efficiency. Default: True
    density : bool, optional
        Normalise histograms so that they represent a probability
        density. Default: False
    remove_outliers : bool, optional
        Remove severe outliers from the distribution. Default: True
    interactive : bool, optional
        Show the plot interactively after its made. Default: False
    save : str, optional
        Save file for the plot. If None then no plot is saved. Default: None 
    bins : int, optional. Default = 50
    xtitle : str, optional
        Provide a custom title for the x axis. Default : `varname`, cut=`cut`.
        Use this if LaTeX complains about the cut expression.
    range : tuple or list, optional
        The lower and upper limits to use in the plot. 
        If None uses the minimum and maximum value from the samples. Default: None
    logy : bool, optional
        Use log scale for the y axis. Default: False
    total : list of str, optional
        Provide a key of config.sample_allocations to stack. Default: ['background']
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['signal', 'background']
    """
    # If nchunks is a list, use corresponding elements
    if remove_outliers:
        if isinstance(nchunks, list):
            values = { sample: outlier_removal(as_array(sample, varname, cut, nchunks[i])) for i, sample in enumerate(cfg.samples) }
        else:
            values = { sample: outlier_removal(as_array(sample, varname, cut, nchunks)) for sample in cfg.samples }
    else:
        if isinstance(nchunks, list):
            values = { sample: as_array(sample, varname, cut, nchunks[i]) for i, sample in enumerate(cfg.samples) }
        else:
            values = { sample: as_array(sample, varname, cut, nchunks) for sample in cfg.samples }

    if range is None:
        xmin = min( [ min(values[sample]) for sample in values ] )
        xmax = max( [ max(values[sample]) for sample in values ] )
    else:
        xmin = range[0]
        xmax = range[1]
    
    hist_settings = histogram_settings()
    #hist_weights = get_weights(cut=cut)
    
    fig, ax = plt.subplots()

    if weight:
        if density:
            print("----> WARNING: `density` incompatible with `weight`, setting to False")
            density = False
        effs = efficiency_finder.get_efficiencies('custom', cut=cut, raw=True, custompath=args.inputpath, verbose=False)
        n_expect = efficiency_finder.get_sample_expectations(effs, signal_bf, save=None, verbose=False, cut=cut)
        ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')

    for allocation in cfg.sample_allocations:
        if allocation not in components:
            continue
        samples = cfg.sample_allocations[allocation]
        hist_x = [ values[sample] for sample in samples ]
        #hist_w = [ np.ones_like( values[sample] ) * hist_weights[sample]/len(values[sample]) for sample in samples ]
        # TEST
        hist_w = [ np.ones_like(values[sample])*n_expect[sample+'_num']/len(values[sample]) for sample in samples ]

        hist_l = [ cfg.titles[sample] for sample in samples ]
        if not weight:
            hist_w = None

        if stacked:
            hist_opts = dict( stacked=True, histtype='stepfilled', alpha=1 )
        else:
            hist_opts = dict( stacked=False, histtype='step', lw=2 )

        if allocation=='signal':
            hist_opts['histtype'] = 'step'
            hist_opts['lw'] = 2
            hist_opts['color'] = 'cornflowerblue'
            hist_opts['hatch'] = '////'
        elif allocation=='background':
            reds = mpl.colormaps['Reds_r']
            hist_opts['color'] = reds( np.linspace(0, 1, len(samples)+2)[1:-1] )
        
        ax.hist( 
            x = hist_x,
            bins = bins,
            range = (xmin,xmax),
            density = density,
            label = hist_l,
            weights = hist_w,
            **hist_opts
        )

        if allocation in total and stacked:
            ax.hist( 
                np.concatenate( hist_x), 
                bins = bins,
                range = (xmin,xmax),
                density = density,
                label = f'Total {allocation}',
                weights = np.concatenate( hist_w ),
                histtype = 'step',
                color = 'k',
                lw = 2,
            )

    ax.legend(reverse=True)
    #if varname in cfg.variable_plot_titles:
    #    ax.set_xlabel( cfg.variable_plot_titles[varname] )
    #else:
    #    ax.set_xlabel(f'{varname}, cut={cut}')
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        ax.set_xlabel(f'{varname} (cut={cut})')

    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Counts')
    
    if logy:
        ax.set_yscale('log')

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)

def make_plots():

    outpath = f"{args.inputpath}/plots"
    if not os.path.exists( outpath ):
        os.system( f"mkdir -p {outpath}" )
    
    for stacked in [True, False]:
        suffix = "_stacked" if stacked else ""
        plot( "EVT_Thrust_Emin_e", bins=100, range=(0,50), stacked=stacked, save=f"{outpath}/EVT_Thrust_Emin_e{suffix}.pdf" ) 
        plot( "EVT_Thrust_Emax_e", bins=100, range=(0,50), stacked=stacked, save=f"{outpath}/EVT_Thrust_Emin_e{suffix}.pdf" )
        plot( "MC_Z_pz", stacked=stacked, save=f"{outpath}/MC_Z_pz{suffix}.pdf")

if __name__=="__main__":

    print( plot.__doc__ )
