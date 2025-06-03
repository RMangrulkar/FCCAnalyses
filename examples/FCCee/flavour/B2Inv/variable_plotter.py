# variable_plotter.py
# Script to interactively plot features from a specified inputpath
import os
from glob import glob
import numpy as np
import pandas as pd
import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak  # Needed if using awkward arrays
plt.style.use('fcc.mplstyle')
from cycler import cycler


import config as cfg
import efficiency_finder


from argparse import ArgumentParser
parser = ArgumentParser(description="Interactively plots features from a specified inputpath")
parser.add_argument("-i","--inputpath", default=f"{cfg.fccana_opts['outputDir']['prelim_cuts']}", help="Path to look for files in, default is the stage2 directory in config.py")
args = parser.parse_args()

def get_list_of_files(folder,inputpath = args.inputpath):
    
    path = os.path.join( inputpath, folder )
    if not os.path.exists( path ):
        raise RuntimeError( f"No such path {path}" )
    
    files = glob( os.path.join(inputpath, folder, "*.root") )
    if len(files)==0:
        raise RuntimeError( f"No root files found in {path}" )

    return files

def get_list_of_branches(folder,inputpath=args.inputpath):
    f0 = get_list_of_files(folder,inputpath)[0]
    try:
        tr = uproot.open( f0+":events" )
        return [ br.name for br in tr.branches ]
    except:
        raise RuntimeError( f"No tree called events found in file {f0}" )


def as_array(folder, varname, cut, nchunks,inputpath =args.inputpath ):
     # if root files
    if len(glob(os.path.join(os.path.abspath(inputpath), folder, "*.root")))>0:
        if nchunks is not None:
            files = glob(os.path.join(os.path.abspath(inputpath), folder, "*.root"))[:nchunks]
            path = [ f"{f}:events" for f in files ]
        else:
            path = os.path.join( os.path.abspath(inputpath), folder, "*.root:events" )

        try: 
            # awkward array instead of numpy -> allows variable length elements
            #arr = uproot.concatenate( path+":events", expressions=varname, library="np")[varname]
            arr = uproot.concatenate( path, expressions=varname, cut=cut)[varname]
        except:
            branches = get_list_of_branches(folder,inputpath)
            print( f"Branches found in files at path {folder}:" )
            for br in branches:
                print('  ', br)
            raise RuntimeError( f"Cannot process expression {varname} in files at path {folder}. Try combinations of branches from the list above." )

        # Return awkward array as a flattened ndarray
        return ak.to_numpy(ak.ravel(arr))

    # if pkl files
    elif len(glob(os.path.join(os.path.abspath(inputpath), folder, "*.pkl")))>0:
        files = glob(os.path.join(os.path.abspath(inputpath), folder, "*.pkl"))
        if nchunks is not None:
            files = files[:nchunks]

        df = pd.concat( [ pd.read_pickle(file) for file in files ], ignore_index=True )
        if cut is not None:
            try:
                df = df.query(cut)
            except:
                print("Tried to place a cut on the dataframe by couldn't. Continuing..")

        return df[varname].to_numpy()
    
    else:
        raise RuntimeError("No suitable root or pkl files found in {inputpath}/{folder}")


# Should work as-is after flattening awkward array `values`
def outlier_removal(values, threshold=7):
    mean = np.mean(values)
    sdev = np.std(values)
    pull = (values - mean)/sdev
    values = values[ np.abs(pull)<=threshold ]

    return values

# Define a custom cycle for hatching patterns - to work on #############################################################
hatch_cycle = cycler(hatch=['///', r'\\\\'])


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
            hist_settings[allocation]['hatch'] = r'////'
            hist_settings[allocation]['fill'] = False
            print('Signal colors:', hist_settings[allocation]['color'])
        elif allocation=='hadronic_background':
            hist_settings[allocation]['histtype'] = 'stepfilled'
            hist_settings[allocation]['color'] = plt.cm.Reds_r( np.linspace(0, 1, len(samples)+2)[1:-1] )
            total_color[allocation] = 'k'
            print('Background colors:', hist_settings[allocation]['color'])
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


def plot(varname,
         var1=None,
         var2=None,
         var3=None, #only used for vector sumquad
         composition=None,
         signal_bf=1e-6,#used for both signal and bkg
         cut=None,
         plot_cutline = None,
         nchunks=None,
         stacked=True, 
         weight=True, 
         density=False, 
         remove_outliers=False, 
         interactive=False, 
         save=None, 
         bins=50,
         binwidth_units=None,
         xtitle=None,
         xrange=None, 
         yrange=None,
         logy=False,
         total=["hadronic_background"], 
         components=["Bssignal", "hadronic_background"],
         verbose=True,
         inputpath = args.inputpath,
         ):
    
    """ 
    plot( varname, **opts ) will plot a variable

    Parameters
    ----------
    varname : str
        The variable to plot (must be branchname in tree). or 'composition'
        If not available those that are will be listed.
    var1 : str
        The first variable to be used in the composition (must be branchname in tree). If varname='composition'. Otherwise ignored
        If not available those that are will be listed. Default=None
    var2 : str
        The second variable to be used in the composition (must be branchname in tree). If varname='composition'. Otherwise ignored
        If not available those that are will be listed. Default=None
    var3 : str
        The third variable to be used in the composition (must be branchname in tree). If varname='composition'. Otherwise ignored
        If not available those that are will be listed. Default=None
    composition:str
        Operator to be used in composition. Currently must be '+','-','*','/',','sumquad','quaddiff','normvect','log'. For 'normvect' var1, var2, var 3 should be the x,y,z components of the depired variable where var1 is the normlaised component desired. If varname='composition'. Otherwise ignored. Default=None
    signal_bf : float, optional
        The assumed signal branching fraction to use with the weights. Default = 10^-6
    cut : str, optional
        Cut branch varname according to a (valid) UPROOT expression. Default: None
    plot_cutline: float,optional
        Plot vertical dashed line at this x value to show prelim cuts if desired. Default: None
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
    xrange : tuple or list, optional
        The lower and upper limits to use in the plot. 
        If None uses the minimum and maximum value from the samples. Default: None
    yrange : tuple or list, optional
        The lower and upper limits to use in the plot y axis. 
    logy : bool, optional
        Use log scale for the y axis. Default: False
    total : list of str, optional
        Provide a key of config.sample_allocations to stack. Default: ['hadronic_background']
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['Bssignal', 'hadronic_background'] - can also add Bd_signal
    verbose : bool, optional
        Print out some useful stuff. Default: True
    inputpath : str, optional
        For when importing function so that can run without argpasser. path to data folder to be used. Default: args.inputpath
    """
    
    decays_list = [cfg.sample_allocations[i] for i in components]
    flat_decays_list = [item for sublist in decays_list for item in sublist]

    if varname == 'composition':
        if var1 ==None or composition ==None:
            raise RuntimeError( f"For composition need to specify at least var1 and composition inputs" )

        if remove_outliers:
            raise RuntimeError( f"cannot have remove_outliers=True for composition" )
        else:
            if isinstance(nchunks, list):
                values1 = { sample: as_array(sample, var1, cut, nchunks[i],inputpath) for i, sample in enumerate(flat_decays_list) }
                if var2!=None:
                    values2 = { sample: as_array(sample, var2, cut, nchunks[i],inputpath) for i, sample in enumerate(flat_decays_list) }
                if var3!=None:
                    values3 = { sample: as_array(sample, var3, cut, nchunks[i],inputpath) for i, sample in enumerate(flat_decays_list) }
            else:
                values1 = { sample: as_array(sample, var1, cut, nchunks,inputpath) for sample in flat_decays_list }
                if var2!=None:
                    values2 = { sample: as_array(sample, var2, cut, nchunks,inputpath) for sample in flat_decays_list }
                if var3!=None:
                    values3 = { sample: as_array(sample, var3, cut, nchunks,inputpath) for sample in flat_decays_list }
       
        if composition == '+':
            values = { sample: (values1[sample] + values2[sample]) for sample in flat_decays_list }
        elif composition == '-':
            values =  { sample: (values1[sample] - values2[sample]) for sample in flat_decays_list }
        elif composition == '/':
            values =  { sample: values1[sample] /values2[sample] for sample in flat_decays_list }
        elif composition == '*':
            values =  { sample: values1[sample] * values2[sample] for sample in flat_decays_list }
        elif composition == 'sumquad':
            values =  { sample: np.sqrt(values1[sample]**2+ values2[sample]**2+ values3[sample]**2) for sample in flat_decays_list }
        elif composition == 'quaddiff':
             values =  { sample: np.sqrt(values1[sample]**2- values2[sample]**2) for sample in flat_decays_list }
        elif composition == 'normvect':
            values =  { sample: values1[sample]/(np.sqrt(values1[sample]**2+ values2[sample]**2+ values3[sample]**2)) for sample in flat_decays_list }
        elif composition == 'log':
            values =  {sample: [np.log(elem) if elem != 0 else 10 for elem in values1[sample]] for sample in flat_decays_list}#{sample: np.log(values1[sample])for sample in cfg.samples} #if values1[sample] != 0 else 0 for sample in cfg.samples}
        
        else:
            raise RuntimeError( f"No such composition {composition}" )

    else:
        
        if var1 !=None or var2 !=None or var3 !=None or composition !=None:
            print(f'var1,var2,composition inputs ignored unless varname=="composition". Currently using varname={varname}')

        # If nchunks is a list, use corresponding elements
        if remove_outliers:
            if isinstance(nchunks, list):
                values = { sample: outlier_removal(as_array(sample, varname, cut, nchunks[i],inputpath)) for i, sample in enumerate(flat_decays_list) }
            else:
                values = { sample: outlier_removal(as_array(sample, varname, cut, nchunks,inputpath)) for sample in flat_decays_list }
        else:
            if isinstance(nchunks, list):
                values = { sample: as_array(sample, varname, cut, nchunks[i],inputpath) for i, sample in enumerate(flat_decays_list) }
            else:
                values = { sample: as_array(sample, varname, cut, nchunks,inputpath) for sample in flat_decays_list }

    if xrange is None:
        xmin = min( [ min(values[sample]) for sample in values ] )
        xmax = max( [ max(values[sample]) for sample in values ] )
    else:
        xmin = xrange[0]
        xmax = xrange[1]

    # If variable is an integer make sure nbins correct
    #for most variables nbins=xmax-xmin
    #unless ChargedRP_fromPV_transformed
    if bins!=50:
        nbins = bins
    elif 'ChargedRP_fromPV_transformed' in varname:
        nbins=3
    elif '_n' in varname and '_norm' not in varname:
        print(varname)
        xmin = 0
        nbins= int(xmax - xmin)
    else:
        nbins=50

    binwidth = (xmax-xmin)/nbins

    fig, ax = plt.subplots()

    hist_settings,total_colours = histogram_settings()

    if weight:
        effs = efficiency_finder.get_efficiencies('custom', cut=cut, raw=True, custompath=inputpath, verbose=verbose,samples=flat_decays_list)
        n_expect = efficiency_finder.get_sample_expectations(effs, signal_bf, save=None, verbose=verbose, cut=cut)

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
                hist_w = [ np.ones_like(values[sample])*n_expect[sample+'_num']/len(values[sample]) for sample in samples ]
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
                np.concatenate(hist_x), 
                bins = nbins,
                range = (xmin,xmax),
                density = density,
                label = f'Total {allocation.replace("_", " ")}',
                weights = np.concatenate( hist_w ) if weight else None,
                histtype = 'step',
                color = tot_colour,#'k',
                lw = 2,
            )
    if plot_cutline is not None:
        ax.axvline(x=plot_cutline, color='k', linestyle='--', lw=1, label=f'Preselection cut')


    ax.legend(reverse=True)


    if xtitle is not None:
        ax.set_xlabel(xtitle)
    else:
        if varname=='composition':
            if composition=='sumquad':
                if cut is not None:
                    ax.set_xlabel(f"$\sqrt({var1}^2+{var2}^2+{var3}^2)$ (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"$\sqrt({var1}^2+{var2}^2+{var3}^2)$ (cut={cut}")
            elif composition=='quaddiff':
                if cut is not None:
                    ax.set_xlabel(f"$\sqrt({var1}^2-{var2}^2$ (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"$\sqrt({var1}^2-{var2}^2)$ (cut={cut}")
            elif composition=='/':
                if cut is not None:
                    ax.set_xlabel(f"${var1}/{var2})$ (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"${var1}/{var2})$ (cut={cut}")
            elif composition=='normvect':
                if cut is not None:
                    ax.set_xlabel(f"Normalised {var1} (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"Normalised {var1} (cut={cut})")
            elif composition=='log':
                if cut is not None:
                    ax.set_xlabel(f" ln({var1}) (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"ln({var1}) (cut={cut})")
            else: 
                if cut is not None:
                    ax.set_xlabel(f"{var1+ composition+var2}(cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"{var1+ composition+var2}(cut={cut})")
        else:
            if cut is not None:
                ax.set_xlabel(f"{varname} (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
            else:
                ax.set_xlabel(f"{varname} (cut={cut})")

    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel(f'Expected Counts/{binwidth}{binwidth_units}')
   
    if yrange:
        ax.set_ylim(yrange)
    
    if logy:
        ax.set_yscale('log')

    fig.tight_layout()

    if interactive:
        plt.show()

    if save is not None:
        fig.savefig(save)


if __name__=="__main__":

    print( plot.__doc__ )
