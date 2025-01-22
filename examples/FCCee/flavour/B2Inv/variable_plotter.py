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
parser.add_argument("-i","--inputpath", default=f"{cfg.fccana_opts['outputDir']['prelim_cuts']}", help="Path to look for files in, default is the stage2 directory in config.py")
# MARK FOR DELETION
# parser.add_argument("-e","--efficiencies", default=None, help="Name of efficiency dictionary key, default is None")
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

        if sample in cfg.sample_allocations["Bssignal"]:
            hist_settings[sample]["histtype"] = "bar"

        elif sample in cfg.sample_allocations["Bdsignal"]:
            hist_settings[sample]["histtype"] = "bar"

        elif sample in cfg.sample_allocations["full_background"]:
            hist_settings[sample]["histtype"] = "step"

        elif sample in cfg.sample_allocations["hadronic_background"]:
            hist_settings[sample]["histtype"] = "step"

        elif sample in cfg.sample_allocations["tau_background"]:
            hist_settings[sample]["histtype"] = "bar"

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

def get_weights(cut=None,signal_bf=1e-6,Bd_signal_bf = 1e-6):
    """
    Returns a dictionary of weights for each sample
    Assumes a placeholder branching fraction of 1e-6 for Bs2NuNu
    """
    hist_weights = {}
    effs = get_efficiencies()
    for sample in cfg.samples:
        hist_weights[sample] = effs[sample][0] * cfg.branching_fractions[sample][0]
        if sample in cfg.sample_allocations['Bssignal']:
            hist_weights[sample] *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample]*signal_bf
        if sample in cfg.sample_allocations['Bdsignal']:
            hist_weights[sample] *= 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample]*Bd_signal_bf
    
    
    if cut is not None:
        cut_efficiency = efficiency_finder.get_efficiencies('custom',
                                                            further_analysis=True,
                                                            cut=cut,
                                                            raw=False,
                                                            custompath=args.inputpath,
                                                            vebose=False)
        
        hist_weights = {sample: hist_weights[sample]*cut_efficiency[sample+'_eff'] for sample in hist_weights}

    return hist_weights

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
         signal_bf=1e-6,
         Bd_signal_bf=1e-6,
         cut=None,
         nchunks=None,
         stacked=True, 
         weight=True, 
         density=False, 
         remove_outliers=False, 
         interactive=False, 
         save=None, 
         bins=50,
         xtitle=None,
         range=None, 
         logy=False,
         total=["hadronic_background"], 
         components=["Bs_signal", "hadronic_background"],
         verbose=True):
    
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
        Operator to be used in composition. Currently must be '+','-','*','/',','sumquad','normvect','log'. For 'normvect' var1, var2, var 3 should be the x,y,z components of the depired variable where var1 is the normlaised component desired. If varname='composition'. Otherwise ignored. Default=None
    signal_bf : float, optional
        The assumed signal branching fraction to use with the weights. Default = 10^-6
    Bd_signal_bf : float, optional
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
        density. 'weight' must be False if density True. Default: False
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
        Provide a key of config.sample_allocations to stack. Default: ['hadronic_background']
    components : list of str, optional
        Distinguish the samples according to cfg.sample_allocations. Default: ['Bssignal', 'hadronic_background'] - can also add Bd_signal
    verbose : bool, optional
        Print out some useful stuff. Default: True
    """
    if density:
        if weight:
            raise RuntimeError( f"Cannot have both density and weight True. Please change one to False." )
    
    if varname == 'composition':
        if var1 ==None or composition ==None:
            raise RuntimeError( f"For composition need to specify at least var1 and composition inputs" )

        if remove_outliers:
            raise RuntimeError( f"cannot have remove_outliers=True for composition" )
        else:
            if isinstance(nchunks, list):
                values1 = { sample: as_array(sample, var1, cut, nchunks[i]) for i, sample in enumerate(cfg.samples) }
                if var2!=None:
                    values2 = { sample: as_array(sample, var2, cut, nchunks[i]) for i, sample in enumerate(cfg.samples) }
                if var3!=None:
                    values3 = { sample: as_array(sample, var3, cut, nchunks[i]) for i, sample in enumerate(cfg.samples) }
            else:
                values1 = { sample: as_array(sample, var1, cut, nchunks) for sample in cfg.samples }
                if var2!=None:
                    values2 = { sample: as_array(sample, var2, cut, nchunks) for sample in cfg.samples }
                if var3!=None:
                    values3 = { sample: as_array(sample, var3, cut, nchunks) for sample in cfg.samples }
       
        if composition == '+':
            values = { sample: (values1[sample] + values2[sample]) for sample in cfg.samples }
        elif composition == '-':
            values =  { sample: (values1[sample] - values2[sample]) for sample in cfg.samples }
        elif composition == '/':
            values =  { sample: values1[sample] /values2[sample] for sample in cfg.samples }
        elif composition == '*':
            values =  { sample: values1[sample] * values2[sample] for sample in cfg.samples }
        elif composition == 'sumquad':
            values =  { sample: np.sqrt(values1[sample]**2+ values2[sample]**2+ values3[sample]**2) for sample in cfg.samples }
        elif composition == 'normvect':
            values =  { sample: values1[sample]/(np.sqrt(values1[sample]**2+ values2[sample]**2+ values3[sample]**2)) for sample in cfg.samples }
        elif composition == 'log':
            values =  {sample: [np.log(elem) if elem != 0 else 10 for elem in values1[sample]] for sample in cfg.samples}#{sample: np.log(values1[sample])for sample in cfg.samples} #if values1[sample] != 0 else 0 for sample in cfg.samples}
        
        else:
            raise RuntimeError( f"No such composition {composition}" )

    else:
        if var1 !=None or var2 !=None or var3 !=None or composition !=None:
            print(f'var1,var2,composition inputs ignored unless varname=="composition". Currently using varname={varname}')
    
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
                # TODO: fix me please!
        # if density:
        #     print("----> WARNING: `density` incompatible with `weight`, setting to False")
        #     density = False
        effs = efficiency_finder.get_efficiencies('custom', cut=cut, raw=True, custompath=args.inputpath, verbose=verbose)
        n_expect = efficiency_finder.get_sample_expectations(effs, signal_bf, save=None, verbose=verbose, cut=cut)
        ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')

    for allocation in cfg.sample_allocations:
        if allocation not in components:
            continue
        samples = cfg.sample_allocations[allocation]
        hist_x = [ values[sample] for sample in samples ]
        if weight:
                hist_w = [ np.ones_like(values[sample])*n_expect[sample+'_num']/len(values[sample]) for sample in samples ]
        else:
                hist_w = None
        hist_l = [ cfg.titles[sample] for sample in samples ]

        if stacked:
            hist_opts = dict( stacked=True, histtype='stepfilled', alpha=1 )
        else:
            hist_opts = dict( stacked=False, histtype='step', lw=2 )

        if allocation=='Bssignal':
            hist_opts['histtype'] = 'step'
            hist_opts['lw'] = 2
            hist_opts['color'] = 'cornflowerblue'
            hist_opts['hatch'] = '////'
        elif allocation=='Bdsignal':
            hist_opts['histtype'] = 'step'
            hist_opts['lw'] = 2
            hist_opts['color'] = 'royalblue'
            hist_opts['hatch'] = '////'
        elif allocation=='combined_signal':
            hist_opts['lw'] = 2
            hist_opts['color'] = ['cornflowerblue','royalblue']
            hist_opts['alpha'] = 0.8
        elif allocation=='hadronic_background':
            reds = mpl.colormaps['Reds_r']
            hist_opts['color'] = reds( np.linspace(0, 1, len(samples)+2)[1:-1] )
        elif allocation=='tau_background':
            hist_opts['histtype'] = 'step'
            hist_opts['lw'] = 2
            hist_opts['color'] = 'mediumvioletred'
            #hist_opts['facecolor'] = 'darkmagenta'
            #hist_opts['fill'] = True
            hist_opts['hatch'] = r'\\\\'
            #hist_opts['alpha'] = 0.6
        elif allocation=='full_background':
            hist_opts['lw'] = 2
            reds = mpl.colormaps['Reds_r']
            hist_opts['color'] = reds( np.linspace(0, 1, len(samples)+2)[1:-1] )
        elif allocation=='bb_only':
            hist_opts['histtype'] = 'step'
            hist_opts['lw'] = 2
            reds = mpl.colormaps['Reds_r']
            hist_opts['alpha'] = 0.6
            hist_opts['color'] = reds(0.25)
            hist_opts['hatch'] = r'\\\\'
            hist_opts['fill'] = False




        
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
                weights = np.concatenate( hist_w ) if weight else None,
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
        if varname=='composition':
            if composition=='sumquad':
                if cut is not None:
                    ax.set_xlabel(f"$\sqrt({var1}^2+{var2}^2+{var3}^2)$ (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"$\sqrt({var1}^2+{var2}^2+{var3}^2)$ (cut={cut}")
            if composition=='/':
                if cut is not None:
                    ax.set_xlabel(f"${var1}/{var2})$ (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"${var1}/{var2})$ (cut={cut}")
            if composition=='normvect':
                if cut is not None:
                    ax.set_xlabel(f"Normalised {var1} (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"Normalised {var1} (cut={cut})")
            if composition=='log':
                if cut is not None:
                    ax.set_xlabel(f" ln({var1}) (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"ln({var1}) (cut={cut})")
            else: 
                if cut is not None:
                    ax.set_xlabel(f"{var1+ composition+var2}(cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
                else:
                    ax.set_xlabel(f"{var1+ composition+var2}(cut={cut}")
        else:
            if cut is not None:
                ax.set_xlabel(f"{varname} (cut={replace_all(replace_all(replace_all(cut,'>','$>$'),'<','$<$'),'&',',')})")
            else:
                ax.set_xlabel(f"{varname} (cut={cut})")

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


### Lists for plots for Jan 2025 - can be removed later#####
variable_list_BDT2_options = [ 'EVT_unitThrust_x',
 'EVT_unitThrust_y',
 'EVT_unitThrust_z',
 'EVT_Thrust_mag',
 'EVT_hemisEmin_maxpChargedRP_px',
 'EVT_hemisEmin_maxpChargedRP_py',
 'EVT_hemisEmin_maxpChargedRP_pz', #from here and above basically no correlation, dont think will include
 'EVT_hemisEmin_nDV',
 'EVT_sum_Rec_vtx_ntracks_exclPV',
 'EVT_hemisEmin_sum_Rec_vtx_ntracks_exclPV',
 'EVT_hemisEmax_sum_Rec_vtx_ntracks_exclPV',
 'EVT_hemisEmax_nDV', #nb. was in twice
 'Rec_track_absd0_max_hemisEmin',
 'Rec_track_absd0_ave_hemisEmin',
 'Rec_track_absd0chi2_max_hemisEmin',
 'Rec_track_absd0chi2_ave_hemisEmin',
 'Rec_track_absz0_max_hemisEmin',
 'Rec_track_absz0_ave_hemisEmin',
 'Rec_track_absz0chi2_max_hemisEmin',
 'Rec_track_absz0chi2_ave_hemisEmin',
 'Rec_track_absz0_min_hemisEmin',
 'Rec_track_absz0chi2_min_hemisEmin',
 'EVT_hemisEmin_maxpChargedRP_p',
 'EVT_hemisEmin_maxpChargedRP_fromPV',
 'EVT_hemisEmax_maxpChargedRP_p',
 'EVT_hemisEmax_maxpChargedRP_fromPV',
 'Rec_vtx_ntracks_max_hemisEmin',
 'Rec_vtx_ntracks_max_hemisEmax',
 'Rec_thrustCosTheta_max_hemisEmin', #nb. was in twice
 'Rec_thrustCosTheta_ave_hemisEmin',#nb. was in twice
 'Rec_thrustCosTheta_max_hemisEmax',#nb. was in twice
 'Rec_thrustCosTheta_ave_hemisEmax',#nb. was in twice
 'Rec_vtx_thrustCosTheta_max_hemisEmin', 
 'Rec_vtx_thrustCosTheta_ave_hemisEmin', 
 'Rec_vtx_thrustCosTheta_max_hemisEmax', 
 'Rec_vtx_thrustCosTheta_ave_hemisEmax', 
 'Rec_vtx_d2PV_max_hemisEmin',
 'EVT_Thrust_deltaE',
 'EVT_hemisEmin_Emiss',
 'EVT_hemisEmax_Emiss',
 'EVT_e',
 'Rec_thrustCosTheta_min_hemisEmin',
 'Rec_thrustCosTheta_min_hemisEmax',
 'PV_Rec_vtx_m',
 'Rec_PV_ntracks',
 'Rec_track_n',
 'Rec_track_absd0_max_hemisEmax',
 'Rec_track_absd0_ave_hemisEmax',
 'Rec_track_absd0chi2_max_hemisEmax',
 'Rec_track_absd0chi2_ave_hemisEmax',
 'Rec_track_absz0_max_hemisEmax',
 'Rec_track_absz0_ave_hemisEmax',
 'Rec_track_absz0chi2_max_hemisEmax',
 'Rec_track_absz0chi2_ave_hemisEmax',
 'Rec_vtx_d2PV_max_hemisEmax',
 'Rec_vtx_d2PV_ave_hemisEmax',
 'Rec_vtx_d2PV_min_hemisEmax',
 'Rec_track_absd0_min_hemisEmin',
 'Rec_track_absd0chi2_min_hemisEmin',
 'Rec_track_absz0chi2_min_hemisEmax',
 'Rec_track_absd0chi2_min_hemisEmax',
 'Rec_track_absd0_min_hemisEmax',
 'Rec_track_absz0_min_hemisEmax',
 'EVT_sum_Rec_px',
 'EVT_sum_Rec_py',
 'EVT_sum_Rec_pz',
 'EVT_p',]

 #can now do command line for loop : for variable in variable_list_BDT2_options: plot(variable, save=f'plots/Data_with_incorrect_BSC_Dec2024/correlation_plot_variables/{variable}.pdf',weight=True,components=['hadronic_background','Bssignal'],total=["hadronic_background"], nchunks=12,signal_bf=1e-3)
 #Would be nice to update variable plotted so doesnt need to use command line. ie. make argpass only  https://stackoverflow.com/questions/44283780/importing-a-python-script-module-that-uses-argparse-into-another-python-script

 

ranges = [(-1,1),
          (-1,1),
          (-1,1),
          (0,1),
          (-15,15),
          (-40,40),
          (-40,40),
          (0,6), #bins=6
          (0,20),#bins=20
          (0,12),#bins=12
          (0,16),#bins=16
          (0,7),#bins=7 
(0,200),
(0,30),
(0,2000),
(0,300),
(0,300),
(0,60),
(0,2000),
(0,600),
(0,1.5),
(0,100),
(0,40),
(-999,501),#bins=3
(0,45),
(-999,501),#bins=3
(0,10),#bins=10
(0,12),#bins=12
(0,1),
(0,1),
(-1,0),
(-1,0),
(0,1),
(0,1), 
(-1,0),
(-1,0),
(0,20),
(0,50),
(0,46),
(0,46),
(0,90),
(0,1),
(-1,0),
(0,80),
(0,40),#bins=40
(0,45),#bins=45
(0,200),
(0,30),
(0,2000),
(0,300),
(0,300),
(0,60),
(0,3000),
(0,600),
(0,20),
(0,20),
(0,20),
(0,0.1),
(0,5),
(0,100),
(0,5),
(0,0.1),
(0,1.5),
(-50,50),
(-50,50),
(-50,50),
(0,50),]

nbins = [50,
          50,
          50,
          50,
          50,
          50,
          50,
          6,
          20,
          12,
          16,
          7,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
3,
50,
3,
10,
12,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
40,
45,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,
50,]


#for i in range(len(nbins)): plot(variable_list_BDT2_options[i],range=ranges[i],bins=nbins[i], save=f'plots/Data_with_correct_BSC_Jan2025/{variable_list_BDT2_options[i]}.pdf',weight=True,components=['hadronic_background','Bssignal'],total=["hadronic_background"], nchunks=12,signal_bf=1e-1)