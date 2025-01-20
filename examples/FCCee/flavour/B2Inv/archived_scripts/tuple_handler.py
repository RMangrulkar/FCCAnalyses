# tuple_handler.py
# WIP -- NEEDS TO BE INTEGRATED WITH THE WORKFLOW
# Currently logically and stylistically outdated
# Contains useful functions related to tupling
# Classes:   TupleHandler
# Functions:
#           - get_vars_from_yaml
#           - load_data

import os
import subprocess
import time
import uproot

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from glob import glob
from yaml import safe_load

# Requires config file in the working directory
import config as cfg

# Set global options
plt.style.use('fcc.mplstyle')


##################################################################
#               PLANNED FEATURES
# Analysis
#   Saving from all/selected background files
#   Save EVT level and Particle level data
#   Cut using EVT level and Particle level cuts
#   Aliases for complicated expressions
#   Easy handling of expressions involving EVT and Particle level expressions
#       For example angle between particle momentum and thrust axis
#
#
# Plotting
#   Plot data from analysis
#   Plot data directly from file
#   Save/show plots
#   Custom range for binning
#   Custom x and y limits
#   Automatic title
#   Distinguish signal from background when plotting
##################################################################
class TupleHandler():
    """
    Class that handles stage0 data, including cutting and plotting
    """

    def __init__(self,
                 inputpath=None,
                 outputpath=None,
                 init_status=True):

        # Check if inputpath is a valid directory, if None use default stage0 directory
        self.set_inputpath(inputpath)

        # Save outputs in self.outputpath, if directory does not exist try to create it
        self.set_outputpath(outputpath)

        self.config_essential_attributes = ['samples',
                                            'stage0dir_in',
                                            'stage0dir_out',
                                            'sample_allocations',
                                            'titles',
                                            'prod_frac',
                                            'branching_fractions',
                                            'efficiencies',
                                            'samples_poststage0']

        # Instantiating methods
        self.configure()

        if init_status:
            self.print_status()

    def set_inputpath(self, inputpath):
        ''' Define directory in which config.samples are located. If None use default'''
        if inputpath is None:
            if not os.path.exists(cfg.stage0dir_in):
                raise RuntimeError(f"Default input path {inputpath} does not exist. Please ensure that the default directory is set correctly or use the optional argument to set it.")
            self.inputpath = cfg.stage0dir_in
        else:
            if not os.path.exists(inputpath):
                raise RuntimeError(f"No such path: {inputpath}")
            self.inputpath = inputpath

    def set_outputpath(self, outputpath):
        ''' Define directory to save files to. If None use default.'''
        if outputpath is None:
            if not os.path.exists(cfg.stage0dir_out):
                try:
                    subprocess.run(["mkdir", cfg.stage0dir_out], check=True)
                except subprocess.CalledProcessError:
                    print(f"Default path {outputpath} does not exist or could not be created. Please ensure that the default directory is set correctly or use the optional argument to set it.")

            self.outputpath = cfg.stage0dir_out

        else:
            if not os.path.exists(outputpath):
                try:
                    subprocess.run(["mkdir", outputpath], check=True)
                except subprocess.CalledProcessError:
                    print(f"Cannot create directory {outputpath}")

            self.outputpath = outputpath

    def configure(self):
        """
        Import config file

        Parameters
        ----------
        config_file : str, optional
            Full path to the config_file.py to import
        """
        if not set(self.config_essential_attributes).issubset(cfg.__dict__.keys()):
            raise AttributeError(f"Config file does not have all essential attributes: {self.config_essential_attributes}")

        self.update_file_dict(cfg.samples)
        self.clear_hist_data()

    def update_file_dict(self, samples):
        """
        Return a dictionary with filepaths as a list and as glob syntax
        path_dict for functions such as uproot.iterate, file_dict for functions such as uproot.concatenate
        """
        self.file_dict = {}
        self.path_dict = {}
        for sample in samples:
            self.path_dict[sample], self.file_dict[sample] = self.get_files(sample)

    def get_files(self, folder):
        """
        Search for .root files in self.inputpath/folder

        Parameters
        ----------
        folder : str, required
            Name of the folder(s) in which to search for the .root files

        Returns
        -------
        path : str
            Path to .root files in `glob` syntax (i.e. as self.inputpath/folder/*.root)
        files : list
            List of .root filenames inside `folder`
        """
        path = os.path.join(self.inputpath, folder)
        if not os.path.exists(path):
            raise RuntimeError(f"No such path: {path}")
        path = os.path.join(path, "*.root")
        files = glob(path)
        if len(files) == 0:
            raise RuntimeError(f"No root files in {path}")
        return path, files

    def print_status(self):
        ''' Print values of class attributes required for start-up'''
        time.sleep(1)
        print("-------------------------------------------------")
        print("               TUPLE HANDLER CLASS               ")
        print("-------------------------------------------------")
        time.sleep(1)
        print(f"\n\nUsing {os.path.abspath(cfg.__file__)} as config...\n")
        print(f"Samples from {self.inputpath}...\n")
        print(f"Saving to {self.outputpath}...\n\n")
        time.sleep(1)

        if self.file_dict is not None:
            print(f"File paths saved")
        if self.hist_data is not None:
            print(f"Current histogram data: {self.hist_data}")

        print("-------------------------------------------------")

    def clear_hist_data(self):
        self.hist_data = {}

    def get_branch_data(self,
                        branches,
                        branchtype,
                        hemisphere,
                        aliases=None,
                        cut_EVTlevel=None,
                        cut_MClevel=None,
                        cut_REClevel=None,
                        sample_select=None,
                        output_type='rt',
                        show_status=True,
                        custom_tree_name=None,
                        custom_out_name=None,
                        overwrite=False):
        """
        Store data from .root files to analyse or plot

        Parameters
        ----------
        branches : str
            Branches or equivalent expression to store
        branchtype : str
            The type of branches. Acceptable entries are 'per-event' or 'per-particle'
        hemisphere : str, optional
            Selection of min/max energy hemisphere. Applies only if branchtype is 'per-particle'
            Options are 'both', 'max' or 'min'. Default = 'both'
        cut_EVTlevel : str, optional
            Cut branches at event level (row). Must be acceptable to uproot's `cut` parameter.
            Applies to both 'per-event' and 'per-particle' branch types. Default = None
        cut_PARTlevel : str, optional
            Cut branches at particle level (column). Must be acceptable to uproot's `cut` parameter.
            Applies only to the 'per-particle' branch types. Default = None
        sample_select : list, optional
            List of indices of cfg.samples to use.  If None uses all of cfg.samples.
            Preferred behaviour is to use an appropriate config and keep this None. Default = None
        output_type : str, optional
            'hist' : flattens data into 1d ndarray for plotting. Needs a single entry in branches
            ---EVENTUALLY---
            'ak'   : returns data as a highlevel awkard Array
            'pd'   : returns data as a pandas DataFrame
            'rt'   : saves cut branches as a .root file in the current directory
        """

        ##############################
        ## EXCEPTION HANDLING
        ##############################
        if (not (branches or branchtype)) or (branchtype not in ['per-event', 'per-mc', 'per-rec']):
            raise ValueError('One of branch or branchtype was empty, or branchtype invalid.')

        if hemisphere[0:4] not in ['both', 'max ', 'min ']:
            raise ValueError('`hemisphere` must be both, max or min followed by variable to use for angle')

        if (not branchtype) and (cut_EVTlevel or cut_MClevel or cut_REClevel):
            raise ValueError(f'branchtype {branchtype} incompatible with non-trivial cuts')
        elif branchtype == 'per-event' and ((cut_MClevel is not None) or (cut_REClevel is not None)):
            raise ValueError('event level branches but particle level cuts provided')
        elif branchtype == 'per-mc' and cut_REClevel is not None:
            raise ValueError('MC variable branches but Reconstructed type cuts provided')
        elif branchtype == 'per-rec' and cut_MClevel is not None:
            raise ValueError('Reconstructed variable branches but MC type cuts provided')

        if sample_select is not None:
            samples = [cfg.samples[i] for i in sample_select]
        else:
            samples = cfg.samples

        if (len(branches) > 1) and (output_type == 'hist'):
            raise ValueError('Cannot create a histogram with more than one branch')

        if (custom_out_name is not None) and (len(custom_out_name) != len(samples)):
            raise ValueError(f'{custom_out_name} cannot be broadcast to {samples}: incorrect length')
        ##############################
        ## END OF EXCEPTION HANDLING
        ##############################

        # Update filenames to use
        self.update_file_dict(samples)

        ##############################
        ## GENERATE CUT EXPRESSION
        ##############################
        # Basic expression without direction information
        cut_expr = None
        if (cut_EVTlevel is not None) and (cut_MClevel is not None):
            cut_expr = f"({cut_EVTlevel}) & ({cut_MClevel})"
        elif (cut_EVTlevel is not None) and (cut_REClevel is not None):
            cut_expr = f"({cut_EVTlevel}) & ({cut_REClevel})"
        elif cut_EVTlevel is not None:
            cut_expr = cut_EVTlevel
        elif cut_MClevel is not None:
            cut_expr = cut_MClevel
        elif cut_REClevel is not None:
            cut_expr = cut_REClevel

        # Append hemisphere condition
        if (hemisphere != 'both') and (branchtype not in ['per-mc', 'per-rec']):
            raise ValueError(f"Hemisphere value {hemisphere} needs `per-particle` branchtype")
        elif hemisphere[0:4] == 'min ':
            var4theta = hemisphere[4:-1]
            poscosTheta = f"(EVT_Thrust_x*{var4theta}x + EVT_Thrust_y*{var4theta}y + EVT_Thrust_z*{var4theta}z) / ((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5 * ({var4theta}x**2 + {var4theta}y**2+{var4theta}z**2)) > 0"

            cut_expr = f"({cut_expr}) & ({poscosTheta})"
        elif hemisphere[0:4] == 'max ':
            var4theta = hemisphere[4:-1]
            negcosTheta = f"(EVT_Thrust_x*{var4theta}x + EVT_Thrust_y*{var4theta}y + EVT_Thrust_z*{var4theta}z) / ((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5 * ({var4theta}x**2 + {var4theta}y**2+{var4theta}z**2)) < 0"

            cut_expr = f"({cut_expr}) & ({negcosTheta})"
        elif hemisphere == 'both':
            pass
        else:
            raise ValueError("Incompatible options for `hemisphere` and `branchtype`")

        ##############################
        ## SAVE TO ROOT FILE AT GIVEN OUTPUTPATH
        ## Issues -------> Remove [] from dataset instead of empty entry in tree
        ##############################
        if output_type == 'rt':
            tree_name = custom_tree_name if custom_tree_name is not None else 'events'
            custom_out_name = custom_out_name if custom_out_name is not None else {sample: sample for sample in samples}
            if show_status:
                time.sleep(0.5)
                print("----------------------------\n")
                print("Preliminary checks passed...")
                print(f"Using name {tree_name} for root TTree...")
                print("\n----------------------------")

            for sample in samples:

                if os.path.exists(os.path.join(self.outputpath, sample+'.root')) and overwrite is False:
                    if show_status:
                        print(f"{os.path.exists(os.path.join(self.outputpath, sample+'.root'))} already exists, skipping...")
                        continue

                with uproot.recreate(os.path.join(self.outputpath, sample+".root")) as outfile:
                    df = ak.Array([])
                    for file in uproot.iterate(self.path_dict[sample]+':events', expressions=branches, aliases=aliases, cut=cut_expr):
                        df = ak.concatenate([df, file])

                    outfile[tree_name] = ak.zip({branch: df[branch] for branch in branches}, depth_limit = 1)
                    if show_status:
                        print(f"Branches from {sample} stored in {os.path.join(self.outputpath, sample+'.root')}")

        ##############################
        ## WIP ----> SAVE TO SELF.HIST_DATA FOR PLOTTING
        ##############################
        if output_type == 'hist':

            if show_status:
                time.sleep(0.5)
                print("----------------------------\n")
                print("Preliminary checks passed...")
                print(f"Saving {branches} for plotting...")
                print("\n----------------------------")

            for sample in samples:
                self.hist_data[sample] = np.array([])
                with uproot.iterate(self.file_dict[sample]+':events', expressions=branches, cut=cut_expr, aliases=aliases) as tree:
                    self.hist_data[sample] = np.append(self.hist_data[sample],
                                                       ak.ravel(tree[branches]).to_numpy())

    ##############################
    ## WIP ----> PLOT DATA FROM SELF.HIST_DATA
    ##############################
    def plot_histogram(self,
                       external_data=None,
                       style='all-backgrounds',
                       stacked=True,
                       density=True,
                       remove_outliers=True,
                       interactive=False,
                       save=None,
                       total=['background'],
                       components=['signal', 'background'],
                       range=None,
                       clear_saved_data=False):
        """
        Plot histogram using self.hist_data generated by self.get_branch_data

        Parameters
        ----------
        """
        if total is None:
            total = self.total

        if components is None:
            components = self.components
        if remove_outliers:
            self.hist_data = {sample: self.remove_outliers(self.hist_data[sample]) for sample in self.hist_data}

        if range is None:
            xmin = min([min(self.hist_data[sample]) for sample in self.hist_data])
            xmax = max([max(self.hist_data[sample]) for sample in self.hist_data])
        else:
            xmin = range[0]
            xmax = range[1]

        #
        # PLOTTING HERE
        #

        if interactive:
            plt.show()

        if save is not None:
            plt.savefig(save)

        if clear_saved_data:
            self.clear_hist_data()

    ################## MC MATCHING
    #
    # Issues ---> eager writing of data (everything stored in an awkward Array before writing)
    #        ---> have to zip different variables together before passing to uproot
    #                 as a result uproot makes two counters nMC and nRec which are clones
    def get_branches_from_recp(self,
                               use_defaults,
                               EVTbranches=None,
                               MCbranches=None,
                               RECbranches=None,
                               EVTaliases=None,
                               MCaliases=None,
                               RECaliases=None,
                               save_customid=True,
                               sample_select=None,
                               show_status=True,
                               tree_name=None,
                               outpath=None,
                               out_name=None,
                               overwrite=False):
        """
        Method to save corresponding MC data of Reconstructed particles using `Rec_MC_index`

        PARAMETERS
        ----------
        use_defaults : bool
            Save the default variables.
        EVTbranches : list, optional
            List of names of event level branches and/or expressions. Only used if use_defaults is False. Default = None
        MCbranches : list, optional
            List of names of particle level branches and/or expressions. These branches specifically are the ones which
            Rec_MC_index acts on. Only used if use_defaults is False. Default = None
        RECbranches : list, optional
            List of names of particle level branches and/or expressions. These should contain reconstructed data. 
            Only used if use_defaults is False. Default = None
        EVTaliases : dict, optional
        MCaliases : dict, optional
        RECaliases : dict, optional
        save_particletype : bool, optional
            Add an additional branch with the type of the reconstructed particle. Default = True.
            The IDs correspond to:
            Photon                 = 10
            Charged leptons ~ 20
                - e+e-             = 21
                - mu+mu-           = 22
                - tau+tau-         = 23
            Neutral hadrons ~ 30
                - Neutral mesons   = 34
                - Neutral baryons  = 35
            Charged hadrons ~ 40
                - Charged mesons   = 44
                - Charged baryons  = 45
            ***** This is added because Rec_type loses information about the sign of the PID
        sample_select : list, optional
            List of indices of cfg.samples to use. Default = None (every element of cfg.samples)
        show_status : bool, optional
            Print progress of the process. Default = True
        tree_name : str, optional
            Name of the .root tree. If None uses 'events'. Default = None
        outpath : str, optional
            Sets directory to save the output files and updates self.outputpath. If None uses self.outputpath without updating. Default = None
        out_name : dict, optional
            Dictionary of the form {sample : name} where sample is the element of cfg.samples used and name is the name of the output .root file. Default = None
        overwrite : bool, optional
            If True, overwrites any existing .root files at the target destination. Default = False
        """
        #################################################
        #
        # Initialisation and error handling
        #
        #################################################
        # Update directory where files are saved
        self.set_outputpath(outpath)

        # Initialise custom sample files
        if sample_select is not None:
            samples = [ cfg.samples[i] for i in sample_select ]
        else:
            samples = cfg.samples

        self.update_file_dict(cfg.samples)
        
        # Update names of root files
        if out_name is None:
            out_name = {cfg.samples[0]: 'Bs2NuNu_fromrecp',
                        cfg.samples[1]: 'bb_fromrecp',
                        cfg.samples[2]: 'cc_fromrecp',
                        cfg.samples[3]: 'ss_fromrecp',
                        cfg.samples[4]: 'ud_fromrecp'}
        
        # Update tree name used by root files
        tree_name = tree_name if tree_name is not None else 'events'
    
        # Print summary
        if show_status:
            time.sleep(0.5)
            print("-------------------------------------------------\n")
            print("Saving MC data from Reconstructed indices")
            print("Preliminary checks passed...")
            time.sleep(0.5)
            print(f"Using {samples}")
            time.sleep(0.5)
            if use_defaults:
                print("Saving default variables")
            else:
                print(f"Saving {EVTbranches}...")
                print(f"Saving {MCbranches}...")
                print(f"Saving {RECbranches}...")
            print(f"Saving custom particle id: {save_customid}")
            time.sleep(0.5)
            print(f"Saving to {self.outputpath}")
            print(f"Using tree name <{tree_name}> for root TTree")
            if overwrite:
                print("WARNING: overwrite parameter is set to True, existing files at the target directory with the same name will be overwritten")
            print("\n-------------------------------------------------")
        
        #################################################
        #
        # Save default variables
        #
        #################################################
        if use_defaults:
            EVTbranches = ['Thrust_Emax',
                           'Thrust_Emin',
                           'Thrust_delE',
                           
                           'Thrust_x',
                           'Thrust_y',
                           'Thrust_z',
                           'Thrust_len',
                           'Thrust_xerr',
                           'Thrust_yerr',
                           'Thrust_zerr',
                           'Thrust_cosrel2beam',

                           'MC_n',
                           'Rec_n']

            MCbranches = ['MC_genstatus',
                          'MC_PDG',
                          'MC_p',
                          'MC_pt',
                          'MC_px',
                          'MC_py',
                          'MC_pz',
                          'MC_e',
                          'MC_m',

                          'MC_etarel2beam',
                          'MC_phirel2beam',
                          'MC_cosrel2beam',
                          'MC_cosrel2thrust']

            RECbranches = ['Rec_type',
                           'Rec_p',
                           'Rec_pt',
                           'Rec_px',
                           'Rec_py',
                           'Rec_pz',
                           'Rec_e',
                           'Rec_m',

                           'Rec_cosrel2beam',
                           'Rec_etarel2beam',
                           'Rec_phirel2beam',
                           'Rec_cosrel2thrust']

            EVTaliases= {'Thrust_Emax' : 'EVT_Thrust_Emax_e',
                         'Thrust_Emin' : 'EVT_Thrust_Emin_e',
                         'Thrust_delE' : 'EVT_Thrust_Emax_e - EVT_Thrust_Emin_e',
                         
                         'Thrust_x'    : 'EVT_Thrust_x',
                         'Thrust_y'    : 'EVT_Thrust_y',
                         'Thrust_z'    : 'EVT_Thrust_z',
                         'Thrust_len'  : '(EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5',

                         'Thrust_xerr' : 'EVT_Thrust_xerr',
                         'Thrust_yerr' : 'EVT_Thrust_yerr',
                         'Thrust_zerr' : 'EVT_Thrust_zerr',
                         
                         'Thrust_cosrel2beam' : '(EVT_Thrust_z)/((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'}

            MCaliases = {'MC_etarel2beam' : 'MC_eta',
                         'MC_phirel2beam' : 'MC_phi',
                         'MC_cosrel2beam' : '(MC_pz)/(MC_p)',
                         'MC_cosrel2thrust' : '(MC_px*EVT_Thrust_x + MC_py*EVT_Thrust_y + MC_pz*EVT_Thrust_z)/(MC_p*(EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'}
            # Future -> error in cosrel2thrust

            RECaliases = {'Rec_etarel2beam' : 'Rec_eta',
                          'Rec_phirel2beam' : 'Rec_phi',
                          'Rec_cosrel2beam' : '(Rec_pz)/(Rec_p)',
                          
                          'Rec_cosrel2thrust' : '(Rec_px*EVT_Thrust_x + Rec_py*EVT_Thrust_y + Rec_pz*EVT_Thrust_z)/(Rec_p*(EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'}
            # Future -> error in cosrel2thrust
        elif (EVTbranches is None) and (MCbranches is None) and (RECbranches is None):
            raise ValueError('Need at least one branch if use_defaults is False')

        # To allow customid saving MC_PDG must be included
        if save_customid and ('MC_PDG' not in MCbranches):
            raise ValueError("save_customid needs an MCbranch `MC_PDG`")

        # Combine lists of expressions
        all_expressions = []
        if EVTbranches is not None:
            all_expressions += EVTbranches
        if MCbranches is not None:
            all_expressions += MCbranches
        if RECbranches is not None:
            all_expressions += RECbranches
        
        all_expressions += ['Rec_MC_index']
        
        all_aliases = {}
        if EVTaliases is not None:
            all_aliases.update({i: EVTaliases[i] for i in EVTaliases.keys()})
        if MCaliases is not None:
            all_aliases.update({i: MCaliases[i] for i in MCaliases.keys()})
        if RECaliases is not None:
            all_aliases.update({i: RECaliases[i] for i in RECaliases.keys()})

        #################################################
        #
        # Save data to files
        #
        #################################################
        for sample in samples:
            if (overwrite is False) and os.path.exists(os.path.join(self.outputpath, out_name[sample]+'.root')):
                if show_status:
                   print(f"{os.path.join(self.outputpath, out_name[sample]+'.root')} already exists, skipping...")
                continue
            with uproot.recreate(os.path.join(self.outputpath, out_name[sample]+'.root')) as outfile:
                df = []
                # Pass all expressions to uproot
                for events in uproot.iterate(self.path_dict[sample], expressions=all_expressions, aliases=all_aliases):
                    if show_status:
                        print(f"Opened {events}, processing...")
                    for MCbranch in MCbranches:
                        events[MCbranch] = events[MCbranch][events['Rec_MC_index']]

                    if save_customid:
                        events['Rec_customid'] = self.get_customid(events['MC_PDG'])
                    
                    df = ak.concatenate([df, events])
                
                print(f"{sample} data finished processing, now saving...")
                
                # Zip all MCbranches together and remove the leading MC_ so uproot can add it back later 
                per_mc = ak.zip({ branch[3:] : df[branch] for branch in MCbranches })
                # Zip all RECbranches together and remove the leading Rec_ so uproot can add it back later
                per_rec = ak.zip({ branch[4:] : df[branch] for branch in RECbranches+['Rec_customid'] })
                # Zip all EVTbranches together and DO NOT remove any leading characters as uproot will not make a counter for these branches
                per_event = ak.zip({ branch : df[branch] for branch in EVTbranches })

                outfile[tree_name] = { "": per_event, "MC": per_mc, "Rec": per_rec }
                if show_status:
                    print(f"Data from {sample} saved to {os.path.join(self.outputpath, out_name[sample]+'.root')}")
                    print("-------------------------------------------------")

    def get_customid(self, PIDarray):
        """
        Return custom id from PDG

        Scheme:
            Neutrals (photons)     = 10
            Charged leptons ~ 20
                - e+e-     = 21
                - mu+mu-   = 22
                - tau+tau- = 23
            Neutral hadrons ~ 30
                - Neutral mesons   = 34
                - Neutral baryons  = 35
            Charged hadrons ~ 40
                - Charged mesons   = 44
                - Charged baryons  = 45
        """
        lengths = ak.num(PIDarray, axis=-1)
        temp = ak.flatten(abs(PIDarray), axis=-1)
        condition_list = [
                # Leptons
                temp == 11,
                temp == 13,
                temp == 15,

                # Mesons
                (temp // 1000 == 0) & (temp // 100 != 0) & ( ((temp//100 % 10)+(temp // 10 % 10)) % 2 == 0 ),
                (temp // 1000 == 0) & (temp // 100 != 0) & ( ((temp//100 % 10)+(temp // 10 % 10)) % 2 != 0 ),

                # Baryons
                (temp // 1000 != 0) & ( ((temp // 1000 % 10)+(temp // 100 % 10)+(temp // 10 % 10)) % 2 == 0 ),
                (temp // 1000 != 0) & ( ((temp // 1000 % 10)+(temp // 100 % 10)+(temp // 10 % 10)) % 2 != 0 )
            ]

        choice_list = [21, 22, 23, 34, 44, 35, 45]

        return ak.unflatten(np.select(condition_list, choice_list, 10), lengths, axis=-1)


def get_vars_from_yaml(yamlpath, key):
    """
    Return the list of variables from the yaml file

    Parameters
    ----------
    yamlpath : str
        Path to the yaml file
    key : str
        Key to extract from the yaml file

    Returns
    -------
    custom
        Value of yamlpath[key]
    """
    with open(yamlpath) as stream:
        yamlFile = safe_load(stream)
        return yamlFile[key]

if __name__=="__main__":

    handler = TupleHandler()
    #handler.get_branches_from_recp(use_defaults=True, overwrite=True)
