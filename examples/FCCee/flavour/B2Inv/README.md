# Prospects of searches for $B_{(s)}^0 \rightarrow$ Invisible decays at FCC-ee

## Contact
Ella Wood: [ella.wood@cern.ch](malito:ella.wood@cern.ch)  
Ritwik Mangrulkar: [ritwik.mangrulkar@cern.ch](malito:ritwik.mangrulkar@cern.ch)

## Pre-print submitted 6th August 2025
- Prospects of searches for $B_{(s)}^0 \rightarrow$ Invisible decays at FCC-ee
: [arXiv:2508.04471](https://arxiv.org/abs/2508.04471)

## Talks
- [Prospects for searches of invisible B-meson decays at FCC-ee](https://indico.cern.ch/event/1469952/contributions/6188405/attachments/2951727/5188847/2024_10_21_B2Inv_Presentation.pdf) Riwik Mangrulkar, first presentation of initial studies at Physics Performance meeting, 21st October 2024.
- [Prospects for searches of invisible $B$-meson decays at FCC-ee](https://indico.cern.ch/event/1557170/contributions/6557371/attachments/3086782/5465338/B(s)2Inv_FCC-ee_flvours_pres.pdf) Ella Wood, approval presentation to Flavours at FCC-ee group, 16th June 2025.
- [Invisible B-decays and related modes at FCC-ee](https://indico.cern.ch/event/1588013/contributions/6762476/attachments/3177116/5650859/Invisible_B_decays_FCCee.pdf) Paula Álvarez Cartelle, presentation at FCC Flavours Workshop, 19th November 2025.


## Workflow overview
1. Produce ntuples using `process_tuples.py` script in FCCAnalyses framework, specifying runmode from config options. Various configurations exist as detailed in  `process_tuples.py`. These are then used for further analysis.

2. Create dataframe to train multiclass BDT using `df_makers/bdt_lh_create_df.py` then train with XGBoost (and optimise hyperparameters with optuna) using `bdt_lh_training.py`.

3. Turn full data in root format into pandas dataframes using `df_makers/pickle_full_data.py`, such that it can then have the multiclass BDT applied.

4. Apply multiclass BDT with `df_makers/apply_bdt_lh.py`.

5. Create interpolated maps of number of events remaining after various BDT cuts using `create_new_interps.py`.

6. Run cut optimisation (using $S/\sqrt{S+B}$ figure of merit) and calculate various sensitivity estimates using `plotters/make_paper_plots/N_interp_plot.py` which also plots them.


Note, scripts in 5. and 6. rely on tools from `bdt_lh_cut_opt_significance.py` and `bdt_lh_cut_opt_significance_exclusive_backgrounds.py`.


## Detailed reference

### Analyzers

Some dedicated C++ functions are added to `FCCAnalyses/analyzers/dataframe/FCCAnalyses` for this analysis. Right now they are mostly just dumped into `myUtils` (the odd one in some other namespace) in labelled code block. In the future they could be moved to other namespaces depending on functionality/similarity with existing functions.

### `config.py`

This contains almost all of the configuration options, most importantly:
- The attributes used by `fccanalysis run` (see [analysis script](https://hep-fcc.github.io/FCCAnalyses/man/latest/fccanalysis-script.html#ATTRIBUTES))
- Path to every important location.
  + Path to the YAML file containing sets of variables saved for different run configurations and stages of the analysis.
  + Output directory of `fccanalysis run`.
- Dicts of important keys/values:
  + Beam spot conditions for various fastsim samples (winter2023 vs. spring 2021). Note here we use winter2023 MC samples. 
  + BDT options including hyperparameters and names of varible lists from YAML file used in training. 
  + Branching fractions and other constants, mostly measured by LEP and taken from HFLV/PDG.
  + Sample names, their allocation (signal vs background etc.), labels to use in plot titles etc.


### `B2Inv.yaml` 

YAML file containing lists of variables saved in ntuples using `process_tuples.py` script, and various sets of variables used for BDT training.
Sets of variables defined include:
   - "full-var" - full list of variables put into ntuples.
   - "MCtruth-vars" - list of useful MC truth variables. 
   - "full-vars-plus-MCtruth" - union of "full-var" and "MCtruth-vars", mainly used to study decays which survive full selection in inclusive backgrounds.
   - "eventdisplay-vars" - list of variables needed to make event-display type plots later in analysis.
   - "baseline-bdth-vars","bdth-plus-vars" - various lists of  variables used in training bdt models during development.
   - "bdtlh-vars-v1" - final set of bdt variables used for multiclass BDT.
   - "flavour-tag-vars" - variables used in study of separation of $B_s^0$ and $B^0$.
   - "bdttau-baseline-vars" and "bdttau-nonNeutrals-vars" - variables used to train a BDT which would further remove $Z \rightarrow \tau\tau$ background if needed. Note this was never used for the analysis as all events in the $Z \rightarrow \tau\tau$ MC sample available were removed by the selection.

### `bdt_lh_cut_opt_significance.py` and `bdt_lh_cut_opt_significance_exclusive_backgrounds.py` 

   Most of the functions for going from pandas df with bdt output to sensitivity estimate live in ```bdt_lh_cut_opt_significance.py``` or `bdt_lh_cut_opt_significance_exclusive_backgrounds.py` scripts.

   Note `bdt_lh_cut_opt_significance_exclusive_backgrounds.py` contains equivalent functions to those in `bdt_lh_cut_opt_significance.py` but now updated to allow inclusion of exclusive background samples (mainly cosmetic changes to plots). In addition, `_nodata` functions were  added which allow complete use of splines (removing need to use dataframes beyond point splines are created). **These "_nodata" functions are recommended as they are more self consistent and are used for the final analysis!.**
   

   This includes:

   - ```create_N_map``` - takes a ntuple in pandas dataframe form, and creates an interpolated map of the number of events remaining as a function of cuts in 1-P(h) and 1-P(l), for each unique decay within the df.
   For main fits, we are using RectBivariateSpline with s=0 (as otherwise it takes efficiencies negative at edges), and kx=ky=2 (ie. quadratic spline in both directions - removes wiggles). This function saves the splines and raw dictionaries of number of events remaining with various cuts.

   - ```interp_N_to_eff_err``` takes interp_N_dict and turns it into a 2D array of efficiencies of cuts and the associated uncertainties. The range (`lrange` and `hrange`) over which efficiencies are calulcates and the granularity of the "grid search" points where the efficiency is calculated (`nlh_plot` - this is the number in each dimension) are inputs. Efficincies are calculated for the full selection using the eventsProcessed for each sample - these should be input from ```config```, although this required manually writing them to config file. This can be done using ```get_eventsProcessed``` function in ```efficiency_tools/efficiency_finder.py```. Note that there is an equivalent function for the raw (non-interpolated) efficiencies which are not currently used in the analysis ```raw_N_to_eff_err```.

   - ```run_2d_optimisation``` takes the interpolated N dictionary produced in ```create_N_map```, calculated the efficinecies using `interp_N_to_eff_err`, and uses this to finds the optimum point using $S/\sqrt{S+B}$. Here components of S and B from each sample are calculated using  ```post_bdt_eff_finder.get_total_SB```, these are combined using ```get_total_SB```. All samples input in ```interp_N_dict``` are therefore used in the calculation of S and B.

      This is repeated at each point of the selected parameter space for a given signal BF (given - must be same as that used for ```interp_N_to_eff_err```) and the FOM calculated for S and B. The FOM array is returned from which the maximum point can be found.

   - ```make_final_binning_plot``` takes BOTH ```df_data``` and ```interp_N_dict```. 

      ```interp_N_dict``` is used to find the optimal cut point, then that cut is applied on ```df_data```, and the number of events in each bin of 2x2 (or however many bins input in function) BDT space is calculated from the non-interpolated MC. From this get per bin N-expected which are combined into S and B -> then option to plot. 

      In the equivalent "_nodata" function in `bdt_lh_cut_opt_significance_exclusive_backgrounds.py` this is corrected so that no `df_data` input is requred and the number of events in each bin of 2x2  BDT space is also calculated from the interpolated maps. In reality, the difference between teh output of these two methods is similar as expected.

   - `likelihood_model_builder` runs toy fit for sensitivity estimate. Again it takes BOTH `df_data` and `interp_N_dict` as it initially uses `make_final_binning_plot` to get S and B per bin. Then uses these S and B values from cases where 4 and 1 bins in the BDT accepted space (the latter for the overall systematic used as gaussian constraint on the bkg) to define the fit to toy data. Again with option of plotting.

      Again here "_nodata" version in `bdt_lh_cut_opt_significance_exclusive_backgrounds.py` uses interpolated maps instead of data, and is used in final analysis.

   - `calculate_BF_sensitivities` currently just takes interp_N_dict and then does S and B calculations from there. This therefore consistent with toy approach using "_nodata" functions in `bdt_lh_cut_opt_significance_exclusive_backgrounds.py`. This also contains the option to make the final sensitivity plot.
   

   Note: Variants of the above titled `_no_shape_assumed` and `_single_signal` were used to investigate the sensitivity of the fit to toys if we dont assume we know anything about the shape of the distribution we're fitting, and to reproduce sensitivity plots we'd get if we assume everything we observe was due to just a Bs or Bd (rather than combined), respectively.

### Other key scripts
   - `basic_functions.py`: defines basic functions for checking input paths etc. Very simple but used in most scripts.

   - `bdt_lh_training`: script used to train multiclass BDT (BDT_lh) using XGBoost from pandas dataframes produced using `df_makers/bdt_lh_create_df.py` script. Includes hyperparameter optimisation using optuna.

   - `create_new_interps.py` and `merge_interps.py`:  quick scripts using tools defined in `bdt_lh_cut_opt_significance.py` to make (and merge in Tau2HNu sample produced separately) final interpolated BDTlh cut maps used in analysis (including those for exclusive B(c)2lnu backgrounds). These use a different number of BDT cuts to interpolate over depending on the stats of the MC sample left with a tight BDT window to ensure best performance of interpolation. Resulting files all found in `8x8xmidstats_4x4lowstats` folder as use 4x4 grid of cuts for low stats, 8x8 for mid stats and 20x20 for high stats samples. High, medium and low stats are defined in config.

   - `get_eventsProcessed.py` and `events_processed_full_data.ipynb`: very quick script/notebook to evaulate events processed for different samples.


### Variable Plotting - `plotters` folder

 - `variable_plotter.py`: simple plotting script for variables from root file that reads from `config.py`. Example usage `python -i variable_plotter.py` can interactively make some plots.
- `post_bdt_variable_plotter.py`: equivalent of `variable_plotter.py` except to read from pandas dataframes with BDT selection applied.
 - `matt_paper_plotter.py`: Updateded plotting script to make paper-ready histograms.
 - `bdt_plotter_multiclass.py`:produce various plots which show BDT performance including ROC curves.
 - `flavtag_multiclass.py`: script runs $B_S^0$-$B^0$ separation study based on studying $K^\pm$ and $K^0$ hadronisation partners. Includes functions to reconstruct $\pi^0$ from $K^0$.
  - `make_paper_plots` and `make_study_plots` contain scripts that run actual plots used for paper and various studies, respectively.

### `df_makers`
Contains scripts to make various pandas dataframes from Rdf for different purposes, as well as applying and cutting on BDT in various configurations. These can be split into function defining scripts:
- `data_to_pickle_function.py`: defines function to turn root Rdf into pandas df such that can be run through BDT (`root_data_to_pickle_df`). But also includes functions which allow root data to be run through BDT directly with BDT output scored saved to a friend tree
(`add_BDT_to_new_root_files`) and then cut on BDT stored in this way (`add_friends_and_bdtcut`). These last functions are mainly used in background analysis.
- `apply_bdt_and_pickle_df.py` - function defined to load and apply generic BDT to a pandas df (`load_bdt_and_apply`). Note in main, takes raw root files, turns into pandas, applies BDT and saves as pickled df.
- `combine_dfs.py`: defines function to load all pickled dataframes of multiple samples and store in single df once cuts applied (`load_all_pickles_into_dataframe`).



and scripts to run various productions:
- `bdt_lh_create_df.py` - create df used to train BDT including weighting based on production fractions and preselection efficiencies, and then additional balancing weights between S and B classes.
- `pickle_full_data.py`: simple script which uses `root_data_to_pickle_df` function defined in `data_to_pickle_function.py` to create df that can have BDT applied to it later.
- `apply_bdt_lh.py` - quick script that uses(`load_bdt_and_apply`) to apply final multiclass BDT (bdt_lh) to data that has already been turned into a pandas df and pickled.
- `process_lnu_backgrounds.py`: very quick script to apply BDT to exclusive B(c)2lnu background samples using `data_to_pickle_function.root_data_to_pickle_df`
- `flavtag_df_maker.py`: very quick script to apply baseline-plus bdt_hl to  data for investigation of $B^0$- $B^0_s$ sepration.
- `bdt_to_root_files.py` - quick script that uses functions defined in `data_to_pickle_function.py` to apply BDT stright to root files, save output as a friend tree and then cut on BDT saving to new root file.  This used for background studies.




Overall, to apply BDT (to pandas df as done for body of analysis):
1. Use `bdt_lh_create_df` to create weighted pandas df to train BDT.
2. Train multiclass BDT using `bdt_lh_training.py` script.
3. Turn all data into pandas df to run throught BDT using `pickle_full_data.py` (or similar script based on `root_data_to_pickle_df` function from `data_to_pickle_function.py`).
4. Load BDT and apply cuts on this data using `apply_bdt_lh.py` (or similar script based on `load_bdt_and_apply` function in `apply_bdt_and_pickle_df.py`).

For additional studies use:
1. `flavtag_df_maker.py`to investigate $B^0$- $B^0_s$ separation.
2. `add_BDT_to_new_root_files`and `add_friends_and_bdtcut` functions in `data_to_pickle_function.py` script to apply BDT to root files to facilitate easy study of surviving background decays (as files get very large with all the MC info too!).


### `efficiecy_tools` 
Contains scripts which define tools to calculate efficiency, as well as $S$ and $B$ expectations, and do error propergation.

- `efficiency_finder.py`: contains functions to calculate efficiencies and sample expectations from root files. Includes functions to get eventsProcessed (`get_eventsProcessed`), calulate efficiency and calculate efficiency uncertainty using a symmetrised [Wilson interval](https://doi.org/10.2307/2276774) (`efficiency_calc`). `get_efficiencies` brings gets eventsProcessed and feeds into `efficiency_calc`.This can then be coverted to a sample expectation using branching fractions in config (`get_sample_expectations`).
- `post_bdtlh_efficiency_finder.py`: function to calculate full selection efficiencies from dictionary of events remaining after BDT cuts (`get_eff_from_nMC_list`) and turning this into a sample expectation (`get_n_expected_components`). Also includes functions to combine these expectations into expected total $S$ and $B$ and do full error propagation for this (`get_total_SB`). 
- `get_eff.py`: small script to print preselection efficiencies and save them to a logfile.


 
### Background studies with `background_analysis` and `pythia decays` folders

- `background_analysis` contains scripts to study exclusive B(c)2lnu sample contributions, and investigate backgrounds that survive full selection from inclusive samples.

   + Scripts analysing surviving decays in inclusive samples:
      - `background_analysis_root.py`: Script containing mutliple functions to analyse decays remaining in various background samples. Eg.  `build_decay_tree_from_Bhadron` which finds heavy hadron decays in backgrounds that survive (both SS and OS) and returns dictionary of results. There are analogous functions for light (ss, ud) samples `build_Zss_decay_tree` where start from quark rather than hadron. These I beleive to be simpler and more recent versions of `build_decay_tree_Dgr2`.
      - `test_tree.ipynb`: attempt to visualise decays in  a tree
      - `background_search.ipynb`: notebook used to test parenting algorithm and check thrust alignment 


   + Scripts looking at exclusive B(c)2lnu:
    - `Bu2taunu_background_check.ipynb` - quick invetsigation into surviving B(c)2lnu backgrounds
    - `calc_B2lnu_eff_avges.py` - script to investigate selection efficiency on these backgrounds and include in final binning plot (double counted as already in inclusive sample)


- `pythia decays` contains [xml file of decays](https://gitlab.com/Pythia8/releases/-/blob/master/share/Pythia8/xmldoc/ParticleData.xml?ref_type=heads) input to pythia (to produce inclusive samples) and a python script `pythia_decays_disentangler.py` to turn PDG codes into human-readable decays for clarity of what decays included in inclusive samples!





### Further notes

#### Logistics of recreating project (to be fleshed out)
1. Clone the repo
   
   ```bash
   git clone git@github.com:RMangrulkar/FCCAnalyses.git
   ```
   or
   ```bash
   git clone https://github.com/RMangrulkar/FCCAnalyses.git
   ```
   
2. Set up the environment [note to build fccanalysis you will need to move any outputs elsewhere to avoid the build being very slow as it copies the whole FCCAnalyses folder]
   
   ```bash
   cd FCCAnalyses
   git switch b2inv
   source setup.sh
   fccanalysis build -j 8
   cd examples/FCCee/flavour/B2Inv/
   ```
3. Produce ntuples using FCCAnalyses framework (having set desired configuration in config.py and B2Inv.yaml)
   ``` fccanalysis run process_tuples.py```

4. offline analysis as described above.


#### Vertexing issue
- The `winter2023` samples do not perform well with the vertexing algorithm used, which leads to very slow tupling (only ~20 events per second!)
  The issue was raised by Aidan [here](https://github.com/HEP-FCC/FCCAnalyses/issues/378) and [here](https://fccsw-forum.web.cern.ch/t/legacy-vertexing-issue/219)
- Believe to be (at least in part) due to use of old hardcoded beamspot constraints in fit. Updatese in FCCAnalyses using function overloading and BCS specified in config.

#### Version incompatibility

Note that FCCAnalyses uses the `key4hep` stack which has older versions of Python, ROOT, xgboost, etc. The latest versions of xgboost and PyROOT (which are much faster) are apparently not backwards compatible with these older version.



 



 
