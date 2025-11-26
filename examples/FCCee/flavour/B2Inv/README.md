# Workflow overview
1. Process tuples using FCCAnalyses framework and `process_tuples.py` script - specifying runmode from config options

2. Create training data tuple with `df_makers` and train BDT with `bdt_lh_training.py` 

3. `pickle_full_data.py` used to turn into dataframes - ensure only run over samples have tuples on otherwise get divide by 0 error

4. Apply multiclass BDT with `apply_bdt_lh.py`

5. Run cut optimisation and significance calculations with `bdt_lh_cut_opt_significance.py`

# Ideal recreation of project (do not attempt now, WIP)
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
3. Process Tuples
4. Create BDT training samples
5. Train BDT
6. Apply BDT 
5. Optimise BDT cut and calculate Significance
   
   Most of the functions for this live in ```bdt_lh_cut_opt_significance.py```, including the functions:

   - i.```create_N_map``` - takes a dataframe, and for each unique decay within the df creates and interpolated map of the number of events remaining as a function of cuts in 1-P(h) and 1-P(l).
   For main fits, we are using RectBivariateSpline with s=0 (as otherwise it takes efficiencies negative at edges), and kx=ky=2 (ie. quadratic spline in both directions - removes wiggles). This function saves the splines and raw dictionaries of number of events remaining with various cuts.

   - ii.```interp_N_to_eff_err``` takes interp_N_dict and turns it into a 2D array of efficiencies of cuts and there errors. The range (lrange and hrange) over which efficiencies are calulcates and the granularity of the "gridsearch" points where the efficiency is calculated (nlh_plot - this is the number in each dimension) are inputs. Efficincies are calculated for the full selection using the eventsProcessed for each sample - these should be put in calculated and stored in ```config```. This can be done using ```get_eventsProcessed``` function in ```efficiency_tools/efficiency_finder.py```.

   Note that there is an equivalent function for the raw (non-interpolated) efficiencies which are not currently used in the analysis ```raw_N_to_eff_err```

   - iii. ```run_2d_optimisation``` takes the interpolated N dictionary produced in ```create_N_map``` and finds the optimum point using S/sqrt(S+B). Here components of S and B from each sample are calculated from  ```post_bdt_eff_finder.get_total_SB```, these are combined using ```get_total_SB```. All samples input in ```interp_N_dict``` are therefore used in the calculation of S and B.

   This is repeated at each point of the selected parameter for a given signal BF (given - must be same as that used for ```interp_N_to_eff_err```) space and the FOM calculated for S and B. The FOM array is returned from which the maximum point can be found.

   - iv.```make_final_binning_plot``` takes BOTH ```df_data``` and ```interp_N_dict```. 

   ```interp_N_dict``` is used to find the optimal cut point, then that cut is applied on ```df_data```, number of events in each bin of 2x2 (or however many bins input in function) BDT space is calculated from the non-interpolated MC. From this get per bin N-expected which are combined into S and B -> then option to plot. Note it does the mechanics to check if extra exlusive backrounds internally.

   - v. `likelihood_model_builder` runs toy fit for sensitivity estimate. Again it takes BOTH `df_data` and `interp_N_dict` as it initially uses `make_final_binning_plot` to get S and B per bin. Then uses these S and B values from when 4 and 1 bin (the latter for the overall systematic used as gaussian constraint n bkg)in BDT accepted space to define the fit to toy data. Again with option of plotting.

   - vi. `calculate_BF_sensitivities` currently just takes interp_N_dict and then does S and B calculations from there - this isn't idea as it's not the same as the binned fit AND means that all samples used in S and B estimates must be used in the cut optimisation. But it is much MUCH quicker than going back to the data!! Therefore to change so that it takes samples want to use in optimisation of cut and then those want to use in S and B calc.

   In future should change the toy approach I think so that also doesn't need to use the data. TO RETHINK - WHAT ARE WE USING THE INTERPOLATION FOR? JUST OPTIMISING THE CUT OR GETTING BINNED S AND B TOO - WHATEVER YOU DECIDE YOU NEED CONSISTENCY BETWEEEN TOYS AND THIS (plus option to run optimisation on one set of data and calculate S and B with a different set of backgrounds considered)...
   Plan:
      1. Create and check interpolated maps for extra backgrounds
      2. Change `calculate_BF_sensitivities` so that can choose which elements used in cut optimisation and then which used in S,B calculation
      3. Make main graph again to check unchanged with this update
      4. Make main graph with additional exclusive backgrounds
      5. Update likelihood model builder so that also doesn't need ot use data - using interpolated maps here might actually help with stability too I guess!



   Note: Variants of the above titled `_no_shape_assumed` and `_single_signal` were used to investigate the sensitivity of the fit to toys if we dont assume we know anything about the shape of the distribution we're fitting, and to reproduce sensitivity plots we'd get if we assume everything we observe was due to just a Bs or Bd (rather than combined), respectively.







 Automated merging of background `.root` files (2000\+ which contain ~500 events each) \- WIP
   - Something to the effect of:
      ```
        #!/bin/bash
        cd outputs/stage1
        for directory in [bb, cc, ss, ud]
            bin file number into bins of 250 files
            hadd -v 0 -k -fk chunk_0.root chunk_{bin1}.root (0 to 249)
            hadd -v 0 -k -fk chunk_1.root chunk_{bin2}.root (250 to 499)
            ...
        end
        ``` 


# Detailed reference

### Analyzers

Some dedicated C++ functions are added to `FCCAnalyses/analyzers/dataframe/` for this analysis. Right now they are mostly just dumped into `myUtils` (the odd one in some other namespace) in labelled code block. In the future they should be moved to the "correct" namespace depending on functionality/similarity with existing functions.
- TODO: Also need to check if B2Inv specific analyzers should be moved to FCCeePhysicsPerformance
### `config.py`

This contains almost all of the configuration options, most importantly:
- The attributes used by `fccanalysis run` (see [analysis script](https://hep-fcc.github.io/FCCAnalyses/man/latest/fccanalysis-script.html#ATTRIBUTES))
- Path to every important location
  + Path to the YAML file containing variable names
  + Output directory of `fccanalysis run`
  + Location of BDT1 model and other outputs
  + Location of BDT2 model and other outputs
- Dicts of important keys/values
  + Feature list from YAML to use for BDT1
  + Feature list from YAML to use for BDT2
  + Branching fractions
  + Various efficiencies (currently have to manually save)
  + Sample names, their allocation (signal v background), labels to use in plot titles etc

- Maybe the efficiencies should be moved to another YAML so it can be updated separately

### `B2Inv.yaml` UPDATE

Contains the branch/feature names used in the analysis. Almost every script depends on this and `config.py` in some way or the other.
 - `bdt1-training-opts`
 - `bdt2-training-opts`
 - `stage2-vars` (not used currently)
 - `stage1-vars`
 - `stage0-vars` (may be out of date)


### Variable Plotting

 - `variable_plotter.py` is a simple plotting script that reads from `config.py`
 - example usage `python -i variable_plotter.py` then can interactively make some plots

### `efficiency_finder.py` OLD UPDATE WITH NEW METHOD

 - A simple script that prints (or saves) the efficiencies of a given cut
 - Justification [here](https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf)
 - The `efficiencies` function performs the calculation

 Given the number of events after the cut $k$ and the number
 before the cut $n$, the efficiency is $\epsilon = \frac{k}{n}$ with variance
 
 $$
 \sigma_\epsilon^2 = \frac{(k+1)(k+2)}{(n+2)(n+3)} - \left(\frac{k+1}{n+2}\right)^2
 $$

 This is from assuming that $\exists$ a "true" efficiency and the probability that $k$ events
 survive the cut is a binomial distribution

 $$
 P(k | n) = \binom{n}{k} \epsilon_\text{true}^k (1-\epsilon_\text{true})^{n-k}
 $$



 
