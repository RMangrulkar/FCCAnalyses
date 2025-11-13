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
   
3. Save the stage1 training sample (defaults to `outputs/stage1_training`)
   ```bash
   fccanalysis run stage1_training.py
   ```

```
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



 
