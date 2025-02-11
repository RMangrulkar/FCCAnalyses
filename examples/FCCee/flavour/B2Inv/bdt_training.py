# Train BDT using prelimcuts files

import os
import glob
import sys

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
from tabulate import tabulate
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.metrics import roc_curve, auc


# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg 
import efficiency_finder

#import bdt_plotter as bdtplt


ROOT.EnableImplicitMT()

##########################################################
# function to retrive lists from yaml and check file paths
###########################################################
# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars

def check_inputpath(inputpath):
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"{inputpath} does not exist")
    return inputpath


def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath


#Define function that does training
def train_bdt(pickled_df_fname = "bdth_dataframe.pkl", 
              config_bdtopts = cfg.bdth_opts,
              training_round = "baseline",
              hps_dict_name = "default-hps",#if not using default, name of hp config in config
              features_list_name = "baseline-bdth-vars",
              bdt_label = 'h'): 
    
    
    ## PREPROCESSING AND CREATING df

    print(f"{30*'-'}")
    print(f"BDT{bdt_label} TRAINING")
    print(f"{30*'-'}\n")
    print("Initialising...")

    # Load configuration
    plt.style.use(os.path.abspath(os.path.join(cfg.FCCAnalysesPath, 'fcc.mplstyle')))

    #path to data and outputs
    outputpath   = set_outputpath(config_bdtopts['outputPath'])
    yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])

    #Getting BDT vars for training from yaml
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'

    #loading saved data
    pickled_df_path = os.path.join(outputpath, pickled_df_fname)
    df = pd.read_pickle(pickled_df_path)

    # matrix of input vars
    x_train = df[ df["sample"]==0][bdtvars]
    x_test  = df[ df["sample"]==1][bdtvars]
    x_valid = df[ df["sample"]==2][bdtvars]

    # array of target
    y_train = df[ df["sample"]==0][ "label" ]
    y_test  = df[ df["sample"]==1][ "label" ]
    y_valid = df[ df["sample"]==2][ "label" ]

    # array of weights
    w_train = df[ df["sample"]==0][ "total_weight" ]
    w_test  = df[ df["sample"]==1][ "total_weight" ]
    w_valid = df[ df["sample"]==2][ "total_weight" ]


    # Set Default hyperparameters - updated with those input
    default_hps = cfg.hp_opts['default-hps']

    if hps_dict_name != 'default_hps':
        hps =  cfg.hp_opts[hps_dict_name]
        # Update default hyperparameters with any provided 
        default_hps.update(hps)

    #Possible hps to try
        #"n_estimators": [100, 150, 200],
        #"learning_rate": [0.1, 0.3],
        #"max_depth": [3, 5, 7],
        #"subsample": [0.7, 1.0],
        #"gamma": [0, 0.1, 0.2],
        #"min_child_weight": [1, 5],

 
    ## TRAINING OF BDT

    ## Currently no cross validation - potentially TO IMPLEMENT LATER
    ## Also for now no hp opt - TO IMPLEMENT LATER

    print("CURRENTLY NO HP OPTIMISATION OR CROSS VALIDATION")
    # now let's train it
    bdt = xgb.XGBClassifier( objective='binary:logistic') #xgb default objective
    #nb. could add early_stopping_rounds=10, eval_metric="auc",  - if specify first want to specify second as this is metric used for early stopping

    bdt.set_params(n_estimators=default_hps['n_estimators'],
                    learning_rate=default_hps['learning_rate'],
                    max_depth=default_hps['max_depth'],
                    gamma=default_hps['gamma'],
                    min_child_weight=default_hps['min_child_weight'],
                    max_delta_step=default_hps['max_delta_step'],
                    subsample=default_hps['subsample']) 
    
    print(f"\n----> INFO: Training using {default_hps}")

    print("\n Training model")
    bdt.fit( x_train, y_train, 
            sample_weight=w_train, 
            eval_set=[(x_test, y_test)], 
            sample_weight_eval_set=[w_test], 
            verbose=10 )


    # now put it's predictions back into the frame 
    # The predict_proba() method returns a 2D array where each row corresponds to a sample
    # each column represents the probability of that sample belonging to a particular class.
    df['bdt_score'] = bdt.predict_proba( df[bdtvars] )[:,1]

    # get the feature importance
    importance_indices = np.argsort(bdt.feature_importances_)
    sorted_features = bdt.feature_names_in_[importance_indices[::-1]]
    sorted_importances = bdt.feature_importances_[importance_indices[::-1]]
    print(f"Feature importance type: {bdt.importance_type}")
    print( "Feature Importance:")
    print( tabulate( zip( sorted_features, sorted_importances ) ) )

    # save the model to a file for use later
    bdt.save_model(os.path.join(outputpath,training_round,f"{bdtname}.json"))

    # Write key info about df to log file
    with open(os.path.join(outputpath,training_round,f'{bdtname}_training_info.log'), 'a') as log_file:
        log_file.write(f'Feature Importance: { tabulate( zip( sorted_features, sorted_importances ) )}\n')

    # Write key info about df to log file
    with open(os.path.join(outputpath,training_round,f'{bdtname}_training_vars.log'), 'a') as log_file:
        log_file.write(f'bdt_training_vars: {bdtvars}\n')

    return bdt, df








'''# branching fractions for weights
branching_fractions = cfg.branching_fractions #dictionary containing decay name and tuple with BF and its error

# print statements to check loading things expect
print(f"----> INFO: Using {bdtvars_list} from")
print(f"{15*' '}{yamlpath}")
print(f"----> INFO: Loading files from")
print(f"{15*' '}{inputpath}")
print(f"----> INFO: Output will be saved to")
print(f"{15*' '}{outputpath}")


#getting training decays from config decay list
signal_decays =  cfg.bdth_opts["signalAllocation"]
background_decays =  cfg.bdth_opts["backgroundAllocation"]
training_decays = signal_decays + background_decays # Not including the Z->ee, Z->mumu, Z->tautau decays in the training but we still want to process them to see how the BDT does

decays_dict={'signal':signal_decays,'background':background_decays}

print(f"----> INFO: Using signal decays:")
print(f"{15*' '}{signal_decays}")
print(f"----> INFO: Using background decays:")
print(f"{15*' '}{background_decays}")

#calculating efficiencies and also saving files paths used to calculate efficiencies to ensure do training on same files
selection_efficiency = efficiency_finder.get_efficiencies('custom',
                                                    further_analysis=True,
                                                    samples = training_decays,
                                                    raw=True, #ie. want full efficiency including tupling and prelim cuts
                                                    custompath=inputpath,
                                                    verbose=False,
                                                    return_files_list=True)
#nb if want to specify a certain number of chunks to use do it in here and will follow through into filepaths

training_filepaths_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_files')}
efficiencies_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_eff')}
efficiencies_err_dict = {key: value for key, value in selection_efficiency.items() if key.endswith('_err')}


# going to print the efficiencies and BF for each now so can manually check
print("Efficiencies and BFs:"+'\n')
eff_to_print = [[ key , efficiencies_dict[key+'_eff'],branching_fractions[key][0] ] for key in training_decays]
print( tabulate(  eff_to_print, headers=["decay", "efficiency", "BF"] ) +'\n')


##############################################################
## Collecting events into dfs and applying weights
##############################################################

# now collect relevant events into a dataframe
df_dict = {}

for decay in training_decays:
    eff = efficiencies_dict[decay+'_eff']
    weight = branching_fractions[decay][0] * eff #need zero index for bf as branching fractions is a tuple in the yaml
    filepaths = training_filepaths_dict[decay+'_files']
    Rdf = ROOT.RDataFrame("events", filepaths)
    Rdf_np = Rdf.AsNumpy(columns= bdtvars+responsevars)
    sub_df = pd.DataFrame(Rdf_np)
    sub_df["decay"] = decay
    sub_df["w1"] = weight / len(sub_df) 
    df_dict[decay] = sub_df

# going to print the sum of weights for each now
# can check this is consistent with BF * eff (which it should be)
print("Sum of weights:"+'\n')
print_rows=[]
for decay in training_decays:
    sumw = df_dict[decay]["w1"].sum()
    nevs = len(df_dict[decay])
    print_rows.append( [ decay, sumw, nevs ] )

print( tabulate( print_rows, headers=["decay", "sumWeights", "numEvents"] ) +'\n')


# this so far has weighted correctly within each type of sample (ie. signal or bkg) - now want to weight so that overall satisfy two conditions
# let weights from previous section be W1_kk where kk is either quark combo or Bs,Bd
# let new weights be W2_s for signal and W2_b for bkg
#1. Account for the fact that our training sample contains more background than signal
#     This required overall: 
#                 sum over q [(W1_qq * n_q)] W2_b = (W1_Bs * n_Bs + W1_Bd * n_Bd) W2_s
#2. Sum of all events * total weight for that event = Sum of all events
#     This requires: 
#                 sum over q [(W1_qq * n_q)] W2_b + (W1_Bs * n_Bs + W1_Bd * n_Bd) W2_s = n_s + n_b
#
# Solving these simultaneously, using the fact that W1_kk = (Bf_k * eff_k) / n_k means that :
#                  W2_s = (n_s + n_b) / 2(W1_Bs * n_Bs + W1_Bd * n_Bd)
#                  W2_b = (n_s + n_b) / 2 * [(W1_qq * n_q)] summed over q        

#total number of events in sig and background (ie. n_s + n_b)
n_total = sum(len(df_dict[s]) for s in training_decays)


for allocation in ["signal","background"]:
    samples =  decays_dict[allocation]
    # Calculate the denominator by summing twice the sum of weights for each sample
    denom = sum(2 * df_dict[s]["w1"].sum() for s in samples)
    w2 = n_total/denom

    for s in samples:
        df_dict[s]["w2"] = w2
        df_dict[s]["total_weight"] = df_dict[s]["w1"] * w2


#############################################################################################
##Checks on  weights performed (as calculaing manually and dont want to have made a mistake)
##############################################################################################
#print out weights for visual check
print("Weights assigned to each decay:"+'\n')
weights_print_row = [[decay,df_dict[decay]["w1"][0],df_dict[decay]["w2"][0],df_dict[decay]["total_weight"][0]] for decay in training_decays]
print( tabulate( weights_print_row, headers=["decay", "w1", "w2","total_weight"] )+'\n' )

### A few checks to make sure the total weights are behaving as desired
#check1 - total sum of weights for events over all decays == total number of events
check1 = np.isclose(sum(len(df_dict[s]) for s in training_decays), sum(df_dict[s]["total_weight"].sum() for s in training_decays),rtol=1e-07) #check agreememt to within relative tolerance of 1e-7
#Check2: Sum of weights in signal == Sum of weights in bkg
check2 = np.isclose(sum(df_dict[s]["total_weight"].sum() for s in signal_decays),sum(df_dict[b]["total_weight"].sum() for b in background_decays),rtol=1e-07)
#"Check3: Ratio sum of weights in Bs:Bd = Bs_eff/Bd_eff"
check3 = np.isclose(df_dict[signal_decays[0]]["total_weight"].sum()/df_dict[signal_decays[1]]["total_weight"].sum(), branching_fractions[signal_decays[0]][0]*efficiencies_dict[signal_decays[0]+'_eff']/ (branching_fractions[signal_decays[1]][0]*efficiencies_dict[signal_decays[1]+'_eff']),rtol=1e-07)
#Check4:Ratio sum of weights in bb:cc = bb_eff*BF(Z->bb)/cc_eff*BF(Z->cc)
check4 = np.isclose(df_dict[background_decays[0]]["total_weight"].sum()/df_dict[background_decays[1]]["total_weight"].sum(), branching_fractions[background_decays[0]][0]*efficiencies_dict[background_decays[0]+'_eff']/ (branching_fractions[background_decays[1]][0]*efficiencies_dict[background_decays[1]+'_eff']),rtol=1e-07)
#Check5: Ratio sum of weights in bb:ss = bb_eff*BF(Z->bb)/ss_eff*BF(Z->ss) 
check5 = np.isclose(df_dict[background_decays[0]]["total_weight"].sum()/df_dict[background_decays[2]]["total_weight"].sum(),branching_fractions[background_decays[0]][0]*efficiencies_dict[background_decays[0]+'_eff']/ (branching_fractions[background_decays[2]][0]*efficiencies_dict[background_decays[2]+'_eff']),rtol=1e-07)
# Check6: Ratio sum of weights in bb:ud = bb_eff*BF(Z->bb)/ud_eff*BF(Z->ud)
check6 = np.isclose(df_dict[background_decays[0]]["total_weight"].sum()/df_dict[background_decays[3]]["total_weight"].sum(),branching_fractions[background_decays[0]][0]*efficiencies_dict[background_decays[0]+'_eff']/ (branching_fractions[background_decays[3]][0]*efficiencies_dict[background_decays[3]+'_eff']),rtol=1e-07)

checks= [check1,check2,check3,check4,check5,check6] 
check_names = ["Check1: Total sum of weights for events over all decays == total number of events","Check2: Sum of weights in signal == Sum of weights in bkg","Check3: Ratio sum of weights in Bs:Bd = Bs_eff/Bd_eff",
               "Check4:Ratio sum of weights in bb:cc = bb_eff*BF(Z->bb)/cc_eff*BF(Z->cc)","Check5: Ratio sum of weights in bb:ss = bb_eff*BF(Z->bb)/ss_eff*BF(Z->ss)", "Check6: Ratio sum of weights in bb:ud = bb_eff*BF(Z->bb)/ud_eff*BF(Z->ud)"]

# Check if any value is False
if any(check == False for check in checks):
    print("Error: weights have not passed all checks: continuing to find problem")
    # Print which checks are False
    for check, name in zip(checks, check_names):
        if not check:
            raise ValueError(f"Weights not calculated correctly, {name}, has failed")
else:
    print("----> INFO: Weights have passed all checks!")


##########################################################################
## Now combining into one df and labelling assuming weights pass checks
#########################################################################

# combining dataframes into one
df = pd.concat( [df_dict[s] for s in training_decays], ignore_index=True )

# want to make sure that integer types are actually set as integers - currenlty stored as float
#if changed branches significantly might be worth checking the list is still right, with current branches expected integers in yaml
integer_branches = [s for s in bdtvars+responsevars if '_n' in s and '_norm' not in s]
for integer_branch in integer_branches:
    df[integer_branch] = df[integer_branch].astype(np.int32)

#  label background and signal events as 0 and 1 for classifier 
def labeller(dec):
    if dec in signal_decays:
        return 1
    else:
        return 0

df["label"] = df["decay"].apply(labeller)

# now shuffle the whole dataframe around to avoid any funny biases
# do this with a random seed so it's reproducible
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# now create labels for the train, test and validation split
# we will give them indices train==0, test==1, validation==2
# use a random seed for this so it's reproducible
np.random.seed(210187)
train_frac = 0.75
test_frac= 0.125
valid_frac = 1-train_frac-test_frac
sample_indices = np.random.choice( [0,1,2], p=[train_frac, test_frac, valid_frac], size=len(df) )
df["sample"] = sample_indices
'''
