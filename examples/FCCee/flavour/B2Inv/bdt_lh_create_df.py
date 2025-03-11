import os
import glob
import sys

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from yaml import safe_load, YAMLError


# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg 
import efficiency_finder


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

#################################
## PREPROCESSING AND CREATING DF
#################################
print(f"{30*'-'}")
print(f"CREATING AND SAVING df FOR BDT-lh TRAINING")
print(f"{30*'-'}\n")


#path to data and outputs
inputpath    = check_inputpath(cfg.bdt_lh_opts['inputPath']) 
outputpath   = set_outputpath(cfg.bdt_lh_opts['outputPath'])
yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])

#Getting BDT vars for training from yaml
bdtvars_list = cfg.bdt_lh_opts['mvaBranchList']
bdtvars      = vars_fromyaml(yamlpath, bdtvars_list)
responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot

# branching fractions for weights
branching_fractions = cfg.branching_fractions #dictionary containing decay name and tuple with BF and its error

# print statements to check loading things expect
print(f"----> INFO: Using {bdtvars_list} from")
print(f"{15*' '}{yamlpath}")
print(f"----> INFO: Loading files from")
print(f"{15*' '}{inputpath}")
print(f"----> INFO: Output will be saved to")
print(f"{15*' '}{outputpath}")


#getting training decays from config decay list
signal_decays =  cfg.bdt_lh_opts["signalAllocation"]
light_bkg =  cfg.bdt_lh_opts["backgroundAllocation_light"]
heavy_bkg =  cfg.bdt_lh_opts["backgroundAllocation_heavy"]
background_decays = heavy_bkg+light_bkg
training_decays = signal_decays + heavy_bkg + light_bkg # Not including the Z->ee, Z->mumu, Z->tautau decays in the training but we still want to process them to see how the BDT does

decays_dict={'signal':signal_decays,'light_background':light_bkg, 'heavy_background':heavy_bkg, 'full_background':background_decays}

print(f"----> INFO: Using signal decays:")
print(f"{15*' '}{signal_decays}")
print(f"----> INFO: Using background decays:")
print(f"{15*' '}{background_decays} split into two classes {heavy_bkg} for heavy hadrons and {light_bkg} for light hadrons ")

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


##################################################################################
#How weights balanced if only S and B classes, keep in df for now                #
##################################################################################

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


for allocation in ["signal","full_background"]:
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


##################################################################################
# Now make equivalent for muliclass               #
##################################################################################

# this so far has weighted correctly within each type of sample (ie. signal or bkg) - now want to weight so that overall satisfy two conditions
# let weights from previous section be W1_kk where kk is either quark combo or Bs,Bd
# let new weights be W3_s for signal,  W3_h for heqvy bkg and W3_l for light bkg
#1. Account for the fact that our training sample contains more background than signal
#     This required overall: 
#                 sum over q [(W1_qq * n_q)] W3_l = [(W1_qq * n_q)] W3_h= (W1_Bs * n_Bs + W1_Bd * n_Bd) W3_s
#2. Sum of all events * total weight for that event = Sum of all events
#     This requires: 
#                 sum over q [(W1_qq * n_q)] W3_l+ [(W1_qq * n_q)] W3_h + (W1_Bs * n_Bs + W1_Bd * n_Bd) W3_s = n_s + n_b
#
# Solving these simultaneously, using the fact that W1_kk = (Bf_k * eff_k) / n_k means that :
#                  W3_s = (n_s + n_b) / 3(W1_Bs * n_Bs + W1_Bd * n_Bd)
#                  W3_h = (n_s + n_b) / 3 * [(W1_qq * n_q)] summed over q = bb and cc  
#                  W3_l = (n_s + n_b) / 3 * [(W1_qq * n_q)] summed over q = ud and ss       


for allocation in ["signal","heavy_background","light_background"]:
    samples =  decays_dict[allocation]
    # Calculate the denominator by summing twice the sum of weights for each sample
    denom = sum(3 * df_dict[s]["w1"].sum() for s in samples)
    w3 = n_total/denom

    for s in samples:
        df_dict[s]["w3"] = w3
        df_dict[s]["total_weight_muliclass"] = df_dict[s]["w1"] * w3


#############################################################################################
##Checks on  weights performed (as calculaing manually and dont want to have made a mistake)
##############################################################################################
#print out weights for visual check
print("Weights assigned to each decay for multiclass BDT:"+'\n')
weights_print_row = [[decay,df_dict[decay]["w1"][0],df_dict[decay]["w3"][0],df_dict[decay]["total_weight_muliclass"][0]] for decay in training_decays]
print( tabulate( weights_print_row, headers=["decay", "w1", "w3","total_weight_muliclass"] )+'\n' )

### A few checks to make sure the total weights are behaving as desired
#check1 - total sum of weights for events over all decays == total number of events
check1 = np.isclose(sum(len(df_dict[s]) for s in training_decays), sum(df_dict[s]["total_weight_muliclass"].sum() for s in training_decays),rtol=1e-07) #check agreememt to within relative tolerance of 1e-7
#Check2: Sum of weights in signal == Sum of weights in light bkg
check2 = np.isclose(sum(df_dict[s]["total_weight_muliclass"].sum() for s in signal_decays),sum(df_dict[b]["total_weight_muliclass"].sum() for b in heavy_bkg),rtol=1e-07)
#Check3: Sum of weights in signal == Sum of weights in heavy bkg
check3 = np.isclose(sum(df_dict[s]["total_weight_muliclass"].sum() for s in signal_decays),sum(df_dict[b]["total_weight_muliclass"].sum() for b in light_bkg),rtol=1e-07)

#Check4: Ratio sum of weights in Bs:Bd = Bs_eff/Bd_eff"
check4 = np.isclose(df_dict[signal_decays[0]]["total_weight_muliclass"].sum()/df_dict[signal_decays[1]]["total_weight_muliclass"].sum(), branching_fractions[signal_decays[0]][0]*efficiencies_dict[signal_decays[0]+'_eff']/ (branching_fractions[signal_decays[1]][0]*efficiencies_dict[signal_decays[1]+'_eff']),rtol=1e-07)
#Check5:Ratio sum of weights in bb:cc = bb_eff*BF(Z->bb)/cc_eff*BF(Z->cc)
check4 = np.isclose(df_dict[background_decays[0]]["total_weight_muliclass"].sum()/df_dict[background_decays[1]]["total_weight_muliclass"].sum(), branching_fractions[background_decays[0]][0]*efficiencies_dict[background_decays[0]+'_eff']/ (branching_fractions[background_decays[1]][0]*efficiencies_dict[background_decays[1]+'_eff']),rtol=1e-07)
# Check6: Ratio sum of weights in ss:ud = bb_eff*BF(Z->bb)/ud_eff*BF(Z->ud)
check6 = np.isclose(df_dict[background_decays[2]]["total_weight_muliclass"].sum()/df_dict[background_decays[3]]["total_weight_muliclass"].sum(),branching_fractions[background_decays[2]][0]*efficiencies_dict[background_decays[2]+'_eff']/ (branching_fractions[background_decays[3]][0]*efficiencies_dict[background_decays[3]+'_eff']),rtol=1e-07)

checks= [check1,check2,check3,check4,check5,check6] 
check_names = ["Check1: Total sum of weights for events over all decays == total number of events","Check2: Sum of weights in signal == Sum of weights in heavy bkg","Check3: Sum of weights in signal == Sum of weights in light bkg","Check4: Ratio sum of weights in Bs:Bd = Bs_eff/Bd_eff",
               "Check5:Ratio sum of weights in bb:cc = bb_eff*BF(Z->bb)/cc_eff*BF(Z->cc)", "Check6: Ratio sum of weights in ss:ud = ss_eff*BF(Z->bb)/ud_eff*BF(Z->ud)"]

# Check if any value is False
if any(check == False for check in checks):
    print("Error: Multiclass weights have not passed all checks: continuing to find problem")
    # Print which checks are False
    for check, name in zip(checks, check_names):
        if not check:
            raise ValueError(f"Multiclass Weights not calculated correctly, {name}, has failed")
else:
    print("----> INFO: Multiclass Weights have passed all checks!")



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
        return 2
    elif dec in heavy_bkg:
        return 1
    elif dec in light_bkg:
        return 0
    else:
        raise ValueError('Expect all data to be either light or heavy background if not signal')

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

# matrix of input vars
x_train = df[ df["sample"]==0][bdtvars]
x_test  = df[ df["sample"]==1][bdtvars]
x_valid = df[ df["sample"]==2][bdtvars]

# array of target
y_train = df[ df["sample"]==0][ "label" ]
y_test  = df[ df["sample"]==1][ "label" ]
y_valid = df[ df["sample"]==2][ "label" ]

# array of weights
w_train = df[ df["sample"]==0][ "total_weight_muliclass" ]
w_test  = df[ df["sample"]==1][ "total_weight_muliclass" ]
w_valid = df[ df["sample"]==2][ "total_weight_muliclass" ]


print("\n----> INFO: Preprocessing done")
print(f"{15*' '}Using {len(x_train):>8} events to train ({100*train_frac:.1f}% of total)")
print(f"{15*' '}Using {len(x_test):>8} events to  test ({100*test_frac:.1f}% of total)")
print(f"\n{30*'-'}\n")


# Save the DataFrame to a pickle file for now - can look at cuts
df.to_pickle(os.path.join(outputpath,'bdt_lh_dataframe.pkl'))
print("DataFrame saved successfully!")

# Write key info about df to log file
with open(os.path.join(outputpath,'bdt_lh_dataframe_info.log'), 'a') as log_file:
    log_file.write(f'data taken from path: {inputpath}\n')
    log_file.write(f'files used: {training_filepaths_dict}\n')
    log_file.write(f'signal decays: {signal_decays}\n')
    log_file.write(f'background decays: {background_decays}\n')
    log_file.write(f'These split into light:{light_bkg} and heavy: {heavy_bkg} backgrounds\n')
    log_file.write(f'data taken from path: {inputpath}\n')
    log_file.write(f'Efficiencies and BF used: {tabulate( eff_to_print, headers=["decay", "efficiency", "BF"])}\n')
    log_file.write(f'\nFirst set of weights: { tabulate( print_rows, headers=["decay", "sumWeights", "numEvents"] )}\n')
    log_file.write(f'\nFull weights: { tabulate( weights_print_row, headers=["decay", "w1", "w2","total_weight"] )}\n')
    log_file.write(f'\ntraining fraction ({100*train_frac:.1f}% \n')
    log_file.write(f'testfraction ({100*test_frac:.1f}% \n')




