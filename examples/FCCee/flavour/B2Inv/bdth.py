# Train BDTh using prelimcuts files

import os
import glob
import sys

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
from tabulate import tabulate
from time import time
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.metrics import roc_curve, auc


# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg #to change back to config but copied so dont mess up data currently running
import bdt_plotter as bdtplt
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
start = time()
print(f"{30*'-'}")
print(f"BDTh TRAINING")
print(f"{30*'-'}\n")
print("Initialising...")

# Load configuration
plt.style.use(os.path.abspath(os.path.join(cfg.FCCAnalysesPath, 'fcc.mplstyle')))

#path to data and outputs
inputpath    = check_inputpath(cfg.bdth_opts['inputPath']) 
outputpath   = set_outputpath(cfg.bdth_opts['outputPath'])
yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])

#Getting BDT vars for training from yaml
bdtvars_list = cfg.bdth_opts['mvaBranchList']
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


print("\n----> INFO: Preprocessing done")
print(f"{15*' '}Using {len(x_train):>8} events to train ({100*train_frac:.1f}% of total)")
print(f"{15*' '}Using {len(x_test):>8} events to  test ({100*test_frac:.1f}% of total)")
print(f"\n{30*'-'}\n")

#################################
## TRAINING OF BDTh
#################################

## Currently no cross validation - potentially TO IMPLEMENT LATER
## Also for now no hp opt - TO IMPLEMENT LATER

print("CURRENTLY NO HP OPTIMISATION OR CROSS VALIDATION - THIS IS NOT THE FINAL BDT")
# now let's train it
bdt = xgb.XGBClassifier( n_estimators=400,
                         max_depth=3,
                         learning_rate=0.1 ) 

print("\n----> INFO: Training using n_estimators=400, maxdepth=3 and learning_rate=0.1")

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
print( "Feature Importance:")
print( tabulate( zip( sorted_features, sorted_importances ) ) )

# save the model to a file for use later
bdt.save_model( "bdt.json" )


####################################################
## Make response plots defined in bdt_plotter.py        # this should be improved and put in a function when you have a sec Ella!! Currently just using plotters from Matt's script
####################################################

# make roc curve (use test sample)
fpr, tpr, thresholds = roc_curve( y_test, bdt.predict_proba( x_test )[:,1], sample_weight=w_test )
roc_auc = auc(fpr, tpr)
print("AUC = ", roc_auc)

fig, ax = plt.subplots()
ax.plot( tpr, 1-fpr, lw=1 )
ax.set_xlabel( "Signal Efficiency" )
ax.set_ylabel( "Background Rate" )
fig.tight_layout()
fig.savefig("roc.png")
fig.savefig("roc.pdf")


# efficiency plot (on total sample)
fig, ax = plt.subplots()
for decay in df["decay"].unique():
    subf = df[ df["decay"]==decay ]
    mva_scores = subf["bdt_score"].values
    weights = subf["total_weight"].values

    sorted_indices = np.argsort( mva_scores )
    sorted_scores = mva_scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    total_weight = np.sum( sorted_weights ) 
    cumalative_weights = np.cumsum( sorted_weights[::-1] )[::-1] # reverse order for efficiency above cut
    efficiency = cumalative_weights / total_weight

    ax.plot( sorted_scores, efficiency, label=decay )

ax.legend()
ax.set_xlabel('MVA Score')
ax.set_ylabel('Efficiency')
ax.set_yscale('log')
ax.grid(visible=True, which='both', linestyle='-', color='0.7', linewidth=0.7, alpha=0.7)
fig.tight_layout()
fig.savefig("eff.png")
fig.savefig("eff.pdf")

# plot of BDT output
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6.4,6.4))

sig_train = df[ (df["sample"]==0) & (df["label"]==1) ]["bdt_score"].values
bkg_train = df[ (df["sample"]==0) & (df["label"]==0) ]["bdt_score"].values
sig_test = df[ (df["sample"]==1) & (df["label"]==1) ]["bdt_score"].values
bkg_test = df[ (df["sample"]==1) & (df["label"]==0) ]["bdt_score"].values
sig_train_w = df[ (df["sample"]==0) & (df["label"]==1) ]["total_weight"].values
bkg_train_w = df[ (df["sample"]==0) & (df["label"]==0) ]["total_weight"].values
sig_test_w = df[ (df["sample"]==1) & (df["label"]==1) ]["total_weight"].values
bkg_test_w = df[ (df["sample"]==1) & (df["label"]==0) ]["total_weight"].values

# plot training sample dists
ax[0].hist( bkg_train, bins=50, range=(0,1), label='Bkg Train', alpha=0.5, ec='none', fc='r', weights=bkg_train_w, density=True )
ax[0].hist( sig_train, bins=50, range=(0,1), label='Sig Train', alpha=0.5, ec='none', fc='b', weights=sig_train_w, density=True )

# plot test sample dists
# if you want the error need to track squared weights (probably a better way of doing this)
nb, xe = np.histogram( bkg_test, bins=50, range=(0,1), weights=bkg_test_w )
nb2, _ = np.histogram( bkg_test, bins=50, range=(0,1), weights=bkg_test_w**2 )
nbe = nb2**0.5 / nb
nb, xe = np.histogram( bkg_test, bins=50, range=(0,1), density=True, weights=bkg_test_w )
nbe = nbe * nb

ns, xe = np.histogram( sig_test, bins=50, range=(0,1), weights=sig_test_w )
ns2, _ = np.histogram( sig_test, bins=50, range=(0,1), weights=sig_test_w**2 )
nse = ns2**0.5 / ns
ns, xe = np.histogram( sig_test, bins=50, range=(0,1), density=True, weights=sig_test_w )
nse = nse * ns

cx = 0.5*(xe[1:]+xe[:-1])
ax[0].errorbar( cx, nb, nbe, fmt='rx', label='Bkg Test' ) 
ax[0].errorbar( cx, ns, nse, fmt='bo', label='Sig Test' ) 

# now plot the residual
nbt, xe = np.histogram( bkg_train, bins=50, range=(0,1), weights=bkg_train_w )
nbt2, _ = np.histogram( bkg_train, bins=50, range=(0,1), weights=bkg_train_w**2 )
nbte = nbt2**0.5 / nbt
nbt, xe = np.histogram( bkg_train, bins=50, range=(0,1), density=True, weights=bkg_train_w )
nbte = nbte * nbt

nst, xe = np.histogram( sig_train, bins=50, range=(0,1), weights=sig_train_w )
nst2, _ = np.histogram( sig_train, bins=50, range=(0,1), weights=sig_train_w**2 )
nste = nst2**0.5 / nst
nst, xe = np.histogram( sig_train, bins=50, range=(0,1), density=True, weights=sig_train_w )
nste = nste * nst

db = nb - nbt
dbe = (nbe**2 + nbte**2)**0.5
ds = ns - nst
dse = (nse**2 + nste**2)**0.5
pb = db / dbe
ps = ds / dse

ax[1].axhline(0, c='k', ls='--' )
ax[1].errorbar( cx, pb, np.ones_like(pb), fmt='rx' )
ax[1].errorbar( cx, ps, np.ones_like(ps), fmt='bo' )
ax[1].set_ylabel('Pull')

ax[0].set_xlabel('MVA Score')
ax[0].set_ylabel('Density')
ax[0].legend()
ax[0].set_yscale('log')
fig.tight_layout()
fig.savefig("bdth.png")
fig.savefig("bdth.pdf")

plt.show()


'''
bdtplt.plot_bdt_response(x, "BDT1 response")
responsepath = os.path.join(outputpath, f"BDTh-response.pdf")
plt.savefig(responsepath)
print(f'BDT1 response curve saved to')
print(f"{15*' '}{responsepath}")
plt.close()

bdtplt.plot_roc(bdt, x_test[bdtvars], y_test, w_test)
rocpath = os.path.join(outputpath, f"BDTh-roc.pdf")
plt.savefig(rocpath)
print(f'ROC saved to')
print(f"{15*' '}{rocpath}")
plt.close()



for feature in responsevars:
    bdtplt.plot_feature(x, 0.6, feature)
    figpath = os.path.join(outputpath, f"BDTh-{feature}-withcut.pdf")
    plt.savefig(figpath)
    print(f'Feature plot with bdt cut saved to')
    print(f"{15*' '}{figpath}")
    plt.close()

    bdtpltplot_feature(x, 0, feature)
    figpath = os.path.join(outputpath, f"BDTh-{feature}-nocut.pdf")
    plt.savefig(figpath)
    print(f'Feature plot without cuts saved to')
    print(f"{15*' '}{figpath}")
    plt.close()
'''


'''

#############################
## MAIN
#############################
if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description=f"Trains BDT1 using bdt1_opts from config.py, saves to {cfg.bdt1_opts['outputPath']}")
    method = parser.add_argument("--method",           required=True, choices=['gridsearch', 'fixed-hyperparams'])
    nchunk = parser.add_argument("--nchunks",          default=None,  nargs='*')
    trainf = parser.add_argument("--train-frac",       default=0.75,  type=float)
    savehp = parser.add_argument("--save-hyperparams", default=False, action='store_true')
    loadhp = parser.add_argument("--load-hyperparams", default=False, action='store_true')
    savemo = parser.add_argument("--save-model",       default=False, action='store_true')
    plotre = parser.add_argument("--plot-results",     default=False, action='store_true')

    # Modify the help message here
    method.help = "Method to train the BDT"
    nchunk.help = "Number of chunks to process, default = None (all are processed)"
    trainf.help = "Fraction of data to use for training, default = 0.75"
    savehp.help = f"If using `gridsearch` method, save the optimum hyperparameters to `{os.path.basename(cfg.bdt1_opts['optHyperParamsFile'])}`, default = False"
    loadhp.help = f"If using `fixed-hyperparams` method, use `{os.path.basename(cfg.bdt1_opts['optHyperParamsFile'])}`, default = False"
    savemo.help = f"Save model to `{os.path.basename(cfg.bdt1_opts['jsonPath'])}` and `{os.path.basename(cfg.bdt1_opts['mvaPath'])}`, default = False"
    plotre.help = f"Plot and save BDT responses to `{os.path.basename(cfg.bdt1_opts['jsonPath'])}` and `{os.path.basename(cfg.bdt1_opts['mvaPath'])}`, default = False"

    args = parser.parse_args()

    #############################
    ## PREPROCESSING
    #############################
    start = time()
    print(f"{30*'-'}")
    print(f"BDT1 TRAINING")
    print(f"{30*'-'}\n")
    print("Initialising...")

    # Load configuration
    plt.style.use(os.path.abspath(os.path.join(cfg.FCCAnalysesPath, 'fcc.mplstyle')))
    inputpath    = check_inputpath(cfg.bdt1_opts['inputPath'])
    outputpath   = set_outputpath(cfg.bdt1_opts['outputPath'])
    yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])
    bdtvars      = vars_fromyaml(yamlpath, cfg.bdt1_opts['mvaBranchList'])
    # Variables not used by the bdt which you want to plot
    responsevars = ["EVT_hemisEmin_Emiss"]

    # Efficiencies of pre-selection cuts and branching fractions
    bfs = {sample: cfg.branching_fractions[sample][0] for sample in cfg.samples}
    eff = {sample: cfg.efficiencies[cfg.bdt1_opts['efficiencyKey']][sample][0] for sample in cfg.samples}

    print(f"----> INFO: Efficiencies loaded from config.py using key `{cfg.bdt1_opts['efficiencyKey']}`")
    print(f"----> INFO: Using {cfg.bdt1_opts['mvaBranchList']} from")
    print(f"{15*' '}{yamlpath}")
    print(f"----> INFO: Loading files from")
    print(f"{15*' '}{inputpath}")
    print(f"----> INFO: Output will be saved to")
    print(f"{15*' '}{outputpath}")

    paths = {sample: os.path.join(inputpath, sample, "*.root") for sample in cfg.samples}

    # Depending on the value passed to --nchunks:
    # Single int: use this number of chunks for every sample
    # len(cfg.samples) ints: use corresponding number
    # None or mismatched length or other: use all chunks
    if (args.nchunks is not None):
        if (len(args.nchunks) == 1):
            files = {sample: glob(paths[sample])[:int(args.nchunks[0])] for sample in cfg.samples}
        elif (len(args.nchunks) == len(cfg.samples)):
            files = {sample: glob(paths[sample])[:int(args.nchunks[i])] for i, sample in enumerate(cfg.samples)}
        warn_about_slowGridSearch = 0
    else:
        print(f"----> INFO: --nchunks is None or invalid, skipping...")
        files = {sample: glob(paths[sample]) for sample in cfg.samples}
        warn_about_slowGridSearch = 1

    x = {sample: None for sample in cfg.samples}
    y = {sample: None for sample in cfg.samples}
    w = {sample: None for sample in cfg.samples}

    for sample in cfg.samples:
        # Weigh each sample by the number of expected events of that sample
        weight = 6e12*eff[sample]*bfs[sample]
        if sample in cfg.sample_allocations['Bssignal']:
            weight *= 2*bfs['p8_ee_Zbb_ecm91']*cfg.prod_frac['Bs']
            x[sample], y[sample], w[sample] = load_data(files[sample], 'Bssignal', weight, bdtvars+responsevars)
        else:
            x[sample], y[sample], w[sample] = load_data(files[sample], 'hadronic_background', weight, bdtvars+responsevars)

    for sample in cfg.samples:
        if (sample in cfg.sample_allocations['Bssignal']) and (np.any(y[sample].to_numpy() == 0)):
            raise ValueError
        elif (sample not in cfg.sample_allocations['Bssignal']) and (np.any(y[sample].to_numpy() == 1)):
            raise ValueError

    rand_state = 7  # For reproducibility
    x_train = pd.concat([x[sample].sample(frac=args.train_frac, random_state=rand_state) for sample in cfg.samples], copy=True, ignore_index=True)
    y_train = pd.concat([y[sample].sample(frac=args.train_frac, random_state=rand_state) for sample in cfg.samples], copy=True, ignore_index=True)
    # Normalise weights by dividing by the total number of events of each type
    w_train = pd.concat([w[sample].sample(frac=args.train_frac, random_state=rand_state)/(args.train_frac*len(x[sample])) for sample in cfg.samples], copy=True, ignore_index=True)

    # Create test dataframes by sampling the indices not used in x/y/w_train
    x_test = pd.concat([x[sample].drop(x[sample].sample(frac=args.train_frac, random_state=rand_state).index) for sample in cfg.samples], copy=True, ignore_index=True)
    y_test = pd.concat([y[sample].drop(y[sample].sample(frac=args.train_frac, random_state=rand_state).index) for sample in cfg.samples], copy=True, ignore_index=True)
    # Normalise weights by dividing by the total number of events of each type
    w_test = pd.concat([w[sample].drop(w[sample].sample(frac=args.train_frac, random_state=rand_state).index)/((1-args.train_frac)*len(w[sample])) for sample in cfg.samples], copy=True, ignore_index=True)

    print(f"\n{30*'-'}\n")
    for sample in cfg.samples:
        print(f"Number of {sample:31} events = {x[sample].shape[0]:>8} using {len(files[sample]):>4} chunks")
    print("\n----> INFO: Preprocessing done")
    print(f"{15*' '}Using {len(x_train):>8} events to train ({100*args.train_frac:.1f}% of total)")
    print(f"{15*' '}Using {len(x_test):>8} events to  test ({100*(1-args.train_frac):.1f}% of total)")
    print(f"\n{30*'-'}\n")

    #############################
    ## TRAINING
    #############################
    # Define BDT
    bdt = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric="auc", n_jobs=-1, objective='binary:logistic')
    print(f"BDT OUTPUT")
    # Tuning hyperparameters
    if args.method == 'gridsearch':
        param_grid = {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.1, 0.3],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "min_child_weight": [1, 5],
        }
        if warn_about_slowGridSearch:
            print(f"----> WARNING: `gridsearch` method is used without specifying --nchunks, hyperparameter tuning may take a long time")
        # Use GridSearchCV to find the best hyperparameters
        # Need to convert to numpy to save to ROOT TMVA file
        grid_search = GridSearchCV(estimator=bdt, param_grid=param_grid, cv=4, scoring="roc_auc", verbose=1, n_jobs=-1)
        grid_search.fit(x_train[bdtvars].to_numpy(), y_train.to_numpy(), sample_weight=w_train.to_numpy(),
                        eval_set = [(x_test[bdtvars].to_numpy(), y_test.to_numpy())], sample_weight_eval_set = [w_test.to_numpy()], verbose=False)

        print(f"Best cross validation score = {grid_search.best_score_:.5f}")
        print(f"Best params = {grid_search.best_params_}\n")

        bdt = grid_search.best_estimator_
        cv_dict = grid_search.best_params_

        ### Save optimum combination of hyperparameters
        if args.save_hyperparams:
            with open(cfg.bdt1_opts['optHyperParamsFile'], 'w') as outfile:
                dump(cv_dict, outfile)
            print(f"----> INFO: Optimum hyperparameters saved to")
            print(f"{15*' '}{cfg.bdt1_opts['optHyperParamsFile']}")
        else:
            print(f"----> INFO: --save-hyperparams not set, skipping...")

    # Using fixed hyperparameters
    elif args.method == 'fixed-hyperparams':

        ### Load hyperparameters
        if args.load_hyperparams:
            with open(cfg.bdt1_opts['optHyperParamsFile'], 'r') as stream:
                config_dict = safe_load(stream)
            print(f"----> INFO: Loading hyperparameters from")
            print(f"{15*' '}{cfg.bdt1_opts['optHyperParamsFile']}")
        else:
            config_dict = {'gamma': 0.2, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 1.0, 'max_depth': 4}
            config_dict = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 7, "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.2, "min_child_weight": 1}
            print(f"----> INFO: --load-hyperparams not set, using default values")
            print(f"{15*' '}{config_dict}")

        bdt.set_params(**config_dict)
        bdt.fit(x_train[bdtvars].to_numpy(), y_train.to_numpy(), sample_weight=w_train.to_numpy(),
                eval_set = [(x_test[bdtvars].to_numpy(), y_test.to_numpy())], sample_weight_eval_set=[w_test.to_numpy()], verbose=100)
        cv_dict = config_dict

    # xgb.cv does not use n_estimators, num_boost_round=50 instead
    cv_dict.pop('n_estimators', None)

    #############################
    ## VALIDATION
    #############################
    data_dmatrix = xgb.DMatrix(data=x_train[bdtvars].to_numpy(), label=y_train.to_numpy(), weight=w_train.to_numpy())
    xgb_cv = xgb.cv(dtrain=data_dmatrix, params=cv_dict, nfold=5, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
    print('\n', xgb_cv.head())
    print(f'...')
    print(xgb_cv.tail())
    print(f"{30*'-'}\n")

    #############################
    ## SAVING MODEL
    #############################
    print(f"SAVING")
    if args.save_model:
        bdt.save_model(cfg.bdt1_opts['jsonPath'])
        ROOT.TMVA.Experimental.SaveXGBoost(bdt, cfg.bdt1_opts['mvaRBDTName'], cfg.bdt1_opts['mvaPath'], num_inputs=len(bdtvars))
        print(f"----> INFO: Model saved to")
        print(f"{15*' '}1. {cfg.bdt1_opts['jsonPath']}")
        print(f"{15*' '}2. {cfg.bdt1_opts['mvaPath']}")

    else:
        print(f"----> INFO: --save-model flag not set, skipping model saving...")

    feature_importances = pd.DataFrame(bdt.feature_importances_,
                                       index = bdtvars,
                                       columns=['importance']).sort_values('importance', ascending=False)

    print(f"\n{30*'-'}\n")
    print(f"FEATURE IMPORTANCES")
    print(feature_importances)
    print(f"\n{30*'-'}\n")
    print("PLOTTING")

    #############################
    ## RESPONSE PLOTS
    #############################
    if args.plot_results:
        for df in x.values():
            df['XGB'] = bdt.predict_proba(df[bdtvars])[:, 1]

        if args.method == 'gridsearch':
            prefix = 'bdt1-grid-'
        else:
            prefix = 'bdt1-'

        plot_bdt_response(x, "BDT1 response")
        responsepath = os.path.join(outputpath, f"{prefix}response.pdf")
        plt.savefig(responsepath)
        print(f'BDT1 response curve saved to')
        print(f"{15*' '}{responsepath}")
        plt.close()

        plot_roc(bdt, x_test[bdtvars], y_test, w_test)
        rocpath = os.path.join(outputpath, f"{prefix}roc.pdf")
        plt.savefig(rocpath)
        print(f'ROC saved to')
        print(f"{15*' '}{rocpath}")
        plt.close()

        plot_punzi_significance(x, np.linspace(0, 1, 20), 5, eff)
        punzipath = os.path.join(outputpath, f"{prefix}punzi.pdf")
        plt.savefig(punzipath)
        print(f'Punzi significance plot saved to')
        print(f"{15*' '}{punzipath}")
        plt.close()

        plot_significance(x, [0.2, 0.6, 0.8, 0.9, 0.99], np.logspace(-9, -4, 100))
        significancepath = os.path.join(outputpath, f"{prefix}significance.pdf")
        plt.savefig(significancepath)
        print(f'Significance plot saved to')
        print(f"{15*' '}{significancepath}")
        plt.close()

        for feature in responsevars:
            plot_feature(x, 0.6, feature)
            figpath = os.path.join(outputpath, f"{prefix}{feature}-withcut.pdf")
            plt.savefig(figpath)
            print(f'Feature plot with bdt cut saved to')
            print(f"{15*' '}{figpath}")
            plt.close()

            plot_feature(x, 0, feature)
            figpath = os.path.join(outputpath, f"{prefix}{feature}-nocut.pdf")
            plt.savefig(figpath)
            print(f'Feature plot without cuts saved to')
            print(f"{15*' '}{figpath}")
            plt.close()

    else:
        print(f"---->INFO: --plot-results flag not set, skipping plotting...")

    end = time()
    print(f"\n{30*'-'}")
    print(f"Execution time  = {timedelta(seconds=end-start)}")
    print(f"{30*'-'}")
'''