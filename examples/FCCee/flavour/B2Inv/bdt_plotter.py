import os
import glob
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
import json
from time import time
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc


import config as cfg
import efficiency_finder
import variable_plotter as vp


# Function to load the BDT model from a JSON file
def load_bdt_model(json_path):
    bdt_model = xgb.Booster()
    bdt_model.load_model(json_path)
    return bdt_model

# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars


def dec2type( dec ):
    if dec in ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"]:
        return "signal"
    elif dec in ["p8_ee_Zbb_ecm91", "p8_ee_Zcc_ecm91"]:
        return "heavy"
    elif dec in ["p8_ee_Zss_ecm91","p8_ee_Zud_ecm91"]:
        return "light"
    elif dec in ["p8_ee_Zee_ecm91", "p8_ee_Zmumu_ecm91", "p8_ee_Ztautau_ecm91"]:
        return "lepton"
    else:
        return "unknown"


###################################
## Define functions for plotting ## - KEY PLOTTERS
###################################

def plot_simple_ROC(df,bdt_name = "BDTh",output_file_name = "ROC",outpath=None):
# make roc curve (use test sample)
    y_test = df[ df["sample"]==1]["label"]
    x_test_bdt = df[df["sample"]==1]["bdt_score"]
    w_test =  df[ df["sample"]==1]["total_weight"]

    fpr, tpr, thresholds = roc_curve( y_test, x_test_bdt, sample_weight=w_test)
    roc_auc = auc(fpr, tpr)
    print("AUC = ", roc_auc)

    fig, ax = plt.subplots()
    ax.plot( tpr, 1-fpr, lw=1 )
    ax.set_xlim(0.9,1)
    ax.set_ylim(0.9,1)
    ax.set_xlabel( "Signal Efficiency" )
    ax.set_ylabel( "Background Rejection (1-fpr)" )
    ax.text(0.905, 0.905, f'AUC: {roc_auc:.5f}')
    ax.grid(visible=True, which='major', alpha = 0.4)
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    fig.tight_layout()

    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f'{bdt_name}_{output_file_name}.pdf')




def plot_ROC_star(df, bdt_name = "BDTh",
                  decay_label_in_df="decay",
                  bdtscore_label="bdt_score",
                  weight_label="total_weight", 
                  comps = [("signal", "heavy"), ("signal", "light"), ("heavy", "light"), ("signal", "background")],
                  output_file_name = "ROC_star",
                  outpath=None):
    
    # make roc curve (use test sample)
    
    df["Type"] = df[decay_label_in_df].apply(dec2type)
    fig, ax = plt.subplots()
    for type1, type2 in comps:
        # if background just sum over everything that isn't signal
        if "background" in [type1, type2]:
            subf = df[df["sample"]==1].copy() # only use test data
            subf.loc[ subf["Type"] != "signal", "Type" ] = "background"
        else:
            subf = df[ (df["Type"]==type1) | (df["Type"]==type2 ) ] 

        y_true = subf["Type"].map( { type1: 1, type2: 0} ).values 
        y_pred = subf[bdtscore_label].values
        weight = subf[weight_label].values

        fpr, tpr, thresholds = roc_curve( y_true, y_pred, sample_weight=weight )
        roc_auc = auc(fpr, tpr)
        print( f"{type1} vs {type2} AUC =", roc_auc )

        ax.plot( tpr, 1-fpr, lw=1, label= f"{type1} vs {type2} ({roc_auc:.5f})" )
    ax.legend()
    ax.set_xlabel( "Signal Efficiency" )
    ax.set_ylabel( "Background Rejection (1-fpr)" )
    
    ax.plot( [1,0], [0,1], color='k', ls='--', lw=1, alpha=0.6 )
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.png'))
        fig.savefig(os.path.join(outpath,f"{bdt_name}_{output_file_name}.pdf"))
    else:
        #fig.savefig(f'{bdt_name}_{output_file_name}.png')
        fig.savefig(f"{bdt_name}_{output_file_name}.pdf")
    ax.set_xticks(np.arange(0.8,1.05, 0.05))
    ax.set_yticks(np.arange(0.8,1.05, 0.05))
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.7, alpha=0.4)
    ax.set_xlim(0.8,1.005)
    ax.set_ylim(0.8,1.005)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outputpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f"f'{bdt_name}_{output_file_name}_zoomedin.pdf")




def plot_bdt_response(df, bdt_name = "BDTh",output_file_name = "response" ,outpath=None):
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

    ax[0].set_xlabel(f'{bdt_name} XGBoost Score')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].set_yscale('log')
    fig.tight_layout()

    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outputpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f"{bdt_name}_{output_file_name}.pdf")



# efficiency plot (on total sample)
def plot_eff(df, bdt_name = "BDTh",output_file_name = 'efficiency_plot',outpath=None):
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
    ax.set_xlabel(f'{bdt_name} XGBoost Score')
    ax.set_ylabel('Efficiency')
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', linestyle='-', color='0.7', linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outputpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f'{bdt_name}_{output_file_name}.pdf')
    #ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.set_xlim(0.95,1.002)
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outputpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f'{bdt_name}_{output_file_name}_zoomedin.pdf')


def post_bdt_variable_plot(df,variable,
                           bdt_cut, #must be string of correct format as in vp
                           bdt_name = "BDTh",
                           weight=True,
                           density=False,
                           signal_bf=1e-6, #for Bs
                           components=["Bssignal", "hadronic_background"],outpath=None):
    if outpath:
        savepath=os.path.join(outpath,f'{variable}_with_{bdt_name}_cut_{bdt_cut}.pdf')
    else:
        savepath= f'{variable}_with_{bdt_name}_cut_{bdt_cut}.pdf'

    vp.plot(varname=variable,cut=bdt_cut, data=df,weight=weight,density=density,signal_bf=signal_bf, components=components,save=savepath)


#######################################################
# Load BDT and apply to loaded data - define as funtion
#######################################################

def load_bdt_and_apply(pickled_df_fname = "bdth_dataframe.pkl", 
                        config_bdtopts = cfg.bdth_opts,
                        training_round = "baseline",
                        hps_dict_name = "default-hps",
                        features_list_name = "baseline-bdth-vars"): # hps dict and features_list_name specift BDT used
    
    
    #path to data and outputs
    outputpath   = config_bdtopts['outputPath']
    yamlpath = cfg.fccana_opts['yamlPath']

    #Getting BDT vars for training from yaml
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDTh_{hps_dict_name}_{features_list_name}'

    #path to pickled data
    pickled_df_path = os.path.join(outputpath, pickled_df_fname)

    #path to saved bdt
    bdt_json_path = os.path.join(outputpath,training_round,f"{bdtname}.json")

    # Load the BDT model
    try:
        print("Loading BDT model...")
        bdt_model = load_bdt_model(bdt_json_path)
        print("BDT model loaded successfully.")
    except Exception as e:
        print(f"Error loading BDT model: {e}")
        quit()

    # Read df saved to pickle and add BDT 

    #Getting BDT vars for training from yaml
    # load pickled df
    df = pd.read_pickle(pickled_df_path)

    #add bdt score
    df["bdt_score"] =  bdt_model.predict(xgb.DMatrix(df[bdtvars])) #Add BDT1 score

    return bdt_model, bdtname, df

#########################################
## Make some plots 
########################################

model, bdtname, df = load_bdt_and_apply( pickled_df_fname = "bdth_dataframe.pkl", 
                        config_bdtopts = cfg.bdth_opts,
                        training_round = "baseline",
                        hps_dict_name = "default-hps",
                        features_list_name = "baseline-bdth-vars")

outputpath = os.path.join(cfg.bdth_opts['outputPath'],"baseline")

#plot_simple_ROC(df, bdt_name = bdtname,outpath=outputpath)
#plot_bdt_response(df,bdt_name = bdtname,outpath=outputpath)
#plot_eff(df,bdt_name = bdtname, outpath=outputpath)
#plot_ROC_star(df,bdt_name = bdtname, outpath=outputpath)
#post_bdt_variable_plot(df,variable='EVT_e',bdt_cut='bdt_score>0.9',bdt_name = bdtname, outpath=outputpath,weight=False,density=True)












####################################### old - to delete when sure dont need #########################################################

'''
#Section needed as currently not got saved df :(
# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)

    return bdtvars

#path to data and outputs
inputpath    = cfg.bdth_opts['inputPath']
outputpath   = cfg.bdth_opts['outputPath']
yamlpath     = cfg.fccana_opts['yamlPath']

#Getting BDT vars for training from yaml
bdtvars_list = cfg.bdth_opts['mvaBranchList']
bdtvars      = vars_fromyaml(yamlpath, bdtvars_list)
responsevars = ["EVT_hemisEmin_Emiss"] # Variables not used by the bdt which you want to plot

# branching fractions for weights
branching_fractions = cfg.branching_fractions #dictionary containing decay name and tuple with BF and its erro

#getting training decays from config decay list
signal_decays =  cfg.bdth_opts["signalAllocation"]
background_decays =  cfg.bdth_opts["backgroundAllocation"]
training_decays = signal_decays + background_decays # Not including the Z->ee, Z->mumu, Z->tautau decays in the training but we still want to process them to see how the BDT does

decays_dict={'signal':signal_decays,'background':background_decays}

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
    sub_df["bdt_score"] =  bdt_model.predict(xgb.DMatrix(sub_df[bdtvars])) #Add BDT1 score
    df_dict[decay] = sub_df


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
'''








'''


##############################
## Ritwik Plotters
##############################

def plot_punzi_significance(x, cuts, sigma, eff_before_bdt):
    B = np.array([])
    epsilon_s = np.array([])
    for cut in cuts:
        temp_B = 0
        for sample in x:
            if sample in cfg.sample_allocations['Bssignal']:
                before = len(x[sample])
                after  = len(x[sample].query(f"XGB > {cut}"))
                epsilon_s = np.append(epsilon_s, after/before)
            else:
                before = len(x[sample])
                after  = len(x[sample].query(f"XGB > {cut}"))
                count = 6e12*cfg.branching_fractions[sample][0]*eff_before_bdt[sample]*after/before
                temp_B += count

        B = np.append(B, temp_B)

    fig, ax = plt.subplots()
    punzi = np.divide(epsilon_s, np.sqrt(B) + sigma/2)
    ax.plot(cuts, punzi)
    ax.axvline(cuts[np.argmax(punzi)], color='black',
               linestyle='--', label = f'Optimal cut ({cuts[np.argmax(punzi)]:.3f})')
    ax.set_xlabel('BDT1 cut value')
    ax.set_ylabel(r'$\frac{\epsilon_s}{\sqrt{B} + \sigma/2},\ \sigma = $'+f'{sigma}')
    ax.set_title('Punzi significance for BDT1')
    ax.legend(loc='best')
    fig.tight_layout()


# Plots shaded area between S/sqrt(S+B) assuming the signal error is sqrt(S) and the background error is sqrt(B)
def plot_significance(x, bdtvals, branching_fracs, eff_before_bdt):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for bdt in bdtvals:
        S = np.array([])
        B = 0
        for sample in x:
            if sample in cfg.sample_allocations['Bssignal']:
                before = len(x[sample])
                after  = len(x[sample].query(f"XGB > {bdt}"))
                temp_S = 6e12*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*2*cfg.prod_frac['Bs']*eff_before_bdt[sample]*after/before*branching_fracs
                S = np.append(S, temp_S)
            else:
                before = len(x[sample])
                after  = len(x[sample].query(f"XGB > {bdt}"))
                count = 6e12*cfg.branching_fractions[sample][0]*eff_before_bdt[sample]*after/before
                B += count

        significance = np.divide(S, np.sqrt(S+B+1e-10))
        ax.plot(branching_fracs, significance, label=f'BDT1 $>$ {bdt:.3f}')

    ax.axvline(1e-6, color='black', linestyle='--')
    ax.set_xlabel(r'$\mathcal{B}(B_s\to\nu\bar{\nu})$')
    ax.set_ylabel(r'$\frac{S}{\sqrt{S+B}}$')
    ax.set_ylim(0, 10)
    ax.legend(loc='best')
    ax.set_title('BDT1 significance (expected "signal to noise" ratio)')
    fig.tight_layout()




'''