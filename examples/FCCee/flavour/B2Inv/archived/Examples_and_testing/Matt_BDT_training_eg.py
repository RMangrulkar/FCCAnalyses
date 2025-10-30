import os
import glob
from tabulate import tabulate
import uproot
import ROOT as r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve, auc

# where the files live
# these should eventually have our "new" preselection an hopefully around 500K of each class

# note that we are not going to include the Z->ee, Z->mumu, Z->tautau samples in the training
# but we still want to process them to see how the BDT does
path = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts_incl_EVT_hemisEmin40"

# all this stuff should really live in config files 
branching_fractions = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 0.01, # put this as a dummy value so the weight values come out near the background
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": 0.01,
    "p8_ee_Zbb_ecm91": 0.1512,
    "p8_ee_Zcc_ecm91": 0.1203,
    "p8_ee_Zss_ecm91": 0.1584,
    "p8_ee_Zud_ecm91": 0.2701,
    "p8_ee_Zee_ecm91": 0.033632,
    "p8_ee_Zmumu_ecm91": 0.033662,
    "p8_ee_Ztautau_ecm91": 0.033696,
}

# list of input vars
bdt_input_var = [ 
    "EVT_Thrust_mag",
    "EVT_Thrust_deltaE",
    "EVT_hemisEmin_e",
    "EVT_hemisEmin_eCharged",
    "EVT_hemisEmin_eNeutral",
    "EVT_hemisEmin_n",
    "EVT_hemisEmin_nCharged",
    "EVT_hemisEmin_nNeutral",
    "EVT_hemisEmin_nDV",
    "EVT_hemisEmax_e",
    "EVT_hemisEmax_eCharged",
    "EVT_hemisEmax_eNeutral",
    "EVT_hemisEmax_n",
    "EVT_hemisEmax_nCharged",
    "EVT_hemisEmax_nNeutral",
    "EVT_hemisEmax_nDV",
    "Rec_track_n",
    "Rec_PV_ntracks",
    "Rec_vtx_n",
    "PV_Rec_vtx_m"
]

# collate the files in a dictionary (should be defined in a module only once)
fdict = {}
for file in glob.glob( f"{path}/*/chunk*_withMVA1.root" ):
    dec = file.split("/")[-2]
    fname = file.split("/")[-1]
    if dec not in fdict.keys():
        fdict[dec] = []
    fdict[dec].append( fname )

# track events which pass presel (should be defined in a module only once)
evdict = {}
for dec, files in fdict.items():
    print(dec)
    totProc = 0
    totSele = 0
    for file in files:
        tf = r.TFile( f"{path}/{dec}/{file}")
        evsProc = tf.Get("eventsProcessed").GetVal()
        evsSele = tf.Get("eventsSelected").GetVal()
        totProc += evsProc
        totSele += evsSele
        print( f"  {file:20s} - {evsSele:8d} / {evsProc:8d}" )
    print( f"TOTAL = {totSele:8d} / {totProc:8d}" )
    evdict[dec] = (totSele, totProc)

# now collect relevant events into a dataframe
df_list = []
for dec in fdict.keys():
    eff = evdict[dec][0] / evdict[dec][1]
    weight = branching_fractions[dec] * eff

    subf = uproot.concatenate( f"{path}/{dec}/chunk*_withMVA1.root:events", bdt_input_var, library="pd" )
    subf["Decay"] = dec
    subf["Weight"] = weight / len(subf) * 1e7 # the 1e7 keeps weights near O(1)
    
    df_list.append( subf )

df = pd.concat( df_list, ignore_index=True )

# want to make sure that integer types are actually set as integers which uproot doesn't do
integer_branches = [
    "EVT_hemisEmin_n",
    "EVT_hemisEmin_nCharged",
    "EVT_hemisEmin_nNeutral",
    "EVT_hemisEmin_nDV",
    "EVT_hemisEmax_n",
    "EVT_hemisEmax_nCharged",
    "EVT_hemisEmax_nNeutral",
    "EVT_hemisEmax_nDV",
    "Rec_track_n",
    "Rec_PV_ntracks",
    "Rec_vtx_n",
]

for integer_branch in integer_branches:
    df[integer_branch] = df[integer_branch].astype(np.int32)

# now label background and signal events as 0 and 1 which is what the classifier wants
def sorb( dec ):
    if dec in ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"]:
        return 1
    else:
        return 0

df["label"] = df["Decay"].apply(sorb)

# going to print the sum of weights for each now
# can check this is consistent with BF * eff (which it should be)
print("Sum of weights:")
print_rows=[]
for dec in df["Decay"].unique():
    sumw = df[ df["Decay"]==dec ]["Weight"].sum()
    nevs = len(df[ df["Decay"]==dec ])
    print_rows.append( [ dec, sumw, nevs ] )

print( tabulate( print_rows, headers=["Decay", "SumWeights", "NumEvents"] ) )

# now we can actually do some training

# we currently have a bit of an inbalance in our samples
# good practise to get an additional weight which balances the classes
# i.e. accounts for the fact that our training sample contains more backgorund than signal
y = df["label"].values
balancing_weights = compute_sample_weight("balanced", y)
df["bdt_weight"] = df["Weight"] * balancing_weights

# now shuffle the whole dataframe around to avoid any funny biases
# do this with a random seed so it's reproducible
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# now create labels for the train, test and validation split
# we will give them indices train==0, test==1, validation==2
# use a random seed for this so it's reproducible
np.random.seed(210187)
sample_indices = np.random.choice( [0,1,2], p=[0.75, 0.125, 0.125], size=len(df) )
df["sample"] = sample_indices

# matrix of input vars
X_train = df[ df["sample"]==0][ bdt_input_var ]
X_test  = df[ df["sample"]==1][ bdt_input_var ]
X_valid = df[ df["sample"]==2][ bdt_input_var ]

# array of target
y_train = df[ df["sample"]==0][ "label" ]
y_test  = df[ df["sample"]==1][ "label" ]
y_valid = df[ df["sample"]==2][ "label" ]

# array of weights
w_train = df[ df["sample"]==0][ "bdt_weight" ]
w_test  = df[ df["sample"]==1][ "bdt_weight" ]
w_valid = df[ df["sample"]==2][ "bdt_weight" ]

### AT THIS POINT YOU CAN PROBABLY DO FANCIER
### STUFF WITH CROSS VALIDATION
### for example
# from sklearn.model_selection import StratifiedKFold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
# for train, test in cv.split(X, y):
#    X_train = df[ bdt_input_var ].values[train]
#    X_test  = df[ bdt_input_var ].values[test]
#    y_train = df[ "label" ].values[train]
#    y_test  = df[ "label" ].values[test]
#    w_train = df[ "bdt_weight"].values[train]
#    w_test  = df[ "bdt_weight"].values[test]
#    # fit
#    bdt = XGBClassifier()
#    bdt.fit( X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], sample_weight_eval_set=[w_test] )


# now let's train it
bdt = xgb.XGBClassifier( n_estimators=400,
                         max_depth=3,
                         learning_rate=0.1 ) 

print("Training model")
bdt.fit( X_train, y_train, 
         sample_weight=w_train, 
         eval_set=[(X_test, y_test)], 
         sample_weight_eval_set=[w_test], 
         verbose=10 )

# now put it's predictions back into the frame
df['bdt_score'] = bdt.predict_proba( df[ bdt_input_var ] )[:,1]

# get the feature importance
importance_indices = np.argsort( bdt.feature_importances_ )
sorted_features = bdt.feature_names_in_[ importance_indices[::-1] ]
sorted_importances = bdt.feature_importances_[ importance_indices[::-1] ]
print( "Feature Importance:" )
print( tabulate( zip( sorted_features, sorted_importances ) ) )

# save the model to a file for use later
bdt.save_model( "bdt.json" )
# and in ROOT format
# r.TMVA.Experimental.SaveXGBoost(bdt, "MyBDT", "bdt.root", num_inputs=len(bdt_input_var))

# make roc curve (use test sample)
fpr, tpr, thresholds = roc_curve( y_test, bdt.predict_proba( X_test )[:,1], sample_weight=w_test )
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
for decay in df["Decay"].unique():
    subf = df[ df["Decay"]==decay ]
    mva_scores = subf["bdt_score"].values
    weights = subf["Weight"].values

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
sig_train_w = df[ (df["sample"]==0) & (df["label"]==1) ]["Weight"].values
bkg_train_w = df[ (df["sample"]==0) & (df["label"]==0) ]["Weight"].values
sig_test_w = df[ (df["sample"]==1) & (df["label"]==1) ]["Weight"].values
bkg_test_w = df[ (df["sample"]==1) & (df["label"]==0) ]["Weight"].values

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
fig.savefig("bdt.png")
fig.savefig("bdt.pdf")

plt.show()
