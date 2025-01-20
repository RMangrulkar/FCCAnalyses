# xgb.py
# LEGACY -- Julio's code
import numpy as np
import ROOT
import random
import sys,os, argparse
import json
import joblib
import glob
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

ROOT.EnableImplicitMT()


def plot_bdt_comparison(var):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    _, bins, _ = plt.hist(df_signal_pd[var], bins=100,
                          histtype='step', label=r'$B_s \rightarrow \nu \bar{\nu}$',density=1)
    _, bins, _ = plt.hist(df_bb_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow b \bar{b}$',density=1)
    _, bins, _ = plt.hist(df_cc_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow c \bar{c}$',density=1)
    _, bins, _ = plt.hist(df_ss_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow s \bar{s}$',density=1)
    _, bins, _ = plt.hist(df_ud_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow q \bar{q}$', density=1)
    plt.xlabel("BDT response")
    plt.ylabel("Normalised counts")
    #plt.xlim(0,60)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='upper center')

    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax.text(0.725, 0.4, transform=ax.transAxes, bbox=props)

def plot_comparison(var):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    _, bins, _ = plt.hist(df_signal_pd[var], bins=100,
                          histtype='step', label=r'$B_s \rightarrow \nu \bar{\nu}$',density=1)
    _, bins, _ = plt.hist(df_bb_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow b \bar{b}$',density=1)
    _, bins, _ = plt.hist(df_cc_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow c \bar{c}$',density=1)
    _, bins, _ = plt.hist(df_ss_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow s \bar{s}$',density=1)
    _, bins, _ = plt.hist(df_ud_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow q \bar{q}$',density=1)
    plt.xlabel(var)
    #plt.xlim(0,60)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='upper right')

def plot_comparison_cut(var):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    _, bins, _ = plt.hist(df_signal_pd_cut[var], bins=100,
                          histtype='step', label=r'$B_s \rightarrow \nu \bar{\nu}$',density=1)
    _, bins, _ = plt.hist(df_bb_pd_cut[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow b \bar{b}$',density=1)
    _, bins, _ = plt.hist(df_cc_pd_cut[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow c \bar{c}$',density=1)
    _, bins, _ = plt.hist(df_ss_pd_cut[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow s \bar{s}$',density=1)
    _, bins, _ = plt.hist(df_ud_pd_cut[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow q \bar{q}$',density=1)
    plt.xlabel(var)
    #plt.xlim(0,60)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='upper right')

def plot_roc(bdt, x_test, y_test):
    y_score = bdt.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    area = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {area:.4f})')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')

def plot_significance(bdt, x_test, y_test):
    y_score = bdt.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    #n_sig = 1200
    #n_bkg = 23000
    n_sig = 90000
    n_bkg = 60000
    S = n_sig*tpr
    B = n_bkg*fpr
    epsilon = 1e-10
    metric = S/np.sqrt(S+B+epsilon)

    plt.plot(thresholds, metric, label=f'XGB')
    plt.xlabel('$P_{XGB}$ cut value')
    plt.ylabel('Signal significance: $\\frac{S}{\\sqrt{S+B}}$')
    plt.xlim(0, 1)

    optimal_cut = thresholds[np.argmax(metric)]
    plt.axvline(optimal_cut, color='black', linestyle='--', label = f'Optimal cut ({optimal_cut:.3f})')


path = 'root://eospublic.cern.ch//eos/experiment/fcc/ee/analyses_storage/flavour/B2Inv/'

listvars = {"EVT_ThrustEmin_E", 
            "EVT_ThrustEmax_E",
            "EVT_ThrustEmin_Echarged",
            "EVT_ThrustEmax_Echarged",
            "EVT_ThrustEmin_Eneutral",
            "EVT_ThrustEmax_Eneutral",
            "EVT_ThrustEmin_Ncharged",
            "EVT_ThrustEmax_Ncharged",
            "EVT_ThrustEmin_Nneutral",
            "EVT_ThrustEmax_Nneutral",
            "EVT_NtracksPV",
            "EVT_NVertex",
            "EVT_ThrustEmin_NDV",
            "EVT_ThrustEmax_NDV",
            "EVT_dPV2DVmin",
            "EVT_dPV2DVmax",
            "EVT_dPV2DVave",
            "EVT_Thrust_Mag",
            "EVT_Thrust_X", 
            "EVT_Thrust_Y", 
            "EVT_Thrust_Z"} 

df_signal = ROOT.RDataFrame('events',path+'Bs2NuNuSig.root',listvars)
df_bb = ROOT.RDataFrame('events',path+'ZbbBkg.root',listvars)
df_cc = ROOT.RDataFrame('events',path+'ZccBkg.root')
df_ss = ROOT.RDataFrame('events',path+'ZssBkg.root')
df_ud = ROOT.RDataFrame('events',path+'ZudBkg.root')

df_signal = df_signal.Define("EVT_ThrustDiff_E","EVT_ThrustEmax_E-EVT_ThrustEmin_E")
df_bb = df_bb.Define("EVT_ThrustDiff_E","EVT_ThrustEmax_E-EVT_ThrustEmin_E")
df_cc = df_cc.Define("EVT_ThrustDiff_E","EVT_ThrustEmax_E-EVT_ThrustEmin_E")
df_ss = df_ss.Define("EVT_ThrustDiff_E","EVT_ThrustEmax_E-EVT_ThrustEmin_E")
df_ud = df_ud.Define("EVT_ThrustDiff_E","EVT_ThrustEmax_E-EVT_ThrustEmin_E")

#cols = ['EVT_ThrustEmin_E', 'EVT_ThrustEmax_E']

cols = ["EVT_ThrustEmin_E",
        "EVT_ThrustEmax_E",
        "EVT_ThrustEmin_Echarged",
        "EVT_ThrustEmax_Echarged",
        "EVT_ThrustEmin_Eneutral",
        "EVT_ThrustEmax_Eneutral",
        "EVT_ThrustEmin_Ncharged",
        "EVT_ThrustEmax_Ncharged",
        "EVT_ThrustEmin_Nneutral",
        "EVT_ThrustEmax_Nneutral",
        "EVT_NtracksPV",
        "EVT_NVertex",
        "EVT_ThrustEmin_NDV",
        "EVT_ThrustEmax_NDV",
        "EVT_dPV2DVmin",
        "EVT_dPV2DVmax",
        "EVT_dPV2DVave",
        "EVT_Thrust_Mag",
        "EVT_Thrust_X",
        "EVT_Thrust_Y",
        "EVT_Thrust_Z"]

training_columns = cols
#training_columns = ["EVT_ThrustEmin_Ncharged","EVT_ThrustEmax_Ncharged","EVT_ThrustEmin_Nneutral","EVT_ThrustEmax_Nneutral"]
#training_columns = ["EVT_ThrustEmin_Ncharged","EVT_ThrustEmax_Ncharged","EVT_ThrustEmin_Nneutral","EVT_ThrustEmax_Nneutral","EVT_NtracksPV","EVT_NVertex", "EVT_ThrustEmin_NDV","EVT_ThrustEmax_NDV"]
#training_columns = ["EVT_ThrustEmin_Ncharged","EVT_ThrustEmax_Ncharged","EVT_ThrustEmin_Nneutral","EVT_ThrustEmax_Nneutral", "EVT_NVertex","EVT_ThrustEmin_NDV","EVT_ThrustEmax_NDV"]
#training_columns = ["EVT_ThrustEmin_Echarged", "EVT_ThrustEmax_Echarged", "EVT_ThrustEmin_Eneutral","EVT_ThrustEmax_Eneutral"]
#training_columns = ["EVT_NtracksPV","EVT_NVertex", "EVT_ThrustEmin_NDV","EVT_ThrustEmax_NDV",]

cols.append("EVT_ThrustDiff_E")

df_signal_np = df_signal.AsNumpy(columns=cols)
df_bb_np = df_bb.AsNumpy(columns=cols)
df_cc_np = df_cc.AsNumpy(columns=cols)
df_ss_np = df_ss.AsNumpy(columns=cols)
df_ud_np = df_ud.AsNumpy(columns=cols)

df_signal_pd = pd.DataFrame(df_signal_np)
df_bb_pd = pd.DataFrame(df_bb_np)
df_cc_pd = pd.DataFrame(df_cc_np)
df_ss_pd = pd.DataFrame(df_ss_np)
df_ud_pd = pd.DataFrame(df_ud_np)

# Branching fractions
bf_bb = 0.1512
bf_cc = 0.1203
bf_uds = 0.6991-0.1512-0.1203

evts_bb = round(df_bb_pd.shape[0]*bf_bb)
evts_cc = round(df_cc_pd.shape[0]*bf_cc)
evts_ud = round(df_ud_pd.shape[0]*2/3)
evts_ss = round(df_ss_pd.shape[0]*1/3)

df_ud_pd_fraction = df_ud_pd[1:evts_ud]
df_ss_pd_fraction = df_ss_pd[1:evts_ss]

dfs_uds = [df_ud_pd_fraction, df_ss_pd_fraction]
n_uds = np.sum(df.shape[0] for df in dfs_uds)
df_uds_pd = pd.concat(dfs_uds, axis=0,copy=True,ignore_index=True).iloc[random.sample(range(n_uds), n_uds),:]
#df_uds_pd = pd.concat(dfs_uds, copy=True,ignore_index=True)

evts_uds = round(df_uds_pd.shape[0]*bf_uds)

df_bb_pd_fraction = df_bb_pd[1:evts_bb]
df_cc_pd_fraction = df_cc_pd[1:evts_cc]
df_uds_pd_fraction = df_uds_pd[1:evts_uds]

dfs = [df_bb_pd_fraction, df_cc_pd_fraction, df_uds_pd_fraction]
n = np.sum(df.shape[0] for df in dfs)
bkg_df = pd.concat(dfs, axis=0, copy=True,ignore_index=True).iloc[random.sample(range(n), n),:]
#bkg_df = pd.concat(dfs, copy=True,ignore_index=True)

#dfs_all = [df_bb_pd, df_cc_pd, df_ss_pd, df_ud_pd]
#bkg_df = pd.concat(dfs_all)

# Preprocessing
bkg_df = bkg_df.copy()
bkg_df['category'] = 0  # Use 0 for background
df_signal_pd['category'] = 1  # Use 1 for signal
#pd_data = pd.concat([bkg_df, df_signal_pd],axis=1,copy=True,ignore_index=True)
pd_data = pd.concat([bkg_df, df_signal_pd],copy=True,ignore_index=True)

print("Sample size")
print("Signal:", df_signal_pd.shape[0])
print("Background:", bkg_df.shape[0])
print("")

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.set_output(transform="pandas")
imp.fit(pd_data)
pd_data_proc = imp.transform(pd_data)

y = pd_data_proc["category"]
x = pd_data_proc[training_columns]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)

test_data = pd.concat([x_test,y_test],axis=1,copy=True,ignore_index=True)

print("Train-test split")
print('x_train: ', len(x_train))
print('x_test: ', len(x_test))
print("")

sc = StandardScaler()
sc.set_output(transform = 'pandas')
sc.fit(x_train)
x_train = sc.transform(x_train)

sc.fit(x_test)
x_test = sc.transform(x_test)

for df in [df_signal_pd, df_bb_pd, df_cc_pd, df_ss_pd, df_ud_pd, bkg_df]:
    df[training_columns] = sc.transform(df[training_columns])

# Weights
# weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Define BDT
config_dict = {
   "n_estimators": 400,
   "learning_rate": 0.3,
   "max_depth": 3,
 }

bdt = xgb.XGBClassifier(n_estimators=config_dict["n_estimators"], learning_rate=config_dict["learning_rate"], max_depth=config_dict["max_depth"])

print("Training model")
print("Training variables:", training_columns)
print("")
bdt.fit(x_train, y_train)

data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)

xgb_cv = xgb.cv(dtrain=data_dmatrix, params=config_dict, nfold=4, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
print(xgb_cv.head())

# Save model to output file
bdt.save_model("/r01/lhcb/rrm42/fcc/bdt1.json")
print("Model saved to /r01/lhcb/rrm42/fcc/bdt1.json")

feature_importances = pd.DataFrame(bdt.feature_importances_,
                                   index = training_columns,
                                   columns=['importance']).sort_values('importance',ascending=False)

print("Feature importances")
print(feature_importances)
print("")

# BDT response
for df in [df_signal_pd, df_bb_pd, df_cc_pd, df_ss_pd, df_ud_pd, bkg_df]:
    df['XGB'] = bdt.predict_proba(df[training_columns])[:,1]

plt.figure()
plot_comparison('EVT_ThrustDiff_E')
plt.savefig("./output/comp_diff_e.pdf")

#print("cc no filter:", df_ud_pd.shape[0])

#print("N_bkg before BDT:", bkg_df.shape[0])

# Apply BDT cut
df_signal_pd_cut = df_signal_pd.query('XGB>0.9')
df_bb_pd_cut = df_bb_pd.query('XGB>0.9')
df_cc_pd_cut = df_cc_pd.query('XGB>0.9')
df_ss_pd_cut = df_ss_pd.query('XGB>0.9')
df_ud_pd_cut = df_ud_pd.query('XGB>0.9')
bkg_df_cut = bkg_df.query('XGB>0.9')
#print("cc with filter:", df_ud_pd_cut.shape[0])

bkg_before = bkg_df.shape[0]
bkg_after = bkg_df_cut.shape[0]
eff_cut = bkg_after/bkg_before

print("N_bkg before BDT:", bkg_before)
print("N_bkg after BDT:", bkg_after)
print("Cut efficiency:", eff_cut)

plt.figure()
plot_comparison_cut('EVT_ThrustDiff_E')
plt.savefig("./output/comp_diff_e_bdt.pdf")

plt.figure()
plot_bdt_comparison('XGB')
plt.savefig("./output/bdt_response.pdf")

# ROC curve
plt.figure()
plot_roc(bdt, x_test, y_test)
plt.savefig('./output/roc_bdt.pdf')

# Significance
plt.figure()
plot_significance(bdt, x_test, y_test)
plt.savefig('./output/significance.pdf')

