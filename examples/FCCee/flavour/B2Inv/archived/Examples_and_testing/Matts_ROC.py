import os
import glob
import uproot
import ROOT as r
import pandas as pd

path = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts"

branching_fractions = {
    "p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu": 1,
    "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu": 1,
    "p8_ee_Zbb_ecm91": 0.1512,
    "p8_ee_Zcc_ecm91": 0.1203,
    "p8_ee_Zss_ecm91": 0.1584,
    "p8_ee_Zud_ecm91": 0.2701,
}

fdict = {}

for file in glob.glob( f"{path}/*/chunk*_withMVA1.root" ):
    dec = file.split("/")[-2]
    fname = file.split("/")[-1]
    if dec not in fdict.keys():
        fdict[dec] = []
    fdict[dec].append( fname )

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

# now collect EVT MVA scores with weights 
df = pd.DataFrame( columns=["EVT_MVA1", "Decay", "Weight"] )
for dec in fdict.keys():
    eff = evdict[dec][0] / evdict[dec][1]
    weight = branching_fractions[dec] * eff

    subf = uproot.concatenate( f"{path}/{dec}/chunk*_withMVA1.root:events", "EVT_MVA1", library="pd" )
    subf["Decay"] = dec
    subf["Weight"] = weight

    # print(subf)

    df = pd.concat( [df, subf], ignore_index=True ) 
                             
# print(df)
# collect different types together
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

# def sorb( dec ):
#     if dec in ["p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu", "p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu"]:
#         return "signal"
#     else:
#         return "background"
    

df["Type"] = df["Decay"].apply( dec2type )
# df["SorB"] = df["Decay"].apply( sorb )

print(df)

# now make some ROC curves
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

comps = [ ("signal", "heavy"), ("signal", "light"), ("heavy", "light"), ("signal", "background") ]

fig, ax = plt.subplots()
for type1, type2 in comps:
    # if background just sum over everything that isn't signal
    if "background" in [type1, type2]:
        subf = df.copy()
        subf.loc[ subf["Type"] != "signal", "Type" ] = "background"
    else:
        subf = df[ (df["Type"]==type1) | (df["Type"]==type2 ) ] 

    y_true = subf["Type"].map( { type1: 1, type2: 0} ).values 
    y_pred = subf["EVT_MVA1"].values
    weight = subf["Weight"].values

    fpr, tpr, thresholds = roc_curve( y_true, y_pred, sample_weight=weight )
    roc_auc = auc(fpr, tpr)
    print( f"{type1} vs {type2} AUC =", roc_auc )

    ax.plot( fpr, tpr, lw=1, label= f"{type1} vs {type2} ({roc_auc:.2f})" )

ax.plot( [0,1], [0,1], color='k', ls='--', lw=1 )
ax.legend()
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
fig.tight_layout()
fig.savefig("roc_n_roll.png")
fig.savefig("roc_n_roll.pdf")
plt.show()
