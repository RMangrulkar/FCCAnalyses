import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

plt.style.use('fcc.mplstyle')

pickled_df_path= '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts_full_data/bdttau_outputs/baseline/bdttau_training_dataframe.pkl'

# load pickled df
df = pd.read_pickle(pickled_df_path)

config_bdtopts = cfg.bdttau_opts

signal_decays =  config_bdtopts["signalAllocation"]
background_decays = config_bdtopts["backgroundAllocation"] 


#ADd label
def labeller(dec):
    if dec in signal_decays:
        return 1
    elif dec in background_decays:
        return 0
    else:
        raise ValueError('Expect all data to be either light or heavy background if not signal')

df["label"] = df["decay"].apply(labeller)

# Labels: 1 for signal, 0 for background
y_true = df["label"]
scores = df["EVT_hemisEmax_n"]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(tpr,1-fpr, label=f'ROC curve (AUC = {roc_auc:.5f})', color='blue', lw=2)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.ylabel('Background Rejection (1-fpr)')
plt.xlabel('Signal Efficiency(tpr)')
plt.title('ROC Curve for EVT_hemisEmax_n Cut')
plt.legend(loc="lower left")
plt.grid()
plt.savefig('ROC_curve_EVT_hemisEmax_n_cut.pdf')