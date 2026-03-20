import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
import bdt_lh_cut_opt_significance
import bdt_lh_cut_opt_significance_exclusive_backgrounds

save_path='outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'
plotpath = 'plots/paper_plots/JHEP_review'

samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']

with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)


print('Full Selection Efficiencies')
BF =  7.967265920370373e-09#7.806964487492601e-09#7.2309062308486705e-09#8.952995613376855e-09 #6.974125279294959e-09
print(f'BF = {BF}')
full_eff, full_eff_err = bdt_lh_cut_opt_significance.return_fullselneff_for_BF(interp_N_dict,BF)

# Prepare the table rows
table_data = [["Sample", "Efficiency", "Error"]]
for key in full_eff:
    eff = full_eff[key][0].item()  # Flatten the 1-element array
    err = full_eff_err[key][0].item()
    table_data.append([key, eff, err])

# Print the table
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

# Now create weighted averages for B(c)2lnu
eff = []
eff_err = []
BF = []
BF_err = []

print("Bu2lnu_background")
for sample in cfg.sample_allocations["Bu2lnu_background"]:
    eff.append(full_eff[sample][0].item())  # Flatten the 1-element array
    eff_err.append(full_eff_err[sample][0].item())
    BF.append(cfg.branching_fractions[sample][0])
    BF_err.append(cfg.branching_fractions[sample][1])


weighted_eff = sum([BF[i]*eff[i] for i in range(len(BF))])/sum([BF[i] for i in range(len(BF))])

eff_BF = [BF[i]*eff[i] for i in range(len(BF))]
sigeff_BF_sq = [(BF[i]*eff_err[i])**2 for i in range(len(BF))]
eff_sigBF_sq = [(BF_err[i]*eff[i])**2 for i in range(len(BF))]
sigBF_sq = [(BF_err[i])**2 for i in range(len(BF))]
sigeff_sq = [(eff_err[i])**2 for i in range(len(BF))]

weighted_eff_err = np.sqrt((sum(eff_BF)/sum(BF))**2*(sum(sigBF_sq)/sum(BF)**2) + sum(sigeff_BF_sq+eff_sigBF_sq)/sum(BF)**2)

print(weighted_eff)
print(weighted_eff_err)

print("err if normal avg for comparison:"+str( np.sqrt(sum(sigeff_sq) /len(BF))))

# Now create weighted averages for B(c)2lnu
eff = []
eff_err = []
BF = []
BF_err = []
print("Bc2lnu_background")
for sample in cfg.sample_allocations["Bc2lnu_background"]:
    eff.append(full_eff[sample][0].item())  # Flatten the 1-element array
    eff_err.append(full_eff_err[sample][0].item())
    BF.append(cfg.branching_fractions[sample][0])
    BF_err.append(cfg.branching_fractions[sample][1])


weighted_eff = sum([BF[i]*eff[i] for i in range(len(BF))])/sum([BF[i] for i in range(len(BF))])

eff_BF = [BF[i]*eff[i] for i in range(len(BF))]
sigeff_BF_sq = [(BF[i]*eff_err[i])**2 for i in range(len(BF))]
eff_sigBF_sq = [(BF_err[i]*eff[i])**2 for i in range(len(BF))]
sigBF_sq = [(BF_err[i])**2 for i in range(len(BF))]
sigeff_sq = [(eff_err[i])**2 for i in range(len(BF))]

weighted_eff_err = np.sqrt((sum(eff_BF)/sum(BF))**2*(sum(sigBF_sq)/sum(BF)**2) + sum(sigeff_BF_sq+eff_sigBF_sq)/sum(BF)**2)

print(weighted_eff)
print(weighted_eff_err)

print("err if normal avg for comparison:"+str( np.sqrt(sum(sigeff_sq) /len(BF))))






