import dill
import os
import numpy as np
import pandas as pd
import pickle
from tabulate import tabulate
import config as cfg
import bdt_lh_cut_opt_with_n_interp

save_path='/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/'

# make N interp plots
with open(os.path.join(bdt_lh_cut_opt_with_n_interp.set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(bdt_lh_cut_opt_with_n_interp.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/paper_plots'

'''
bdt_lh_cut_opt_with_n_interp.plot_N(N_dict, interp_N_dict, lrange=(0.995,1) ,hrange=(0.995,1),nlh=20,normalised = True, separate_cbar = True, slice=False, save_path=plotpath)


# make final binning plot

#Load dataframe with bdtlh version applied
data={}
dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut' 
for sample in cfg.samples:
    data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))

# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
full_data['P_signal'] = full_data['bdt_score_2']
full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
full_data['P_not_light'] = 1-full_data['bdt_score_0'] 

#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus


bdt_lh_cut_opt_with_n_interp.make_final_binning_plot(full_data, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 200, signal_BF=1e-6, eventsProcessed_dict = cfg.eventsProcessed , histbins=(2,2), components =  ['hadronic_background','combined_signal'], binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                            plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True)


bdt_lh_cut_opt_with_n_interp.likelihood_model_builder(full_data, interp_N_dict, signal_BF=7.5e-8,#4e-7,#
                             lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh=200,bins = (2,2),
                             ntoys = 1,
                             fit_plotpath=plotpath, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), 
                             spread_plotpath=None, logpath = plotpath, hcut = None, lcut =None)
'''
opt_path = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/optimisation/0995'
plotpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/no_smoothing/0995/optimisation'
   
# load sicts for final sensitivity plot
with open(os.path.join(opt_path,'naive_sensitivity_dict.pkl'), 'rb') as f:
    naive_dict = pickle.load(f)
with open(os.path.join(opt_path,'toys_sensitivity_dict.pkl'), 'rb') as f:
    toys_dict = pickle.load(f)
with open(os.path.join(opt_path,'incl_syst_sensitivity_dict.pkl'), 'rb') as f:
    incl_syst_dict = pickle.load(f)
with open(os.path.join(opt_path,'optimal_bdt_cuts_dict.pkl'), 'rb') as f:
    BDT_cuts_dict = pickle.load(f)


bdt_lh_cut_opt_with_n_interp.sensitivity_CL_plotter(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=plotpath)
'''
print('Full Selection Efficiencies')
BF =  7.2309062308486705e-09#8.952995613376855e-09 #6.974125279294959e-09
print(f'BF = {BF}')
full_eff, full_eff_err = bdt_lh_cut_opt_with_n_interp.return_fullselneff_for_BF(interp_N_dict,BF)

# Prepare the table rows
table_data = [["Sample", "Efficiency", "Error"]]
for key in full_eff:
    eff = full_eff[key][0].item()  # Flatten the 1-element array
    err = full_eff_err[key][0].item()
    table_data.append([key, eff, err])

# Print the table
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
'''