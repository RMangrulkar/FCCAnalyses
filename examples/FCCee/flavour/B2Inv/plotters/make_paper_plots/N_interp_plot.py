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
plotpath = 'plots/paper_plots/JHEP_proofs_replies/'

samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']

# make N interp plots
with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(bdt_lh_cut_opt_significance.set_outputpath(save_path), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)


#bdt_lh_cut_opt_significance.plot_N(N_dict, interp_N_dict, lrange=(0.999,1) ,hrange=(0.999,1),nlh_highstats=20, nlh_midstats=8, nlh_lowstats=4,normalised=True, separate_cbar = True,slice=False, save_path=plotpath)
"""
opt_path = 'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats/only_hadbkg'
# load dicts for final sensitivity plot
with open(os.path.join(opt_path,'naive_sensitivity_dict.pkl'), 'rb') as f:
    naive_dict = pickle.load(f)
with open(os.path.join(opt_path,'toys_sensitivity_dict.pkl'), 'rb') as f:
    toys_dict = pickle.load(f)
with open(os.path.join(opt_path,'incl_syst_sensitivity_dict.pkl'), 'rb') as f:
    incl_syst_dict = pickle.load(f)
with open(os.path.join(opt_path,'optimal_bdt_cuts_dict.pkl'), 'rb') as f:
    BDT_cuts_dict = pickle.load(f)


print('Full Selection Efficiencies')
BF =   7.574532510734046e-09#7.967265920370373e-09#7.806964487492601e-09#7.2309062308486705e-09#8.952995613376855e-09 #6.974125279294959e-09
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
"""

bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,
                                       cut_opt_samples=samples_for_cut_opt,  
                                       lrange_interp_N_dict=(0.999,1),
                                       hrange_interp_N_dict=(0.999,1),
                                       nlh = 200, signal_BF=1e-6, #7.5e-8,
                                       eventsProcessed_dict = cfg.eventsProcessed, 
                                       histbins=(2,2), 
                                       components_to_plot =  ['hadronic_background','combined_signal'],#, 'B2lnu_background_combined'], 
                                       model_all_1prong_leptonic_tau = True,
                                       binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                       plot_signal_components=True,  
                                       final_plot_path = plotpath, 
                                       pull_type_plot=True,  
        )

bdt_lh_cut_opt_significance_exclusive_backgrounds.likelihood_model_builder_extra_bkgs_nodata(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=['hadronic_background','combined_signal'],#, 'B2lnu_background_combined'], 
                                                                                             model_all_1prong_leptonic_tau = True, signal_BF=7.106591034230751e-08,#7.5e-8,#6.9e-8,
                                                                                             lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh=200,bins = (2,2),
                                                                                             ntoys = 1, fit_plotpath=plotpath, x_values = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]), spread_plotpath=None, logpath=None, lcut=None, hcut=None)

"""
bdt_lh_cut_opt_significance.sensitivity_CL_plotter_v2(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=plotpath, spine_sampling=24)#21 #23

dict_path = os.path.join(save_path,"only_hadbkg")
#only need this is dont have SB_dict saved in pickle format with correct components etc.
#print("Running: calc_BF_sig_stderrs_nodata")
#bdt_lh_cut_opt_significance_exclusive_backgrounds.calc_BF_sig_stderrs_nodata(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=['hadronic_background','combined_signal', 'B2lnu_background_combined'], model_all_1prong_leptonic_tau=True, lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),plot=True,saveplotpath = plotpath ,dict_path = dict_path, plot_dotted_line_list =  ["noBsyst","2percent_Bsyst" ,"10percent_Bsyst"])


#otherwise load this file and plot directly
with open(os.path.join(dict_path,'SB_dict.pkl'), 'rb') as f:
    SB_dict = pickle.load(f)


bdt_lh_cut_opt_significance_exclusive_backgrounds.plot_BF_sig_stderrs(SB_dict["S"],SB_dict["B"],SB_dict["BFs"],plotpath,plot_dotted_line_list= ["noBsyst","2percent_Bsyst" ,"10percent_Bsyst"])



"""
#bdt_lh_cut_opt_significance.sensitivity_CL_plotter_monotonic_interp(naive_dict, incl_syst_dict, toys_dict = toys_dict, savepath=plotpath, spine_sampling=26)
#bdt_lh_cut_opt_significance.run_2d_optimisation(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200, sig_BF=4/10*1.4e-4,incl_other_syst = True)
