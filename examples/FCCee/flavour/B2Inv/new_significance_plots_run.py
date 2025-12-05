import dill
import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
from tabulate import tabulate
import matplotlib.pyplot as plt



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import bdt_lh_cut_opt_significance_exclusive_backgrounds
import bdt_lh_cut_opt_significance
from bdt_lh_cut_opt_significance import flatten_list
import basic_functions


# New method that doesn't require data

path_for_interpolated_dicts = 'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'

#read N interp dicts - overwrite variable name with ones including exclusive backgrounds

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)


dict_path = os.path.join(path_for_interpolated_dicts,"incl_B2lnu/all_oneprong_tau")
plotpath = 'plots/exclusive_background_studies/20h_8m_4l_N_interp/incl_B2lnu/all_oneprong_tau'

samples_allocations = ["combined_signal", "hadronic_background","Bu2lnu_background","Bc2lnu_background"] 
samples = flatten_list([cfg.sample_allocations[component] for component in samples_allocations])
samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']

# plotting final binning plot with new splines - same samples used for cut optimisation

BF = 7.5e-8
bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components_to_plot =  samples_allocations, model_all_1prong_tau = True, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), model_all_1prong_tau = True, components_to_plot =  ["combined_signal", "Bu2lnu_background","Bc2lnu_background"] , binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

BF = 1e-6
bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2),  model_all_1prong_tau = True,components_to_plot =  samples_allocations, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 


bdt_lh_cut_opt_significance_exclusive_backgrounds.calculate_BF_sensitivities_extra_bkgs_nodata(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=samples_allocations, model_all_1prong_tau = True,  lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=True, ntoys=200,plot=True,saveplotpath = plotpath, toyplotpath = os.path.join(plotpath,"toys"), dict_path = dict_path) 


