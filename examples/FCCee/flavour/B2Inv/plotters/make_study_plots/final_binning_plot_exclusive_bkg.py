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

plotpath = 'plots/exclusive_background_studies/baseline_N_interp/all_oneprong_tau'
samples_allocations = ["combined_signal", "hadronic_background","Bu2lnu_background","Bc2lnu_background"] 
samples = flatten_list([cfg.sample_allocations[component] for component in samples_allocations])
 

#Load dataframe with bdtlh version applied
data={}

for sample in samples:
    print(sample)
    if sample in cfg.exclusive_backgrounds:
        allocation = next((key for key, samples_list in cfg.sample_allocations.items() if sample in samples_list),None)

        if allocation is None:
            print(f"Set up needed for other exclusive backgrounds: {sample}")
            continue  # skip this sample or handle appropriately

        dir = cfg.fccana_opts["outputDir"][f"{allocation}_no_lepton_veto"]
        folder = 'dataframes_cutEVT_hemisEmin_nLept----0/bdtlh_099cut' #use samples that already have lepton veto applied

        pkl_files = glob(os.path.join(dir, folder,sample, "*.pkl"))
    
        if len(pkl_files) == 0:
            raise FileNotFoundError("No pickle files found in the folder.")

        dfs = [pd.read_pickle(f) for f in pkl_files]

        data[sample] = pd.concat(dfs, ignore_index=True)


    else:
        #inclusive backgrounds and signal samples
        dir = cfg.fccana_opts["outputDir"]["prelim_cuts_full"]
        folder = 'baseline_plus_bdtlh_dataframes/full_sample_medium_bdtlh_cut'
        data[sample] = pd.read_pickle(os.path.join(dir,folder,f'{sample}.pkl'))



# Concatenate all DataFrames
full_data = pd.concat(data.values(), ignore_index=True)


#add extra rows so can work in range0.9-1
#full_data['P_signal'] = full_data['bdt_score_2']
#full_data['P_not_heavy'] = 1-full_data['bdt_score_1'] 
#full_data['P_not_light'] = 1-full_data['bdt_score_0'] 
#add extra rows so can work in range0.9-1
full_data = full_data.assign(
    P_signal=full_data['bdt_score_2'],
    P_not_heavy=1 - full_data['bdt_score_1'],
    P_not_light=1 - full_data['bdt_score_0'],
)


#add any extra cuts need here###########################
full_data = full_data.query('EVT_hemisEmax_n>10') #veto on taus

'''
print('Full Selection Efficiencies')
BF =  7.5e-8
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


#bdt_lh_cut_opt_significance.make_final_binning_plot(full_data, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components =  samples_allocations, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

'''

path_for_interpolated_dicts =  'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0995'


#create Ninterp maps 
#bdt_lh_cut_opt_significance.create_N_map(full_data,lrange=(0.995,1) ,hrange=(0.995,1),nlh=20, smoothing=False, kx=2,ky=2,save_path=path_for_interpolated_dicts) #df should be full data for final result

#read N interp dicts - overwrite variable name with ones including exclusive backgrounds
with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)

dict_path = os.path.join(path_for_interpolated_dicts,"all_oneprong_tau")

#using same samples as before so that the cut points are the same
samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']


print("starting sensitivity calculation")
bdt_lh_cut_opt_significance_exclusive_backgrounds.calculate_BF_sensitivities_extra_bkgs(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=samples_allocations, model_all_1prong_tau = True,  lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=True, full_df=full_data, ntoys=200,plot=True,saveplotpath = plotpath, toyplotpath = os.path.join(plotpath,"toys"), dict_path = dict_path) 
BF = 7.5e-8
bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs(full_data, interp_N_dict, lrange_interp_N_dict=(0.995,1) ,hrange_interp_N_dict=(0.995,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components_to_plot =  samples_allocations, model_all_1prong_tau = True,binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 



#plot_BF_sensitivities(interp_N_dict,lrange_plot=(0.995,1) ,hrange_plot=(0.995,1), nlh=200 , sig_BFs=np.logspace(np.log10(1e-9),np.log10(4e-6),300), incl_other_syst=True, incl_toys_fit=True, full_df=full_data, ntoys=10000,plot=True,saveplotpath = plotpath ,toyplotpath = plotpath, dict_path = dict_path)


'''
# New method that doesn't require data

path_for_interpolated_dicts = 'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'

#read N interp dicts - overwrite variable name with ones including exclusive backgrounds
with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "N_remaining_dictionary"), "rb") as dill_file:
        N_dict = dill.load(dill_file)

with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)


dict_path = os.path.join(path_for_interpolated_dicts,"incl_B2lnu")
plotpath = 'plots/exclusive_background_studies/20h_8m_4l_N_interp/incl_B2lnu'

samples_allocations = ["combined_signal", "hadronic_background","Bu2lnu_background","Bc2lnu_background"] 
samples = flatten_list([cfg.sample_allocations[component] for component in samples_allocations])
samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']

# plotting final binning plot with new splines - same samples used for cut optimisation

BF = 7.5e-8
bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components_to_plot =  samples_allocations, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components_to_plot =  ["combined_signal", "Bu2lnu_background","Bc2lnu_background"] , binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 

BF = 1e-6
bdt_lh_cut_opt_significance_exclusive_backgrounds.make_final_binning_plot_extra_bkgs_nodata(interp_N_dict,cut_opt_samples=samples_for_cut_opt, lrange_interp_N_dict=(0.999,1) ,hrange_interp_N_dict=(0.999,1),nlh = 200, signal_BF=BF, eventsProcessed_dict = cfg.eventsProcessed , 
                                                    histbins=(2,2), components_to_plot =  samples_allocations, binned_x_axis = np.array([['Signal depleted','Heavy background \n enriched'],['Light background \n enriched','Signal enriched']]),
                                                    plot_signal_components=True,  nMC_plots_path=None, final_plot_path = plotpath, pull_type_plot=True) 


bdt_lh_cut_opt_significance_exclusive_backgrounds.calculate_BF_sensitivities_extra_bkgs_nodata(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=samples_allocations,  lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),incl_other_syst=True, incl_toys_fit=True, ntoys=200,plot=True,saveplotpath = plotpath, toyplotpath = os.path.join(plotpath,"toys"), dict_path = dict_path) 



# S and B saving and plootting sensitivities with standard error


path_for_interpolated_dicts = 'outputs/B2lnu_backgrounds_included/BDTlh_baseline_plus_cut_optimisation/no_smoothing/0999/8x8xmidstats_4x4lowstats'

#read N interp dicts - overwrite variable name with ones including exclusive backgrounds
with open(os.path.join(basic_functions.set_outputpath(path_for_interpolated_dicts), "interpolated_N_remaining_dictionary"), "rb") as dill_file:
    interp_N_dict = dill.load(dill_file)


dict_path = os.path.join(path_for_interpolated_dicts,"only_hadbkg")
plotpath = 'plots/exclusive_background_studies/20h_8m_4l_N_interp/only_hadbkg'

samples_allocations = ["combined_signal", "hadronic_background"]#,"Bu2lnu_background","Bc2lnu_background"] 
samples = flatten_list([cfg.sample_allocations[component] for component in samples_allocations])
samples_for_cut_opt = cfg.sample_allocations['hadronic_background'] + cfg.sample_allocations['combined_signal']


#bdt_lh_cut_opt_significance_exclusive_backgrounds.calc_BF_sig_stderrs_nodata(interp_N_dict, cut_opt_samples=samples_for_cut_opt, componenets_in_FOM=samples_allocations,  lrange_plot=(0.999,1) ,hrange_plot=(0.999,1), nlh=200 , sig_BFs=np.logspace(-9,-4,250),plot=True,saveplotpath = plotpath ,dict_path = dict_path, plot_dotted_line_list =  ["noBsyst","2percent_Bsyst" ,"10percent_Bsyst"])

#read SB dict - overwrite variable name with ones including exclusive backgrounds
with open(os.path.join(basic_functions.set_outputpath(dict_path), "SB_dict.pkl"), "rb") as file:
    SB_dict = pickle.load(file)


bdt_lh_cut_opt_significance_exclusive_backgrounds.plot_BF_sig_stderrs(SB_dict["S"],SB_dict["B"],SB_dict["BFs"],plotpath,plot_dotted_line_list= ["noBsyst","2percent_Bsyst" ,"10percent_Bsyst"])


plt.figure()
plt.plot(SB_dict["BFs"], SB_dict["l_cut"], label="1-P(l) cut")
plt.plot(SB_dict["BFs"], SB_dict["h_cut"], label="1-P(h) cut")
plt.xscale('log')
plt.legend()
plt.xlabel(r'$\mathcal{B}(B_{(s)}^0 \rightarrow$ invisible$)$')
plt.ylabel("BDT cut")
plt.savefig(os.path.join(plotpath,"BDT_cut.pdf"))

'''