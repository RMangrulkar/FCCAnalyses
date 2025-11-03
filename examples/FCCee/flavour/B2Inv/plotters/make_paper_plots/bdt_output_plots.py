import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from plotters import bdt_plotter_multiclass as bdt_plotter

#######################
#make BDT output plots
#######################

#load data with BDT applied
model, bdtname, dataframe = bdt_plotter.load_bdt_and_apply(pickled_df_path = os.path.join(cfg.optimised_bdt_lh_opts['outputPath'], "bdt_lh_dataframe.pkl"),
                            config_bdtopts = cfg.optimised_bdt_lh_opts,
                            training_round = "baseline-plus-hps",
                            hps_dict_name = "baseline-plus-hps",
                            features_list_name = "bdtlh-vars-v1",
                            bdt_label = '_lh')

outputpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/plots/paper_plots'

print('Now plotting...')

bdt_plotter.plot_bdt_response(dataframe, bdt_name = "BDT_lh" ,outpath=outputpath,bdt_score='bdt_score_0',categories = ['signal','heavy_background','light_background'],labels_map = bdt_plotter.labels, pt_fmt = bdt_plotter.blobs, colors = bdt_plotter.colors,xrange=(0,1))
bdt_plotter.plot_bdt_response(dataframe, bdt_name = "BDT_lh" ,outpath=outputpath,bdt_score='bdt_score_1',categories = ['signal','heavy_background','light_background'],labels_map = bdt_plotter.labels, pt_fmt = bdt_plotter.blobs, colors = bdt_plotter.colors,xrange=(0,1))
bdt_plotter.plot_bdt_response(dataframe, bdt_name = "BDT_lh" ,outpath=outputpath,bdt_score='bdt_score_2',categories = ['signal','heavy_background','light_background'],labels_map = bdt_plotter.labels, pt_fmt = bdt_plotter.blobs, colors = bdt_plotter.colors,xrange=(0,1))
   