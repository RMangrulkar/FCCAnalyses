import os
import glob
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
import json
from time import time
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import log_loss


import config as cfg
import efficiency_finder
import variable_plotter as vp
plt.style.use('fcc.mplstyle')

#multivariate map
labels = {'signal':2,'heavy_background':1 ,'light_background':0}
labels_dict_inverted = {v: k for k, v in labels.items()}
colors = {'signal':'mediumblue','heavy_background':'y' ,'light_background':'g','background':'r'}
blobs = {'signal':'bo','heavy_background':'y+' ,'light_background':'g+','background':'r+' }


# Function to load the BDT model from a JSON file
def load_bdt_model(json_path):
    bdt_model = xgb.Booster()
    bdt_model.load_model(json_path)
    return bdt_model

# Function to load the BDT model from a JSON file
def load_bdt_model_sklearn(json_path):
    bdt_model = XGBClassifier()
    bdt_model.load_model(json_path)
    return bdt_model

# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars



###################################
## Define functions for plotting ## - KEY PLOTTERS
###################################



def plot_2d_bdt_output(df, bdt_name = "BDT_lh",output_file_name = '2dbdt_response_plot',outpath=None,bdt_probs = [0,1],labels=labels_dict_inverted, vmax=20, xrange=[0,1],yrange=[0,1],density=True):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9,8))

    df_bb = df[df['decay']=='p8_ee_Zbb_ecm91']
    df_cc = df[df['decay']=='p8_ee_Zcc_ecm91']
    df_ss = df[df['decay']=='p8_ee_Zss_ecm91'] 
    df_ud = df[df['decay']=='p8_ee_Zud_ecm91'] 


    # Big bins
    axs[0, 0].set_title((cfg.titles['p8_ee_Zbb_ecm91']))
    axs[0,0].set_ylabel(f'bdt_hl {labels[bdt_probs[0]]} probability')
    #axs[0, 0].set_xticklabels([])
    
    # Small bins
    axs[0, 1].set_title((cfg.titles['p8_ee_Zcc_ecm91']))
    #axs[0, 1].set_xticklabels([])
    #axs[0, 1].set_yticklabels([])


    axs[1, 0].set_title((cfg.titles['p8_ee_Zss_ecm91']))
    axs[1,0].set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')
    axs[1,0].set_ylabel(f'bdt_hl {labels[bdt_probs[0]]} probability')


    axs[1, 1].set_title((cfg.titles['p8_ee_Zud_ecm91']))
    axs[1,1].set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')
    #axs[1, 1].set_yticklabels([])


    h1 = axs[0, 0].hist2d(df_bb[f"bdt_score_{bdt_probs[1]}"], df_bb[f"bdt_score_{bdt_probs[0]}"], bins=(50, 50), range= [xrange,yrange], weights=df_bb[f"total_weight"],cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)
    h2 = axs[0, 1].hist2d(df_cc[f"bdt_score_{bdt_probs[1]}"], df_cc[f"bdt_score_{bdt_probs[0]}"], bins=(50, 50), range= [xrange,yrange], weights=df_cc[f"total_weight"], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)
    h3 = axs[1, 0].hist2d(df_ss[f"bdt_score_{bdt_probs[1]}"], df_ss[f"bdt_score_{bdt_probs[0]}"], bins=(50, 50), range= [xrange,yrange], weights=df_ss[f"total_weight"], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)
    h4 = axs[1, 1].hist2d(df_ud[f"bdt_score_{bdt_probs[1]}"], df_ud[f"bdt_score_{bdt_probs[0]}"], bins=(50, 50), range= [xrange,yrange], weights=df_ud[f"total_weight"], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)

    # Visualizing colorbar 
    fig.colorbar(h1[3], ax=axs[0, 0])
    fig.colorbar(h2[3], ax=axs[0, 1])
    fig.colorbar(h3[3], ax=axs[1, 0])
    fig.colorbar(h4[3], ax=axs[1, 1])

    if xrange !=[0,1] or yrange !=[0,1]:
        comment='zoomed'
    else:
        comment = ''

    if outpath:
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_backgrounds_{comment}.pdf'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_backgrounds_{comment}.png'))
    else:
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_backgrounds_{comment}.pdf')
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_backgrounds_{comment}.png')
    

    
    #now plotting signal
    df_Bs = df[df['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu']
    df_Bd = df[df['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9,4))

    axs[0].set_title(cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'])
    axs[0].set_ylabel(f'bdt_hl {labels[bdt_probs[0]]} probability')
    axs[0].set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')
    axs[1].set_title(cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'])
    axs[1].set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')

    hs1=axs[0].hist2d(df_Bs[f"bdt_score_{bdt_probs[1]}"], df_Bs[f"bdt_score_{bdt_probs[0]}"], weights=df_Bs[f"total_weight"], bins=(50, 50), range= [xrange,yrange], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)
    hs2=axs[1].hist2d(df_Bd[f"bdt_score_{bdt_probs[1]}"], df_Bd[f"bdt_score_{bdt_probs[0]}"], weights=df_Bd[f"total_weight"], bins=(50, 50), range= [xrange,yrange], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)

    # Visualizing colorbar 
    fig.colorbar(hs1[3], ax=axs[0])
    fig.colorbar(hs2[3], ax=axs[1])
    
    if outpath:
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_signals_{comment}.pdf'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_signals_{comment}.png'))
    else:
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_signals_{comment}.pdf')
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_signals_{comment}.png')


def plot_2d_bdt_output_any_sample(df, bdt_name = "BDT_lh",output_file_name = '2dbdt_response_plot',outpath=None,bdt_probs = [0,1],labels=labels_dict_inverted, vmax=20, xrange=[0,1],yrange=[0,1],density=True,sample = 'p8_ee_Ztautau_ecm91'):

    df = df[df['decay']==sample]

    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9,8))

    axs.set_title(cfg.titles[sample])
    axs.set_ylabel(f'bdt_hl {labels[bdt_probs[0]]} probability')
    axs.set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')

    hs1=axs.hist2d(df[f"bdt_score_{bdt_probs[1]}"], df[f"bdt_score_{bdt_probs[0]}"], bins=(50, 50), range= [xrange,yrange], cmap=plt.cm.Blues_r,vmin=0,vmax=vmax,density=density)

    # Visualizing colorbar 
    fig.colorbar(hs1[3], ax=axs)

    if xrange !=[0,1] or yrange !=[0,1]:
        comment='zoomed'
    else:
        comment = ''

    if outpath:
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_{sample}_{comment}.pdf'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_{sample}_{comment}.png'))
    else:
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_{sample}_{comment}.pdf')
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}_{sample}_{comment}.png')



def post_bdt_variable_plot(data,variable,
                           bdt_cut = None, #must be string of correct format as in vp
                           bdt_name = "BDT_lh",
                           weight=True,
                           density=True,
                           bins=50,
                           xrange=None,
                           outpath=None,
                           stacked=True, 
                           total=["hadronic_background"], 
                           components=["Bssignal", "hadronic_background"], #currently not set up to do Bd and Bs with separate nominal bfs but thsi shouldnt be an issue
                           signal_bf=1e-6):
    if outpath:
        savepath=os.path.join(outpath,f'{variable}_with_{bdt_name}_cut_{bdt_cut}.pdf')
    else:
        savepath= f'{variable}_with_{bdt_name}_cut_{bdt_cut}.pdf'


    decays_list = [cfg.sample_allocations[i] for i in components]
    flat_decays_list = [item for sublist in decays_list for item in sublist]


    if bdt_cut:
        df = data.copy().query(bdt_cut)
    else:
        df = data.copy()


    values = {sample: df[df['decay']==sample][variable].to_numpy() for sample in flat_decays_list}

    if xrange is None:
        xmin = min( [ min(values[sample]) for sample in values ] )
        xmax = max( [ max(values[sample]) for sample in values ] )
    else:
        xmin = xrange[0]
        xmax = xrange[1]

    # If variable is an integer make sure nbins correct
    #for most variables nbins=xmax-xmin
    #unless ChargedRP_fromPV_transformed
    if bins!=50:
        nbins = bins
    elif 'ChargedRP_fromPV_transformed' in variable:
        nbins=3
    elif '_n' in variable and '_norm' not in variable:
        print(variable)
        xmin = 0
        nbins= int(xmax - xmin)
    else:
        nbins=50

    fig, ax = plt.subplots()

    hist_settings,total_colours = vp.histogram_settings()

    if weight: 
        desired_signal_samples = [sample for sample in cfg.sample_allocations['combined_signal'] if sample in flat_decays_list]
        for sample in desired_signal_samples:
            df.loc[df['decay'] == sample, 'w1']=df[df['decay']==sample]['w1']* 2*cfg.branching_fractions['p8_ee_Zbb_ecm91'][0]*cfg.prod_frac[sample][0]*signal_bf


        if density==False:
            ax.set_title(f'Assuming signal branching fraction = {signal_bf:.1e}')

    for allocation in cfg.sample_allocations:

        if allocation not in components:
            continue

        samples = cfg.sample_allocations[allocation]
        hist_x = [ values[sample] for sample in samples ]
        hist_l = [ cfg.titles[sample] for sample in samples ]
        hist_opts = hist_settings[allocation]
        tot_colour= total_colours[allocation]

        if weight:
            hist_w= [np.array(df[df['decay']==sample]['w1'])*6e12 for sample in samples]
            #print(variable)
            #print(hist_w)
        else:
            hist_w = None

        if stacked:
            hist_opts['stacked'] = True
        else:
            hist_opts['stacked'] = False

        ax.hist( 
            x = hist_x,
            bins = nbins,
            range = (xmin,xmax),
            density = density,
            label = hist_l,
            weights = hist_w,
            **hist_opts
        )

        if allocation in total and stacked:
            ax.hist( 
                np.concatenate( hist_x), 
                bins = nbins,
                range = (xmin,xmax),
                density = density,
                label = f'Total {allocation.replace("_", " ")}',
                weights = np.concatenate( hist_w ) if weight else None,
                histtype = 'step',
                color = tot_colour,#'k',
                lw = 2,
            )

    ax.legend(reverse=True)



    if bdt_cut is not None:
        ax.set_xlabel(f"{variable} (cut={vp.replace_all(vp.replace_all(vp.replace_all(bdt_cut,'>','$>$'),'<','$<$'),'&',',')})")
    else:
        ax.set_xlabel(f"{variable} (cut={bdt_cut})")

    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Counts')
   

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(savepath)



#######################################################
# Load BDT and apply to loaded data - define as funtion
#######################################################

def load_bdt_and_apply(pickled_df_path = os.path.join(cfg.bdt_lh_opts['outputPath'], "bdt_lh_dataframe.pkl"), 
                        config_bdtopts = cfg.bdt_lh_opts,
                        training_round = "multiclass_baseline",
                        hps_dict_name = "default-hps",
                        features_list_name = "bdth-plus-vars",
                        bdt_label = '_lh'
                        ): # hps dict and features_list_name specift BDT used
    
    
    #path to data and outputs
    outputpath   = config_bdtopts['outputPath']
    yamlpath = cfg.fccana_opts['yamlPath']

    #Getting BDT vars for training from yaml
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'


    #path to saved bdt
    bdt_json_path = os.path.join(outputpath,training_round,f"{bdtname}.json")

    # Load the BDT model
    try:
        print("Loading BDT model...")
        bdt_model = load_bdt_model_sklearn(bdt_json_path)
        print("BDT model loaded successfully.")
    except Exception as e:
        print(f"Error loading BDT model: {e}")
        quit()

    # Read df saved to pickle and add BDT 

    #Getting BDT vars for training from yaml
    # load pickled df
    df = pd.read_pickle(pickled_df_path)

    #add bdt score
    class_names = bdt_model.classes_
    probabilities = bdt_model.predict_proba(df[bdtvars])
    for i in range(probabilities.shape[1]):
        df[f'bdt{bdt_label}_score_{class_names[i]}'] = probabilities[:, i]


    return bdt_model, bdtname, df

