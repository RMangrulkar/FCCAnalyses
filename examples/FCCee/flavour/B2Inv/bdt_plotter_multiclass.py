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


def plot_ROC_star(df, bdt_name = "BDT_lh",
                  bdtscore_label="bdt_score_2",
                  weight_label="total_weight", 
                  comps = [("signal", "heavy_background"), ("signal", "light_background"), ("heavy_background", "light_background"), ("signal", "background")],
                  output_file_name = "ROC_star",
                  outpath=None):
    
    # make roc curve (use test sample)
    
    df['Type'] = df[f'label'].apply(lambda l: labels_dict_inverted.get(l, "unknown"))
    fig, ax = plt.subplots()
    for type1, type2 in comps:
        # if background just sum over everything that isn't signal
        if "background" in [type1, type2]:
            subf = df[df["sample"]==1].copy() # only use test data
            subf.loc[ subf["Type"] != "signal", "Type" ] = "background"
        else:
            subf = df[ (df["Type"]==type1) | (df["Type"]==type2 ) ] 
        
        y_true = subf["Type"].map( { type1: 1, type2: 0} ).values 
        y_pred = subf[bdtscore_label].values
        weight = subf[weight_label].values

        fpr, tpr, thresholds = roc_curve( y_true, y_pred, sample_weight=weight )
        roc_auc = auc(fpr, tpr)
        print( f"{type1} vs {type2} AUC =", roc_auc )

        ax.plot( tpr, 1-fpr, lw=1, label= f"{type1} vs {type2} ({roc_auc:.5f})" )
    ax.legend()
    ax.set_xlabel( "Signal Efficiency" )
    ax.set_ylabel( "Background Rejection (1-fpr)" )
    
    ax.plot( [1,0], [0,1], color='k', ls='--', lw=1, alpha=0.6 )
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.png'))
        fig.savefig(os.path.join(outpath,f"{bdt_name}_{output_file_name}.pdf"))
    else:
        #fig.savefig(f'{bdt_name}_{output_file_name}.png')
        fig.savefig(f"{bdt_name}_{output_file_name}.pdf")
    ax.set_xticks(np.arange(0.8,1.05, 0.05))
    ax.set_yticks(np.arange(0.8,1.05, 0.05))
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.7, alpha=0.4)
    ax.set_xlim(0.8,1.005)
    ax.set_ylim(0.8,1.005)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f"{bdt_name}_{output_file_name}_zoomedin.pdf")




def plot_bdt_response(df, bdt_name = "BDT_lh",output_file_name = "response" ,outpath=None,bdt_score='bdt_score_2',categories = ['signal','background'],labels_map = labels, pt_fmt = blobs, colors = colors,xrange=(0,1)):
    '''
    categories (bool): which catagories to plot, must be valid category in labels_map or 'background'
    
    '''
    # plot of BDT output
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6.4,6.4))

    train_dict={}
    test_dict={}
    train_w_dict={}
    test_w_dict={}

    for cat in categories:
        if cat == 'background':
            train_dict[cat] = df[ (df["sample"]==0) & (df["label"]!=labels_map['signal']) ][bdt_score].values
            test_dict[cat] = df[ (df["sample"]==1) & (df["label"]!=labels_map['signal']) ][bdt_score].values
            train_w_dict[cat] = df[ (df["sample"]==0) & (df["label"]!=labels_map['signal']) ]["total_weight"].values
            test_w_dict[cat] = df[ (df["sample"]==1) & (df["label"]!=labels_map['signal']) ]["total_weight"].values
        
        else:
            train_dict[cat] = df[ (df["sample"]==0) & (df["label"]==labels_map[cat]) ][bdt_score].values
            test_dict[cat] = df[ (df["sample"]==1) & (df["label"]==labels_map[cat]) ][bdt_score].values
            train_w_dict[cat] = df[ (df["sample"]==0) & (df["label"]==labels_map[cat]) ]["total_weight"].values
            test_w_dict[cat] = df[ (df["sample"]==1) & (df["label"]==labels_map[cat]) ]["total_weight"].values
        
    
        # plot training sample dists
        ax[0].hist( train_dict[cat], bins=50, range=xrange, label=f'{cat} train', alpha=0.5, ec='none', fc=colors[cat], weights=train_w_dict[cat], density=True )

        # plot test sample dists
        # if you want the error need to track squared weights (probably a better way of doing this)
        n, xe = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat] )
        n2, _ = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat]**2 )
        ne = n2**0.5 / n
        n, x = np.histogram( test_dict[cat], bins=50, range=xrange, density=True, weights=test_w_dict[cat] )
        ne = ne * n


        cx = 0.5*(xe[1:]+xe[:-1])
        ax[0].errorbar( cx, n, ne, fmt=pt_fmt[cat], label=f'{cat} test') 


        # now plot the residual
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat] )
        nt2, _ = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat]**2 )
        nte = nt2**0.5 / nt
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, density=True, weights=train_w_dict[cat] )
        nte = nte * nt


        d = n - nt
        de = (ne**2 + nte**2)**0.5
        p = d / de

        
        ax[1].errorbar( cx, p, np.ones_like(p), fmt=pt_fmt[cat] )

    ax[1].axhline(0, c='k', ls='--' )   
    ax[1].set_ylabel('Pull')
    ax[0].set_xlabel(f'{bdt_name} XGBoost Signal Probability')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].set_yscale('log')
    fig.tight_layout()

    if xrange==(0,1):
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf")
    else:
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed.pdf")
'''
def plot_bdt_response_combinedcut(df, bdt_name = "BDT_lh",output_file_name = "response_combinedcut" ,outpath=None,categories = ['signal','background'],labels_map = labels, pt_fmt = blobs, colors = colors,xrange=(0,1)):
    
    #categories (bool): which catagories to plot, must be valid category in labels_map or 'background'
    
    
    # plot of BDT output
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6.4,6.4))

    train_dict={}
    test_dict={}
    train_w_dict={}
    test_w_dict={}

    for cat in categories:
        if cat == 'background':

            subdf_train = df[ (df["sample"]==0) & (df["label"]!=labels_map['signal']) ]
            subdf_test = df[ (df["sample"]==1) & (df["label"]!=labels_map['signal']) ]
        
        else:
            subdf_train = df[ (df["sample"]==0) & (df["label"]==labels_map[cat]) ]
            subdf_test = df[ (df["sample"]==1) & (df["label"]==labels_map[cat]) ]


        train_dict[cat] = subdf_train["bdt_score_2"].values * (1-subdf_train["bdt_score_1"].values) * (1-subdf_train["bdt_score_0"].values)
        test_dict[cat] = subdf_test["bdt_score_2"].values * (1-subdf_test["bdt_score_1"].values) * (1-subdf_test["bdt_score_0"].values)
        train_w_dict[cat] = subdf_train["total_weight"].values
        test_w_dict[cat] = subdf_test["total_weight"].values
    
    
        # plot training sample dists
        ax[0].hist( train_dict[cat], bins=50, range=xrange, label=f'{cat} train', alpha=0.5, ec='none', fc=colors[cat], weights=train_w_dict[cat], density=True )

        # plot test sample dists
        # if you want the error need to track squared weights (probably a better way of doing this)
        n, xe = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat] )
        n2, _ = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat]**2 )
        ne = n2**0.5 / n
        n, x = np.histogram( test_dict[cat], bins=50, range=xrange, density=True, weights=test_w_dict[cat] )
        ne = ne * n


        cx = 0.5*(xe[1:]+xe[:-1])
        ax[0].errorbar( cx, n, ne, fmt=pt_fmt[cat], label=f'{cat} train') 


        # now plot the residual
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat] )
        nt2, _ = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat]**2 )
        nte = nt2**0.5 / nt
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, density=True, weights=train_w_dict[cat] )
        nte = nte * nt


        d = n - nt
        de = (ne**2 + nte**2)**0.5
        p = d / de

        
        ax[1].errorbar( cx, p, np.ones_like(p), fmt=pt_fmt[cat] )

    ax[1].axhline(0, c='k', ls='--' )   
    ax[1].set_ylabel('Pull')
    ax[0].set_xlabel(f'{bdt_name} XGBoost P(signal)[(1-P(light))(1-P(heavy))]')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].set_yscale('log')
    fig.tight_layout()

    if xrange==(0,1):
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf")
    else:
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed..pdf")
'''

# efficiency plot (on total sample)
def plot_eff(df, bdt_name = "BDT_lh",output_file_name = 'efficiency_plot',outpath=None):
    fig, ax = plt.subplots()
    for decay in df["decay"].unique():
        subf = df[ df["decay"]==decay ]
        mva_scores = subf["bdt_score_2"].values
        weights = subf["total_weight"].values

        sorted_indices = np.argsort( mva_scores )
        sorted_scores = mva_scores[sorted_indices]
        sorted_weights = weights[sorted_indices]

        total_weight = np.sum( sorted_weights ) 
        cumalative_weights = np.cumsum( sorted_weights[::-1] )[::-1] # reverse order for efficiency above cut
        efficiency = cumalative_weights / total_weight

        ax.plot( sorted_scores, efficiency, label=decay )

    ax.legend()
    ax.set_xlabel(f'{bdt_name} XGBoost Signal Probability')
    ax.set_ylabel('Efficiency')
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', linestyle='-', color='0.7', linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f'{bdt_name}_{output_file_name}.pdf')
    #ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.set_xlim(0.95,1.002)
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f'{bdt_name}_{output_file_name}_zoomedin.pdf')


# efficiency plot (on total sample)
def plot_eff_combinedcut(df, bdt_name = "BDT_lh",output_file_name = 'efficiency_plot_combinedcut',outpath=None):
    fig, ax = plt.subplots()
    for decay in df["decay"].unique():
        subf = df[ df["decay"]==decay ]
        mva_scores = subf["bdt_score_2"].values * (1-subf["bdt_score_1"].values) * (1-subf["bdt_score_0"].values)
        weights = subf["total_weight"].values

        sorted_indices = np.argsort( mva_scores )
        sorted_scores = mva_scores[sorted_indices]
        sorted_weights = weights[sorted_indices]

        total_weight = np.sum( sorted_weights ) 
        cumalative_weights = np.cumsum( sorted_weights[::-1] )[::-1] # reverse order for efficiency above cut
        efficiency = cumalative_weights / total_weight

        ax.plot( sorted_scores, efficiency, label=decay )

    ax.legend()
    ax.set_xlabel(f'{bdt_name} XGBoost P(signal)[(1-P(light))(1-P(heavy))]')
    ax.set_ylabel('Efficiency')
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', linestyle='-', color='0.7', linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f'{bdt_name}_{output_file_name}.pdf')
    #ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.set_xlim(0.95,1.002)
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f'{bdt_name}_{output_file_name}_zoomedin.pdf')



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
    df_Bs = dataframe[dataframe['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu']
    df_Bd = dataframe[dataframe['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu']

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


def plot_2d_scatter(df, bdt_name = "BDT_lh",output_file_name = '2dbdt_scatter_plot',outpath=None,bdt_probs = [0,1],labels=labels_dict_inverted,xrange=(0,0.1),yrange=(0,0.1)):

    df_bb = df[df['decay']=='p8_ee_Zbb_ecm91']
    df_cc = df[df['decay']=='p8_ee_Zcc_ecm91']
    df_ss = df[df['decay']=='p8_ee_Zss_ecm91'] 
    df_ud = df[df['decay']=='p8_ee_Zud_ecm91'] 
    df_Bs = dataframe[dataframe['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu']
    df_Bd = dataframe[dataframe['decay']=='p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu']

    fig, ax = plt.subplots()

    ax.set_ylabel(f'bdt_hl {labels[bdt_probs[0]]} probability')
    ax.set_xlabel(f'bdt_hl {labels[bdt_probs[1]]} probability')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)


    ax.scatter(df_bb[f"bdt_score_{bdt_probs[1]}"][0:round(len(df_bb["bdt_score_1"])/10)], df_bb[f"bdt_score_{bdt_probs[0]}"][0:round(len(df_bb["bdt_score_1"])/10)],marker='+',  label=cfg.titles['p8_ee_Zbb_ecm91'])
    ax.scatter(df_cc[f"bdt_score_{bdt_probs[1]}"][0:round(len(df_cc["bdt_score_1"])/10)], df_cc[f"bdt_score_{bdt_probs[0]}"][0:round(len(df_cc["bdt_score_1"])/10)],marker='+',  label=cfg.titles['p8_ee_Zcc_ecm91'])
    ax.scatter(df_ss[f"bdt_score_{bdt_probs[1]}"][0:round(len(df_ss["bdt_score_1"])/10)], df_ss[f"bdt_score_{bdt_probs[0]}"][0:round(len(df_ss["bdt_score_1"])/10)],marker='+',  label=cfg.titles['p8_ee_Zss_ecm91'])
    ax.scatter(df_ud[f"bdt_score_{bdt_probs[1]}"][0:round(len(df_ud["bdt_score_1"])/10)], df_ud[f"bdt_score_{bdt_probs[0]}"][0:round(len(df_ud["bdt_score_1"])/10)],marker='+',  label=cfg.titles['p8_ee_Zud_ecm91'])
    ax.scatter(df_Bs[f"bdt_score_{bdt_probs[1]}"][0:200], df_Bs[f"bdt_score_{bdt_probs[0]}"][0:200],c='b',marker='.',  label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'])
    ax.scatter(df_Bd[f"bdt_score_{bdt_probs[1]}"][0:200], df_Bd[f"bdt_score_{bdt_probs[0]}"][0:200], c='darkblue',marker='.', label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'])
    
    ax.legend()
    if outpath:
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}.pdf'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}.png'))
    else:
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}.pdf')
        fig.savefig(f'{bdt_name}_{output_file_name}_bdt_probs{bdt_probs[0]}{bdt_probs[1]}.png')
    


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

def load_bdt_and_apply(pickled_df_path = os.path.join(cfg.baseline_bdt_lh_opts['outputPath'], "bdt_lh_dataframe.pkl"), 
                        config_bdtopts = cfg.baseline_bdt_lh_opts,
                        training_round = "multiclass_baseline",
                        hps_dict_name = "default-hps",
                        features_list_name = "bdth-plus-vars",
                        bdt_label = '_lh',
                        test_train_valid = True,
                        ): # hps dict and features_list_name specift BDT used
    
    
    #path to data and outputs
    outputpath   = config_bdtopts['outputPath']
    yamlpath = cfg.fccana_opts['yamlPath']

    #Getting BDT vars for training from yaml
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'

    #path to pickled data

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
        df[f'bdt_score_{class_names[i]}'] = probabilities[:, i]

    if test_train_valid == True:
        #calculate logloss
        validation_df = df[df["sample"]==2]
        test_df = df[df["sample"]==1]
        train_df = df[df["sample"]==0]

        y_true_valid = validation_df['label']
        y_true_test = test_df['label']
        y_true_train = train_df['label']
        
        # Corresponding predicted probabilities (softmax output)
        y_pred_valid = validation_df[['bdt_score_0','bdt_score_1','bdt_score_2']]
        y_pred_test = test_df[['bdt_score_0','bdt_score_1','bdt_score_2']]
        y_pred_train = train_df[['bdt_score_0','bdt_score_1','bdt_score_2']]
        
        # Compute log loss
        validation_loss = log_loss(y_true_valid, y_pred_valid, labels=[0,1,2])
        test_loss = log_loss(y_true_test, y_pred_test, labels=[0,1,2])
        train_loss = log_loss(y_true_train, y_pred_train, labels=[0,1,2])
        diff_logloss = np.abs(validation_loss-train_loss)/ validation_loss

        print(f'validation logloss: {validation_loss}' )
        print(f'train-validation logloss fractional difference: {diff_logloss}' )


    return bdt_model, bdtname, df


#########################################
## Make some plots 
########################################

if __name__=="__main__":
    '''
    model, bdtname, dataframe = load_bdt_and_apply(pickled_df_path = os.path.join(cfg.baseline_bdt_lh_opts['outputPath'], "bdt_lh_dataframe.pkl"),
                            config_bdtopts = cfg.baseline_bdt_lh_opts,
                            training_round = "multiclass_baseline",
                            hps_dict_name = "default-hps",
                            features_list_name = "bdth-plus-vars",
                            bdt_label = '_lh')
    '''
    model, bdtname, dataframe = load_bdt_and_apply(pickled_df_path = os.path.join(cfg.baseline_bdt_lh_opts['outputPath'], "bdt_lh_dataframe.pkl"),
                            config_bdtopts = cfg.baseline_bdt_lh_opts,
                            training_round = "test_hpopt_small_sample/optimum_hps",
                            hps_dict_name = "default-hps",
                            features_list_name = "bdtlh-vars-v1",
                            bdt_label = '_lh')
    '''
    model, bdtname, dataframe = load_bdt_and_apply(pickled_df_path = os.path.join(cfg.bdt_lh_opts_nleptfail['outputPath'], "bdt_lh_dataframe_nlept_fail.pkl"), 
                        config_bdtopts = cfg.bdt_lh_opts,
                        training_round = "multiclass_baseline",#"test_hpopt_small_sample/optimum_hps",
                        hps_dict_name = "default-hps",
                        features_list_name = "bdth-plus-vars",#"bdtlh-vars-v1",
                        bdt_label = '_lh')
    '''
    outputpath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/full_prelim_cuts_500k/bdt_lh_outputs/test_hpopt_small_sample/optimum_hps'

    print('Now plotting...')

    plot_bdt_response(dataframe,bdt_name = 'BDT_lh',outpath=outputpath)
    #plot_bdt_response(dataframe,bdt_name = 'BDT_lh',outpath=outputpath, categories=['signal','light_background','heavy_background'])
    #plot_eff(dataframe,bdt_name = 'BDT_lh', outpath=outputpath)
    #plot_ROC_star(dataframe,bdt_name = 'BDT_lh', outpath=outputpath, comps = [("signal", "heavy_background")])
    
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,1],vmax=1)
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,2],vmax=1)
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [1,2],vmax=1)

    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,1],yrange=[0,0.2],xrange=[0,0.2],vmax=400)
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,2],yrange=[0,0.2],xrange=[0.8,1],vmax=400)
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [1,2],yrange=[0,0.2],xrange=[0.8,1],vmax=400)

    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,1],yrange=[0,0.005],xrange=[0,0.005],vmax=480000, output_file_name = '2dbdt_response_plot_1e-3zoom')
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [0,2],yrange=[0,0.05],xrange=[0.95,1],vmax=2000)
    #plot_2d_bdt_output(dataframe, outpath=outputpath, bdt_probs = [1,2],yrange=[0,0.05],xrange=[0.95,1],vmax=2000)

    #plot_2d_scatter(dataframe, outpath=outputpath)

    #plot_eff_combinedcut(dataframe,bdt_name = 'BDT_lh', outpath=outputpath)
    #plot_bdt_response_combinedcut(dataframe,bdt_name = 'BDT_lh',outpath=outputpath,xrange=(0.95,1))

    #post_bdt_variable_plot(dataframe,'EVT_unitThrust_z',bdt_cut='bdt_score_2>0.95', outpath=outputpath)


'''
def plot_bdt_response_combinedcut_2lph(df, bdt_name = "BDT_lh",output_file_name = "response_combinedcut_2lph" ,outpath=None,categories = ['signal','background'],labels_map = labels, pt_fmt = blobs, colors = colors,xrange=(0,1)):

    #categories (bool): which catagories to plot, must be valid category in labels_map or 'background'
    

    # plot of BDT output
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6.4,6.4))

    train_dict={}
    test_dict={}
    train_w_dict={}
    test_w_dict={}

    for cat in categories:
        if cat == 'background':

            subdf_train = df[ (df["sample"]==0) & (df["label"]!=labels_map['signal']) ]
            subdf_test = df[ (df["sample"]==1) & (df["label"]!=labels_map['signal']) ]
        
        else:
            subdf_train = df[ (df["sample"]==0) & (df["label"]==labels_map[cat]) ]
            subdf_test = df[ (df["sample"]==1) & (df["label"]==labels_map[cat]) ]


        train_dict[cat] = 1-subdf_train["bdt_score_1"].values-subdf_train["bdt_score_0"].values
        test_dict[cat] = 1-subdf_test["bdt_score_1"].values-subdf_test["bdt_score_0"].values
        train_w_dict[cat] = subdf_train["total_weight"].values
        test_w_dict[cat] = subdf_test["total_weight"].values
    
    
        # plot training sample dists
        ax[0].hist( train_dict[cat], bins=50, range=xrange, label=f'{cat} train', alpha=0.5, ec='none', fc=colors[cat], weights=train_w_dict[cat], density=True )

        # plot test sample dists
        # if you want the error need to track squared weights (probably a better way of doing this)
        n, xe = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat] )
        n2, _ = np.histogram( test_dict[cat], bins=50, range=xrange, weights=test_w_dict[cat]**2 )
        ne = n2**0.5 / n
        n, x = np.histogram( test_dict[cat], bins=50, range=xrange, density=True, weights=test_w_dict[cat] )
        ne = ne * n


        cx = 0.5*(xe[1:]+xe[:-1])
        ax[0].errorbar( cx, n, ne, fmt=pt_fmt[cat], label=f'{cat} train') 


        # now plot the residual
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat] )
        nt2, _ = np.histogram( train_dict[cat], bins=50, range=xrange, weights= train_w_dict[cat]**2 )
        nte = nt2**0.5 / nt
        nt, xe = np.histogram( train_dict[cat], bins=50, range=xrange, density=True, weights=train_w_dict[cat] )
        nte = nte * nt


        d = n - nt
        de = (ne**2 + nte**2)**0.5
        p = d / de

        
        ax[1].errorbar( cx, p, np.ones_like(p), fmt=pt_fmt[cat] )

    ax[1].axhline(0, c='k', ls='--' )   
    ax[1].set_ylabel('Pull')
    ax[0].set_xlabel(f'{bdt_name} XGBoost 1-2P(light)-P(heavy)]')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].set_yscale('log')
    fig.tight_layout()

    if xrange==(0,1):
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}.pdf")
    else:
        if outpath:
            #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
            fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed.pdf'))
        else:
            #fig.savefig(f"{output_file_name}.png")
            fig.savefig(f"{bdt_name}_{output_file_name}_ncat{len(categories)}_zoomed.pdf")


# efficiency plot (on total sample)
def plot_eff_combinedcut_2lph(df, bdt_name = "BDT_lh",output_file_name = 'efficiency_plot_combinedcut2lph',outpath=None):
    fig, ax = plt.subplots()
    for decay in df["decay"].unique():
        subf = df[ df["decay"]==decay ]
        mva_scores = 1-subf["bdt_score_1"].values -2*subf["bdt_score_0"].values
        weights = subf["total_weight"].values

        sorted_indices = np.argsort( mva_scores )
        sorted_scores = mva_scores[sorted_indices]
        sorted_weights = weights[sorted_indices]

        total_weight = np.sum( sorted_weights ) 
        cumalative_weights = np.cumsum( sorted_weights[::-1] )[::-1] # reverse order for efficiency above cut
        efficiency = cumalative_weights / total_weight

        ax.plot( sorted_scores, efficiency, label=decay )

    ax.legend()
    ax.set_xlabel(f'{bdt_name} XGBoost 1-2P(light)-P(heavy))]')
    ax.set_ylabel('Efficiency')
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', linestyle='-', color='0.7', linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+'.png'))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}.pdf'))
    else:
        #fig.savefig(f"{output_file_name}.png")
        fig.savefig(f'{bdt_name}_{output_file_name}.pdf')
    #ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', alpha = 0.2) #need minor ticks on to see these
    ax.set_xlim(0.95,1.002)
    if outpath:
        #fig.savefig(os.path.join(outpath,output_file_name+"_zoomedin.png"))
        fig.savefig(os.path.join(outpath,f'{bdt_name}_{output_file_name}_zoomedin.pdf'))
    else:
        #fig.savefig(f"{output_file_name}_zoomedin.png")
        fig.savefig(f'{bdt_name}_{output_file_name}_zoomedin.pdf')

'''