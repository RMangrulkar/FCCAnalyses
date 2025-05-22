# Train BDT using prelimcuts files

import os
import glob
import sys
import re

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
from tabulate import tabulate
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.metrics import roc_curve, auc, roc_auc_score



# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg 
import efficiency_finder
import bdt_plotter  as bp
import post_bdt_application as post_bdt

#import bdt_plotter as bdtplt


ROOT.EnableImplicitMT()

##########################################################
# function to retrive lists from yaml and check file paths
###########################################################
# Return list of variables to use in the bdt as a python list
def vars_fromyaml(path, bdtlist):
    with open(path) as stream:
        try:
            file = safe_load(stream)
            bdtvars = file[bdtlist]
        except YAMLError as exc:
            print(exc)
    return bdtvars

def check_inputpath(inputpath):
    if not os.path.exists(inputpath):
        raise FileNotFoundError(f"{inputpath} does not exist")
    return inputpath


def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath

def add_to_numbers(text):
    return re.sub(r'\d+\.\d+|\d+', lambda x: str(round(float(x.group()) + 0.1, 1)), text)

################################################################
## create tau_training df from all data
###############################################################


#things need:
    #BDT to pass and apply
    #test, train, validation split
    #BDT cut want to apply
    #number of events want to train on for each sample
    #vars list but i think this can be taken out for training - on just tau and sample should be too big

def create_multisample_df(df_folder_path = cfg.bdttau_opts["inputPath"],
                          samples = cfg.bdttau_opts["signalAllocation"] + cfg.bdttau_opts["backgroundAllocation"],
                          config_bdtopts = cfg.bdt_lh_opts, #for bdt applying to data
                          training_round = "multiclass_baseline",
                          hps_dict_name = "default-hps",
                          features_list_name = "bdth-plus-vars",
                          bdt_label = '_lh'):

    #get paths and vars from config
    branching_fractions = cfg.branching_fractions

    df_dict={}
    df_sample_dict={}

    print(f'creating dataframe of decays requested for training: {samples}')

    for sample in samples:

        df_folder = os.path.join(df_folder_path,sample)
        files = os.listdir(df_folder)

        for f in files:
            fpath = os.path.join(df_folder,f)
            model, bdtname, dataframe_chunk = post_bdt.load_bdt_and_apply(pickled_df_path = fpath, 
                                config_bdtopts = config_bdtopts,
                                training_round = training_round,
                                hps_dict_name = hps_dict_name,
                                features_list_name = features_list_name,
                                bdt_label = bdt_label)
            df_sample_dict[f] = dataframe_chunk

        dataframe = pd.concat( [df_sample_dict[s] for s in files], ignore_index=True )
        weight = branching_fractions[sample][0] * dataframe["eff_presel"][0] #need zero index for bf as branching fractions is a tuple in the yaml
        dataframe["w1"] = weight / len(dataframe)
        df_dict[sample] = dataframe

    df = pd.concat( [df_dict[s] for s in samples], ignore_index=True )

    return df


def train_bdttau(df, #df should be a dataframe with the bdt you want to cut on already applied
                 train_frac=0.75,
                 test_frac=0.125,
                 bdt_cuts='(bdt_lh_score_1<0.4)&(bdt_lh_score_0<0.4)',
                 n_events_per_type=500000, 
                 config_bdtopts = cfg.bdttau_opts,
                 training_round = "baseline",
                 hps_dict_name = "default-hps-tau",#if not using default, name of hp config in config
                 hp_opt=None,
                 opt_hp_val_path=None #path to optimum hps want to use if hp_opt='use_optimised_hps'
                ):
    
    #get bdt training vars
    bdttau_training_vars = config_bdtopts["mvaBranchList"]
    #read yaml and get list
    yamlpath = cfg.fccana_opts['yamlPath']
    bdtvars = vars_fromyaml(yamlpath,bdttau_training_vars)

    print(bdtvars)

    #get paths and variables from config
    bdt_label = config_bdtopts['label']
    outputpath   = set_outputpath(config_bdtopts['outputPath'])
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{bdttau_training_vars}'

    #avoiding weights altogether as one bkg and signal 50:50 - just using 250000 of each signal

    #getting training decays from config decay list
    signal_decays =  config_bdtopts["signalAllocation"]
    background_decays = config_bdtopts["backgroundAllocation"]
    training_decays = signal_decays + background_decays 


    print(f"----> INFO: Using signal decays:")
    print(f"{15*' '}{signal_decays}")
    print(f"----> INFO: Using background decays:")
    print(f"{15*' '}{background_decays}")

    # Cut on BDTlh
    df_cut = df.copy().query(bdt_cuts)

   

    #selecting number of events of each type
    signal_events = {decay: df_cut[df_cut['decay']==decay].sample(n=round(n_events_per_type/len(signal_decays)), random_state=25) for decay in signal_decays}
    
    for decay in background_decays:
        if round(n_events_per_type/len(background_decays)) <= len(df_cut[df_cut['decay']==decay]):
            print(len(df_cut[df_cut['decay']==decay]))
            continue
        else:
            raise ValueError(f'Insuffiencient data to train bdt on, only {len(df_cut[df_cut["decay"]==decay])} events remaining after initial BDT cut - please go and loosen cuts')

    bkg_events = {decay: df_cut[df_cut['decay']==decay].sample(n=round(n_events_per_type/len(background_decays)), random_state=25) for decay in background_decays}
    events = signal_events | bkg_events

    training_data = pd.concat( [events[s] for s in training_decays], ignore_index=True)


    #  label background and signal events as 0 and 1 for classifier 
    def labeller(dec):
        if dec in signal_decays:
            return 1
        elif dec in background_decays:
            return 0
        else:
            raise ValueError('Expect all data to be either light or heavy background if not signal')

    training_data["label"] = training_data["decay"].apply(labeller)

    # now shuffle the whole dataframe around to avoid any funny biases
    # do this with a random seed so it's reproducible
    training_data = training_data.sample(frac=1, random_state=25).reset_index(drop=True)

    # now create labels for the train, test and validation split
    # we will give them indices train==0, test==1, validation==2
    # use a random seed for this so it's reproducible
    np.random.seed(2025)

    valid_frac = 1-test_frac - train_frac
    sample_indices = np.random.choice( [0,1,2], p=[train_frac, test_frac, valid_frac], size=len(training_data) )
    training_data["sample"] = sample_indices

    # matrix of input vars
    x_train = training_data[ training_data["sample"]==0][bdtvars]
    x_test  = training_data[ training_data["sample"]==1][bdtvars]
    x_valid = training_data[ training_data["sample"]==2][bdtvars]

    # array of target
    y_train = training_data[ training_data["sample"]==0][ "label" ]
    y_test  = training_data[ training_data["sample"]==1][ "label" ]
    y_valid = training_data[ training_data["sample"]==2][ "label" ]

    # array of weights - dont need any weights


    print("\n----> INFO: Preprocessing done")
    print(f"{15*' '}Using {len(x_train):>8} events to train ({100*train_frac:.1f}% of total)")
    print(f"{15*' '}Using {len(x_test):>8} events to  test ({100*test_frac:.1f}% of total)")
    print(f"\n{30*'-'}\n")


    if hp_opt=='run_optimisation':

        # Use Optuna to fo hp optimisation using bayesian optimisation
        # define a function that optuna is going to try and optimize
        # you can define min and max for each hyperpar
        # you can also pass log = True to some of them (e..g learning rate) so it knows to move around the space logartihmically not linearly
        # instead of search in a grid it will move around in a more optimal way
        # in this case it will train the BDT and return the AUC
        def objective( trial ):
            # example hyperparameters to try
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
            
            bdt = xgb.XGBClassifier( objective='binary:logistic', ** params)
            bdt.set_params(early_stopping_rounds=10)
            bdt.fit(x_train, y_train,  
                eval_set=[(x_test, y_test)],  
                verbose=False )
            
            # Predict probabilities
            y_pred = bdt.predict_proba(x_valid)[:,1]

            # Compute AUC
            return roc_auc_score(y_valid, y_pred) 
        
        # Create Optuna study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        # Best hyperparameters
        best_hps = study.best_params
        print("Best hyperparameters:", best_hps)

        #to look at from here
        print(f"Best AUC score = {study.best_value:.5f}")

        model_folder  = set_outputpath(os.path.join(outputpath,training_round,'optimum_hps'))
        hp_file_path =  os.path.join(model_folder,f'opt_hps_bdt{bdt_label}_{training_round}.yaml')


        ### Save optimum combination of hyperparameters
        with open(hp_file_path, 'w') as outfile:
            dump(best_hps, outfile)
        print(f"----> INFO: Optimum hyperparameters saved to")
        print(f"{15*' '}{hp_file_path}")

        #now training and saving model with best hps

        bdt = xgb.XGBClassifier( objective='binary:logistic', **best_hps)
        bdt.set_params(early_stopping_rounds=10)

        print(f"\n----> INFO: Training using {best_hps}")

        print("\n Training model")
        bdt.fit( x_train, y_train, 
                eval_set=[(x_test, y_test)], 
                verbose=10 )
        
        # get the feature importance
        importance_indices = np.argsort(bdt.feature_importances_)
        sorted_features = bdt.feature_names_in_[importance_indices[::-1]]
        sorted_importances = bdt.feature_importances_[importance_indices[::-1]]
        print(f"Feature importance type: {bdt.importance_type}")
        print( "Feature Importance:")
        print( tabulate( zip( sorted_features, sorted_importances ) ) )

        # save the model to a file for use later
        bdt.save_model(os.path.join(model_folder,f"{bdtname}.json"))

        # Write key info about df to log file
        with open(os.path.join(model_folder,f'{bdtname}_optimum_hps_training_info.log'), 'a') as log_file:
            log_file.write(f'Feature Importance: { tabulate( zip( sorted_features, sorted_importances ) )}\n')

        # Write key info about df to log file
        with open(os.path.join(model_folder,f'{bdtname}_optimum_hps_training_vars.log'), 'a') as log_file:
            log_file.write(f'bdt_training_vars: {bdtvars}\n')


    elif hp_opt=='use_optimised_hps':

        if not opt_hp_val_path:
            raise ValueError('To use saved hps from optimisation a path to the file must be specified, else use a dictionary from config and set hp_opt to None')
        else:
            path_hps_at = opt_hp_val_path

        
        # section to make to use saved hps
        with open(path_hps_at, 'r') as stream:
            best_hps = safe_load(stream)
        print(f"----> INFO: Loading hyperparameters from")
        print(f"{15*' '}{path_hps_at}")

        #now training and saving model with best hps
            # define Bdt to train
        bdt = xgb.XGBClassifier( objective='binary:logistic', **best_hps)
        bdt.set_params(early_stopping_rounds=10)


        print(f"\n----> INFO: Training using {best_hps}")

        print("\n Training model")
        bdt.fit( x_train, y_train,  
                eval_set=[(x_test, y_test)], 
                verbose=10 )
        
        # get the feature importance
        importance_indices = np.argsort(bdt.feature_importances_)
        sorted_features = bdt.feature_names_in_[importance_indices[::-1]]
        sorted_importances = bdt.feature_importances_[importance_indices[::-1]]
        print(f"Feature importance type: {bdt.importance_type}")
        print( "Feature Importance:")
        print( tabulate( zip( sorted_features, sorted_importances ) ) )

        # save the model to a file for use later
        model_folder  = set_outputpath(os.path.join(outputpath,training_round,'optimum_hps'))
        bdt.save_model(os.path.join(model_folder,f"{bdtname}.json"))

        # Write key info about df to log file
        with open(os.path.join(outputpath,training_round,f'{bdtname}_optimum_hps_training_info.log'), 'a') as log_file:
            log_file.write(f'Feature Importance: { tabulate( zip( sorted_features, sorted_importances ) )}\n')

        # Write key info about df to log file
        with open(os.path.join(outputpath,training_round,f'{bdtname}_optimum_hps_training_vars.log'), 'a') as log_file:
            log_file.write(f'bdt_training_vars: {bdtvars}\n')


    else:    

        # Set Default hyperparameters - updated with those input
        default_hps = cfg.hp_opts['default-hps']

        if hps_dict_name != 'default_hps':
            hps =  cfg.hp_opts[hps_dict_name]
            # Update default hyperparameters with any provided 
            default_hps.update(hps)

        #Possible hps to try
            #"n_estimators": [100, 150, 200],
            #"learning_rate": [0.1, 0.3],
            #"max_depth": [3, 5, 7],
            #"subsample": [0.7, 1.0],
            #"gamma": [0, 0.1, 0.2],
            #"min_child_weight": [1, 5],


        ## TRAINING OF BDT

        ## Currently no cross validation - potentially TO IMPLEMENT LATER

        print("CURRENTLY NO HP OPTIMISATION OR CROSS VALIDATION")
        # now let's train it
        bdt = xgb.XGBClassifier( objective='binary:logistic') #xgb default objective
        #nb. could add early_stopping_rounds=10, eval_metric="auc",  - if specify first want to specify second as this is metric used for early stopping

        bdt.set_params(n_estimators=default_hps['n_estimators'],
                        learning_rate=default_hps['learning_rate'],
                        max_depth=default_hps['max_depth'],
                        gamma=default_hps['gamma'],
                        min_child_weight=default_hps['min_child_weight'],
                        max_delta_step=default_hps['max_delta_step'],
                        subsample=default_hps['subsample']) 
        
        print(f"\n----> INFO: Training using {default_hps}")

        print("\n Training model")
        bdt.fit(x_train, y_train,  
                eval_set=[(x_test, y_test)],  
                verbose=10 )

        # get the feature importance
        importance_indices = np.argsort(bdt.feature_importances_)
        sorted_features = bdt.feature_names_in_[importance_indices[::-1]]
        sorted_importances = bdt.feature_importances_[importance_indices[::-1]]
        print(f"Feature importance type: {bdt.importance_type}")
        print( "Feature Importance:")
        print( tabulate( zip( sorted_features, sorted_importances ) ) )

        # save the model to a file for use later
        model_folder  = set_outputpath(os.path.join(outputpath,training_round))
        bdt.save_model(os.path.join(model_folder,f"{bdtname}.json"))

        # Write key info about df to log file
        with open(os.path.join(outputpath,training_round,f'{bdtname}_training_info.log'), 'a') as log_file:
            log_file.write(f'Feature Importance: { tabulate( zip( sorted_features, sorted_importances ) )}\n')

        # Write key info about df to log file
        with open(os.path.join(outputpath,training_round,f'{bdtname}_training_vars.log'), 'a') as log_file:
            log_file.write(f'bdt_training_vars: {bdtvars}\n')

    # now put it's predictions back into the frame 
    # The predict_proba() method returns a 2D array where each row corresponds to a sample
    # each column represents the probability of that sample belonging to a particular class.
    training_data[f'bdt{bdt_label}_score'] = bdt.predict_proba( training_data[bdtvars] )[:,1]

    #add weights of one to training data so that the plotter doesnt throw a hissy fit
    training_data["total_weight"] = np.ones(len(training_data[f'bdt{bdt_label}_score']))



    bp.plot_simple_ROC(training_data,bdtlabel = "tau",output_file_name = "ROC",outpath=model_folder)
    bp.plot_bdt_response(training_data, bdtlabel = "tau",output_file_name = "response" ,outpath=model_folder)
    bp.plot_eff(training_data, bdtlabel = "tau",output_file_name = 'efficiency_plot',outpath=model_folder)

    # Save the DataFrame to a pickle file for now - can look at cuts
    df.to_pickle(os.path.join(model_folder,f'bdt{bdt_label}_training_dataframe.pkl'))
    print("DataFrame saved successfully!")


    return bdt, training_data


if __name__=="__main__":

    df= create_multisample_df(df_folder_path = cfg.bdttau_opts["inputPath"],
                          samples = cfg.bdttau_opts["signalAllocation"] + cfg.bdttau_opts["backgroundAllocation"],
                          config_bdtopts = cfg.bdt_lh_opts, #for bdt applying to data
                          training_round = "multiclass_baseline",
                          hps_dict_name = "default-hps",
                          features_list_name = "bdth-plus-vars",
                          bdt_label = '_lh')
    
    train_bdttau(df, 
                 train_frac=0.75,
                 test_frac=0.125,
                 bdt_cuts='(bdt_lh_score_1<0.4)&(bdt_lh_score_0<0.4)',
                 n_events_per_type=500000, 
                 config_bdtopts = cfg.bdttau_opts,
                 training_round = "baseline_hpopt",
                 hps_dict_name = "default-hps-tau",#if not using default, name of hp config in config
                 hp_opt='run_optimisation',
                 opt_hp_val_path=None #path to optimum hps want to use if hp_opt='use_optimised_hps'
                )


