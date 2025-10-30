# Train BDT using prelimcuts files

import os
import glob
import sys

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb  # Has to be imported first to avoid conflicts with PyROOT
from tabulate import tabulate
from datetime import timedelta
from yaml import safe_load, YAMLError, dump
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
import optuna


# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg
import efficiency_finder
import bdt_plotter as bp

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


#Define function that does training
def train_bdt(pickled_df_fname = "bdt_lh_dataframe.pkl", 
              config_bdtopts = cfg.bdt_lh_opts,
              training_round = "multiclass_baseline",
              hps_dict_name = "default-hps",#if not using default, name of hp config in config 
              features_list_name = "bdth-plus-vars",
              bdt_label = '_lh',
              hp_opt=None,
              opt_hp_val_path=None): #path to optimum hps want to use if hp_opt='use_optimised_hps'
    
    
    ## PREPROCESSING AND CREATING df

    print(f"{30*'-'}")
    print(f"BDT{bdt_label} TRAINING")
    print(f"{30*'-'}\n")
    print("Initialising...")

    # Load configuration
    plt.style.use(os.path.abspath(os.path.join(cfg.FCCAnalysesPath, 'fcc.mplstyle')))

    #path to data and outputs
    outputpath   = set_outputpath(config_bdtopts['outputPath'])
    yamlpath     = check_inputpath(cfg.fccana_opts['yamlPath'])

    #Getting BDT vars for training from yaml
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'

    #loading saved data
    pickled_df_path = os.path.join(outputpath, pickled_df_fname)
    df = pd.read_pickle(pickled_df_path)

    # matrix of input vars
    x_train = df[ df["sample"]==0][bdtvars].head(500000)
    x_test  = df[ df["sample"]==1][bdtvars].head(500000)
    x_valid = df[ df["sample"]==2][bdtvars].head(500000)

    # array of target
    y_train = df[ df["sample"]==0][ "label" ].head(500000)
    y_test  = df[ df["sample"]==1][ "label" ].head(500000)
    y_valid = df[ df["sample"]==2][ "label" ].head(500000)

    # array of weights
    w_train = df[ df["sample"]==0][ "total_weight_muliclass" ].head(500000)
    w_test  = df[ df["sample"]==1][ "total_weight_muliclass" ].head(500000)
    w_valid = df[ df["sample"]==2][ "total_weight_muliclass" ].head(500000)


    ######################################################
    #Training itself
    #######################################################

    print("Currently no k-folding, this probably okay, using test:train split instead")


    if hp_opt=='run_optimisation':
        #using optuna for more efficient bayesian optimisation (compared to gridsearch)


        # Use Optuna to fo hp optimisation using bayesian optimisation
        # define a function that optuna is going to try and optimize
        # you can define min and max for each hyperpar
        # you can also pass log = True to some of them (e..g learning rate) so it knows to move around the space logartihmically not linearly
        # instead of search in a grid it will move around in a more optimal way
        # in this case it will train the BDT and return the log loss
        def objective( trial ):
            # example hyperparameters to try
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }

            bdt = xgb.XGBClassifier( objective='multi:softprob', eval_metric= 'mlogloss',**params) #I believe mlogloss is the default (also known as cross entropy)
            bdt.set_params(early_stopping_rounds=10)

            # Train XGBoost model
            bdt.fit( x_train, y_train, 
                sample_weight=w_train, 
                eval_set=[(x_test, y_test)], 
                sample_weight_eval_set=[w_test], 
                verbose=False)

            # Predict probabilities
            y_pred = bdt.predict_proba(x_valid)

            # Compute log loss
            return log_loss(y_valid, y_pred) #don't need to use one-hot encodeing for labels (ie. yvalid)
        
        # Create Optuna study and optimize
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        # Best hyperparameters
        best_hps = study.best_params
        print("Best hyperparameters:", best_hps)

        #to look at from here
        print(f"Best log loss score = {study.best_value:.5f}")

        model_folder  = set_outputpath(os.path.join(outputpath,training_round,'optimum_hps'))
        hp_file_path =  os.path.join(model_folder,f'opt_hps_bdt{bdt_label}_{training_round}.yaml')

        ### Save optimum combination of hyperparameters
        with open(hp_file_path, 'w') as outfile:
            dump(best_hps, outfile)
        print(f"----> INFO: Optimum hyperparameters saved to")
        print(f"{15*' '}{hp_file_path}")

        #now training and saving model with best hps

        bdt = xgb.XGBClassifier( objective='multi:softprob', eval_metric= 'mlogloss',**best_hps) #I believe mlogloss is the default (also known as cross entropy) 
    
        print(f"\n----> INFO: Training using {best_hps}")

        print("\n Training model")
        bdt.fit( x_train, y_train, 
                sample_weight=w_train, 
                eval_set=[(x_test, y_test)], 
                sample_weight_eval_set=[w_test], 
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
        bdt = xgb.XGBClassifier( objective='multi:softprob', eval_metric= 'mlogloss',**best_hps) #I believe mlogloss is the default (also known as cross entropy)


        print(f"\n----> INFO: Training using {best_hps}")

        print("\n Training model")
        bdt.fit( x_train, y_train, 
                sample_weight=w_train, 
                eval_set=[(x_test, y_test)], 
                sample_weight_eval_set=[w_test], 
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
        
        print("CURRENTLY NO HP OPTIMISATION OR CROSS VALIDATION")

 
        ## TRAINING OF BDT

        ## Currently no cross validation - potentially TO IMPLEMENT LATER

        #nb. could add early_stopping_rounds=10, eval_metric="auc",  - if specify first want to specify second as this is metric used for early stopping

        bdt.set_params(n_estimators=default_hps['n_estimators'],
                        learning_rate=default_hps['learning_rate'],
                        max_depth=default_hps['max_depth'],
                        gamma=default_hps['gamma'],
                        min_child_weight=default_hps['min_child_weight'],
                        max_delta_step=default_hps['max_delta_step'],
                        subsample=default_hps['subsample']) 
        
        print(f"\n----> INFO: Training using {default_hps}")

        #let's finally run the training
        print("\n Training model")
        bdt.fit( x_train, y_train, 
                sample_weight=w_train, 
                eval_set=[(x_test, y_test)], 
                sample_weight_eval_set=[w_test], 
                verbose=10 )


        # now put it's predictions back into the frame 
        # The predict_proba() method returns a 2D array where each row corresponds to a sample
        # each column represents the probability of that sample belonging to a particular class.
        #df['bdt_score'] = bdt.predict_proba( df[bdtvars] )[:,1]

        class_names = bdt.classes_
        probabilities = bdt.predict_proba(df[bdtvars])
        for i in range(probabilities.shape[1]):
            df[f'bdt_score_{class_names[i]}'] = probabilities[:, i]


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

        #bp.plot_simple_ROC(df,bdt_name = f"BDT{bdt_label}",output_file_name = "ROC",outpath=model_folder)
        #bp.plot_bdt_response(df, bdt_name = f"BDT{bdt_label}",output_file_name = "response" ,outpath=model_folder)
        #bp.plot_eff(df, bdt_name =f"BDT{bdt_label}",output_file_name = 'efficiency_plot',outpath=model_folder)

        return bdt, df




