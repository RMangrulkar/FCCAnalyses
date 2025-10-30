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
import json


# Path to config.py and variable_plotter.py
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))

import config as cfg 
import efficiency_finder
import plotters.bdt_plotter_multiclass as bp

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
              config_bdtopts = cfg.optimised_bdt_lh_opts,
              training_round = "multiclass_baseline",
              hps_dict_name = "default-hps",#if not using default, name of hp config in config 
              bdt_label = '_lh',
              hp_opt=None,#'run_optimisation','use_optimised_hps' or none 
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
    features_list_name = config_bdtopts["mvaBranchList"]
    bdtvars      = vars_fromyaml(yamlpath, features_list_name)
    bdtname      = f'BDT{bdt_label}_{hps_dict_name}_{features_list_name}'

    #loading saved data
    pickled_df_path = os.path.join(outputpath, pickled_df_fname)
    df = pd.read_pickle(pickled_df_path)

    # matrix of input vars 
    x_train = df[ df["sample"]==0][bdtvars]#.head(100000)
    x_test  = df[ df["sample"]==1][bdtvars]#.head(100000)
    x_valid = df[ df["sample"]==2][bdtvars]#.head(100000)

    # array of target
    y_train = df[ df["sample"]==0][ "label" ]#.head(100000)
    y_test  = df[ df["sample"]==1][ "label" ]#.head(100000)
    y_valid = df[ df["sample"]==2][ "label" ]#.head(100000)

    # array of weights
    w_train = df[ df["sample"]==0][ "total_weight_muliclass" ]#.head(100000)
    w_test  = df[ df["sample"]==1][ "total_weight_muliclass" ]#.head(100000)
    w_valid = df[ df["sample"]==2][ "total_weight_muliclass" ]#.head(100000)


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

        def objective( trial ):
            # example hyperparameters to try
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
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
            y_train_pred = bdt.predict_proba(x_train)

            valid_logloss = log_loss(y_valid, y_pred) #don't need to use one-hot encodeing for labels (ie. yvalid)
            diff_logloss = np.abs(valid_logloss-log_loss(y_train, y_train_pred))/ valid_logloss

            # add in pruning to stop trials with large log loss difference
            # Define stopping threshold
            threshold = 0.1  # loosish for now!

            # Check if any diff_logloss exceeds the threshold → Stop trial early
            if diff_logloss > threshold:
                raise optuna.TrialPruned()  # Stop this trial


            # Compute log loss and difference between train and valoidation log loss
            return valid_logloss, diff_logloss
        
        targetNames = ["valid_logloss", "diff_logloss"] 
 
        # Create Optuna study and optimize
        study = optuna.create_study(directions=['minimize', 'minimize'])
        study.set_metric_names(targetNames)
        study.optimize(objective, n_trials=40)
        
        
        model_folder  = set_outputpath(os.path.join(outputpath,training_round,'optimum_hps'))
        hp_file_path =  os.path.join(model_folder,f'opt_hps_bdt{bdt_label}_{training_round}.yaml')


        best_trials = study.best_trials
        #print("Best trials:", best_trials)
        ### Save optimum trials
        with open(hp_file_path, 'w') as outfile:
            dump(best_trials, outfile)
        print(f"----> INFO: Optimum trials saved to")
        print(f"{15*' '}{hp_file_path}")

        print("Number of trials: ", len(study.trials))
        print("Number of best (Pareto front) trials: ", len(study.best_trials))

        ## plot hp importance
        plt.figure(figsize=(15, 9))
        optuna.visualization.matplotlib.plot_param_importances(study,target=lambda t: t.values[0], target_name="valid_logloss")
        output_plot = os.path.join(model_folder,"hp_importance_validation_logloss.pdf")
        plt.savefig(output_plot)
        plt.savefig(output_plot.replace(".pdf", ".png"))
        plt.close()

        plt.figure(figsize=(15, 9))
        optuna.visualization.matplotlib.plot_param_importances(study,target=lambda t: t.values[1], target_name="diff_logloss")
        output_plot = os.path.join(model_folder,"hp_importance_logloss_difference.pdf")
        plt.savefig(output_plot)
        plt.savefig(output_plot.replace(".pdf", ".png"))
        plt.close()

        ## plot optimization history
        plt.figure()
        optuna.visualization.matplotlib.plot_optimization_history(
        study, target=lambda t: t.values[0], target_name="valid_logloss")
        output_plot = os.path.join(model_folder,"optimisation_valid_logloss.pdf")
        plt.savefig(output_plot)
        plt.savefig(output_plot.replace(".pdf", ".png"))
        plt.close()

        plt.figure()
        optuna.visualization.matplotlib.plot_optimization_history(
            study, target=lambda t: t.values[1], target_name="diff_logloss")
        output_plot = os.path.join(model_folder,"optimisation_diff_logloss.pdf")
        plt.savefig(output_plot)
        plt.savefig(output_plot.replace(".pdf", ".png"))
        plt.close()

        fig = optuna.visualization.plot_pareto_front(study, target_names=["valid_logloss","diff_logloss"])
        output_plot = os.path.join(model_folder,"pareto-plot.pdf")
        fig.write_image(output_plot)
        fig.write_image(output_plot.replace(".pdf", ".png"))


        #save best model inputs
        study_xgb_df = study.trials_dataframe()
        assert isinstance(study_xgb_df, pd.DataFrame)
        print(study_xgb_df.sort_values(by='values_valid_logloss', ascending=True).head(5))
        study_xgb_df.to_csv(os.path.join(model_folder,"optuna_trials.csv"), index=False)

        ## some way of determining best trial!
        # Impose some constraint on loss_diff, take best solution from there
        lossdiffThreshold = 0.04
        threshold_df = study_xgb_df.query(f"values_diff_logloss<{lossdiffThreshold}")
        best_trial_number = threshold_df.loc[threshold_df[f"values_valid_logloss"].idxmax()].number

        best_trial = next((x for x in study.best_trials if x.number == best_trial_number), None)
        print(best_trial)
        optuna_results = best_trial.params
        optuna_results.update(best_trial.user_attrs)
        outputFileName = os.path.join(model_folder,"optuna_results.json")

        with open(outputFileName, "w") as f:
            json.dump(optuna_results, f, indent=4)

        study_xgb_df.to_csv(os.path.join(model_folder,"optuna_trials.csv"), index=False)


        #now training and saving model with best hps

        bdt = xgb.XGBClassifier( objective='multi:softprob', eval_metric= 'mlogloss',**best_hps) #I believe mlogloss is the default (also known as cross entropy) 
        bdt.set_params(early_stopping_rounds=10)
    
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
        bdt.set_params(early_stopping_rounds=10)


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
        
        else:
            print("CURRENTLY NO HP OPTIMISATION")

 
        ## TRAINING OF BDT

        ## nb. no cross validation

        bdt = xgb.XGBClassifier( objective='multi:softprob', eval_metric= 'mlogloss',**default_hps) 
        bdt.set_params(early_stopping_rounds=10)

        
        print(f"\n----> INFO: Training using {default_hps}")

        #run the training
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

        bp.plot_ROC_star(df,bdt_name = f"BDT{bdt_label}",output_file_name = "ROC_star",outpath=model_folder)
        bp.plot_bdt_response(df, bdt_name = f"BDT{bdt_label}",output_file_name = "response" ,outpath=model_folder)
        bp.plot_eff(df, bdt_name =f"BDT{bdt_label}",output_file_name = 'efficiency_plot',outpath=model_folder)


        #also calculating and saving logloss as a check
        # Predict probabilities
        y_pred_valid = bdt.predict_proba(x_valid)
        print(f'validation log loss = {log_loss(y_valid, y_pred_valid)}')
        y_pred_test = bdt.predict_proba(x_test)
        print(f'Test log loss = {log_loss(y_test, y_pred_test)}')
        y_pred_train = bdt.predict_proba(x_train)
        print(f'Train log loss = {log_loss(y_train, y_pred_train)}')

        with open(os.path.join(outputpath,training_round,f'{bdtname}_training_info.log'), 'a') as log_file:
            log_file.write(f'validation log loss = {log_loss(y_valid, y_pred_valid)}\n')
            log_file.write(f'test log loss = {log_loss(y_test, y_pred_test)}\n')
            log_file.write(f'train log loss = {log_loss(y_train, y_pred_train)}\n')


        return bdt, df






train_bdt(pickled_df_fname = "bdt_lh_dataframe.pkl", 
              config_bdtopts = cfg.optimised_bdt_lh_opts,
              training_round = "default-plus-hps",
              hps_dict_name = "default-plus-hps",#if not using default, name of hp config in config 
              bdt_label = '_lh',
              hp_opt=None,
              opt_hp_val_path=None)

'''
train_bdt(pickled_df_fname = "bdt_lh_dataframe.pkl", 
              config_bdtopts = cfg.optimised_bdt_lh_opts,
              training_round = "test_hp_optimisation",
              bdt_label = '_lh',
              hp_opt='run_optimisation',
              opt_hp_val_path=None)

'''