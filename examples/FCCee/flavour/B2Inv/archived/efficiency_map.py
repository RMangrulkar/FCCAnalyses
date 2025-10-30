# efficiency_map.py
# Creates a 2D map of BDT1 and BDT2 efficiencies for each sample,
#     as well as a 1D map of BDTComb efficiency for each sample

import os

import pandas as pd
from time import time
from datetime import timedelta

import config as cfg
from efficiency_finder import get_efficiencies

if __name__ == "__main__":
    
    ##############################
    ## Initialisation
    ##############################
    start = time()

    print(f"{30*'-'}")
    print(f"EFFICIENCY MAP GENERATOR")
    print(f"{30*'-'}\n")
    print(f"Initialising...") 

    outputpath = cfg.poststage2_opts['outputPath']
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print(f"----> Output directory does not exist, creating")
        print(f"{15*' '}")
    else:
        print(f"----> INFO: Saving results to")
        print(f"{15*' '}{outputpath}")

    ##############################
    ## Choice of BDT cuts
    ##############################
    bdt2_vals =    [0,   0.2, 0.4, 0.6, 0.7, 0.8, 0.9,  0.95, 0.99]
    bdt1_vals =    [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    bdtComb_vals = [0,   0.2, 0.4, 0.6, 0.7, 0.8, 0.9,  0.95, 0.99]
    
    print(f"\n{30*'-'}\n")
    print(f"----> INFO: Using BDT1 cuts:")
    print(f"{15*' '}{bdt1_vals}")
    print(f"----> INFO: Using BDT2 cuts:")
    print(f"{15*' '}{bdt2_vals}")
    print(f"----> INFO: Using BDTComb cuts:")
    print(f"{15*' '}{bdtComb_vals}")

    ##############################
    ## List of cut expressions to pass to uproot
    ##############################
    #map_bdt1 =    [f"EVT_MVA1 >= {i}" for i in bdt1_vals]
    #map_bdt2 =    [f"EVT_MVA2 >= {i}" for i in bdt2_vals]
    map_2d =      [f"(EVT_MVA1 >= {i}) & (EVT_MVA2 >= {j})" for j in bdt2_vals for i in bdt1_vals]
    map_bdtComb = [f"EVT_MVAComb >= {i}" for i in bdtComb_vals]
    
    # This is for the dataframe, nested loops must be in the same order as `map_2d`
    # Surely there's a better way to do this
    cuts_2d_1 = [i for j in bdt2_vals for i in bdt1_vals]
    cuts_2d_2 = [j for j in bdt2_vals for i in bdt1_vals]

    ##############################
    ## Get efficiencies and errors
    ##############################
    #databdt1 = get_efficiencies(inputtype='stage2', cut=map_bdt1, verbose=False, save=os.path.join(outputpath, 'bdt1_efficiency_dict.csv'))
    #databdt2 = get_efficiencies(inputtype='stage2', cut=map_bdt2, verbose=False, save=os.path.join(outputpath, 'bdt2_efficiency_dict.csv')) 
    data2d = get_efficiencies(inputtype='stage2', cut=map_2d, verbose=False, save=os.path.join(outputpath, '2d_efficiency_dict.csv'))
    databdtComb = get_efficiencies(inputtype='stage2', cut=map_bdtComb, verbose=False, save=os.path.join(outputpath, 'bdtComb_efficiency_dict.csv'))
    
    print(f"\n{30*'-'}\n")
    print(f"----> INFO: Efficiency dictionaries saved")
    print(f"{15*' '}2d_efficiency_dict.csv, bdtComb_efficiency_dict.csv")

    ##############################
    ## Construct dataframes
    ##############################
    # Assumes that the keys in `data2d` are of the type `sample_eff` and `sample_err`
    df_2d = pd.DataFrame(data2d, index=None, columns=list(data2d.keys()))
    df_2d.rename(columns={key: cfg.sample_shorthand[key[:-4]]+key[-4:] for key in list(data2d.keys())}, inplace=True)
    df_2d.insert(0, "BDT1", cuts_2d_1)
    df_2d.insert(1, "BDT2", cuts_2d_2)
    df_2d.to_csv(os.path.join(outputpath, '2d_map.csv'), index=False)
    
    print(f"----> INFO: Saved data for 2d map")
    print(f"{15*' '}{os.path.join(outputpath, '2d_map.csv')}")

    df_1d = pd.DataFrame(databdtComb, index=None, columns=list(databdtComb.keys()))
    df_1d.rename(columns={key: cfg.sample_shorthand[key[:-4]]+key[-4:] for key in list(databdtComb.keys())}, inplace=True)
    df_1d.insert(0, "BDTComb", bdtComb_vals)
    df_1d.to_csv(os.path.join(outputpath, '1d_map.csv'), index=False)

    print(f"----> INFO: Saved data for 1d map")
    print(f"{15*' '}{os.path.join(outputpath, '1d_map.csv')}")
    
    ##############################
    ## Plotting - in the future can also perform the fitting here
    ##############################
    
    # Summary
    end = time()
    print(f"\n{30*'-'}")
    print(f"Execution time = {timedelta(seconds=end-start)}")
    print(f"{30*'-'}")
