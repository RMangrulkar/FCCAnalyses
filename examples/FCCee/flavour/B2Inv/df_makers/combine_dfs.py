import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg 


def load_all_pickles_into_dataframe(runmode,folder,sample):
    all_dfs = []

    root_dir = os.path.join(cfg.fccana_opts["outputDir"][runmode], folder, sample)
    print(root_dir)
    print(os.walk(root_dir))

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".pkl"):
                file_path = os.path.join(dirpath, file)
                try:
                    df = pd.read_pickle(file_path)
                    all_dfs.append(df)
                    print(f"Loaded: {file_path} (rows: {len(df)})")
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal combined rows: {len(combined_df)}")

        # Get the root directory name for naming the final pickle and save one level up
        folder_name = os.path.basename(os.path.abspath(root_dir))
        parent_dir = os.path.dirname(os.path.abspath(root_dir))
        output_path = os.path.join(parent_dir, f"{folder_name}_combined.pkl")

        #ensure no columns are RVecs
        for col in combined_df.columns:
            if combined_df[col].dtype == "object":
                combined_df[col] = combined_df[col].apply(lambda x: list(x) if "RVec" in str(type(x)) else x)


        combined_df.to_pickle(output_path)
        print(f"\nSaved combined DataFrame to: {output_path}")

        return combined_df
    else:
        print("No .pkl files found.")
        return pd.DataFrame() 
    

if __name__ == "__main__":
     
    runmode = 'process_with_MC_full_prelim'
    folder= 'baseline_plus_bdtlh_dataframes/bdtlh_9990cut'
    
    for sample in cfg.sample_allocations['hadronic_background']+cfg.sample_allocations['combined_signal']:
        load_all_pickles_into_dataframe(runmode,folder,sample)
   

        
