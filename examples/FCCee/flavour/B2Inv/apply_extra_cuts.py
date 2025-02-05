import numpy as np
import glob
import os
import ROOT

import config as cfg


#function to read root data one file at a time
def read_data(inpath, folder, cut, chunk_index=0):

    # Get the list of ROOT files in the specified folder and select one requested by start index
    file_path = sorted(glob.glob(os.path.join(os.path.abspath(inpath), folder, "chunk_*.root")))[chunk_index]
    chunk_processed = os.path.basename(file_path).replace(".root", "")
    
    # read file and load RDataFrame or event data
    file = ROOT.TFile(file_path, 'READ') 

    # Fetch the TParameters as integers
    tparams = {}
    param_names = ['eventsProcessed', 'eventsSelected']

    for param_name in param_names:
        param = file.Get(param_name)
        tparams[param_name] = param.GetVal()

     # Check if the TTree named "events" exists
    if not file.GetListOfKeys().Contains("events"):
        print(f"No events in: {file.GetName()}")
        return None, tparams, chunk_processed
    
    df = ROOT.RDataFrame("events", file) 
    df = df.Filter(cut)
    print(f"Number of entries after PV cut: {df.Count().GetValue()}")

    return df, tparams,chunk_processed

tupling_output_path = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/" 
data_folder = "prelim_cuts_noPV_ntracks_cut" 
input = os.path.join(os.path.abspath(tupling_output_path), data_folder)
decay_list =  ["p8_ee_Zmumu_ecm91","p8_ee_Zee_ecm91",] #cfg.samples
cuts = "Rec_PV_ntracks>1"


for decay in decay_list:
    print(f"Decay: {decay}")
    # Loop over each chunk of data for each decay
    for index in range(0, len(glob.glob(os.path.join(os.path.abspath(input), decay, "chunk_*.root"))), 1):

        # Read data and TParameters (eventsProcessed, eventsSelected) for each decay sample
        df, tparams, chunk= read_data(inpath=input, folder=decay, cut=cuts, chunk_index=index)
       
        if df:

            # Define a new column that checks the hemisphere assignment between vtx_p and PV-SV vector matches
            df = df.Define("matching_vtx_assignment", "Rec_vtx_in_hemisEmin==Rec_vtx_in_hemisEmin_d2PV")
        
            # Define a new column to check the condition [0, any number of 1s]
            df = df.Define("HasFormZeroOnes", "matching_vtx_assignment.size() > 0 && matching_vtx_assignment.at(0) == 0 && Sum(matching_vtx_assignment) == matching_vtx_assignment.size() - 1")
                    
            # Filter the DataFrame based on the matching condition
            filtered_df = df.Filter("HasFormZeroOnes==1")


            # Count the total events after the cuts
            events_selected_after_both_cuts = filtered_df.Count().GetValue()

            # Create TParameters for event counts
            filtered_tparams = {
                "eventsProcessed":  tparams["eventsProcessed"],
                "eventsSelectedbyTupling": tparams["eventsSelected"],
                "eventsSelected": events_selected_after_both_cuts,
            }
            
            # Print the results
            print(f'Chunk: {chunk}')
            print(f'Total events processed by tupling: {tparams["eventsProcessed"]}')
            print(f'Total events selected by tupling: {tparams["eventsSelected"]}')
            print(f'Events selected after two cuts: {events_selected_after_both_cuts}')
            print(f'Fraction left after two cuts: {events_selected_after_both_cuts/tparams["eventsSelected"]}')
        

            # Save the filtered data to a new ROOT file
            # Check if the folder exists
            output_folder_path = os.path.join(os.path.abspath(tupling_output_path), "prelim_and_PVreco_vtxassignment_cuts")
            
            if not os.path.exists(output_folder_path):
                # Create the folder if it doesn't exist
                os.makedirs(output_folder_path)

            if not os.path.exists(os.path.join(os.path.abspath(output_folder_path), decay)):
                # Create the folder if it doesn't exist
                os.makedirs(os.path.join(os.path.abspath(output_folder_path), decay))
                
            output_file_path = os.path.join(os.path.abspath(output_folder_path), decay, chunk +".root")

            # Create a new ROOT file
            output_file = ROOT.TFile(output_file_path, 'RECREATE')

            # Save the RDataFrame as a TTree in the ROOT file
            filtered_df.Snapshot("events", output_file_path)  

            # Open the same ROOT file in UPDATE mode to write the TParameters
            output_file = ROOT.TFile(output_file_path, 'UPDATE')
            
            # Write the TParameters as separate objects to the ROOT file
            for param_name, param_values in filtered_tparams.items():
                param = ROOT.TParameter('int')(param_name, param_values)
                param.Write() 

            output_file.Close()

        else:
            # Save the filtered data to a new ROOT file
            # Check if the folder exists
            output_folder_path = os.path.join(os.path.abspath(tupling_output_path), "prelim_and_PVreco_vtxassignment_cuts")
            
            if not os.path.exists(output_folder_path):
                # Create the folder if it doesn't exist
                os.makedirs(output_folder_path)

            if not os.path.exists(os.path.join(os.path.abspath(output_folder_path), decay)):
                # Create the folder if it doesn't exist
                os.makedirs(os.path.join(os.path.abspath(output_folder_path), decay))
                
            output_file_path = os.path.join(os.path.abspath(output_folder_path), decay, chunk +".root")

            # Create a new ROOT file
            output_file = ROOT.TFile(output_file_path, 'RECREATE')
            
            # Write the TParameters as separate objects to the ROOT file
            for param_name, param_values in tparams.items():
                param = ROOT.TParameter('int')(param_name, param_values)
                param.Write() 

            output_file.Close()






