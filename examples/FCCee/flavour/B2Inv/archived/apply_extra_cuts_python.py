import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import uproot
import glob
import os
import ROOT

import config as cfg


#function to read root data one file at a time and return pandas df
def read_root_data(inpath, folder, varnames, cut,chunk_index=0):
    """Reads event data and TParameters from ROOT files."""

    # Get the list of ROOT files in the specified folder and select one requested by start index
    file = sorted(glob.glob(os.path.join(os.path.abspath(inpath), folder, "chunk_*.root")))[chunk_index]

    # Read event data using uproot
    event_data = uproot.concatenate([f"{file}:events"], expressions=varnames, cut=cut, library="np")

    # Read TParameters from the file
    root_file = uproot.open(file)

    # Fetch the TParameters as integers, with a default of 0 if not present
    events_processed = root_file.get("eventsProcessed", 0).value  # Events processed in tupling
    events_selected = root_file.get("eventsSelected", 0).value    # Events selected by tupling (excludes PVfit cut)

    #return chunk processed as string
    chunk_processed = os.path.basename(file).replace(".root", "")
    
    return pd.DataFrame(event_data), events_processed, events_selected, chunk_processed



tupling_output_path = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/" 
data_folder = "test" #"prelim_cuts_noPV_ntracks_cut" # to change post test
input = os.path.join(os.path.abspath(tupling_output_path), data_folder)
vars = ["Rec_vtx_thrustCosTheta", "Rec_vtx_in_hemisEmin","Rec_vtx_thrustCosTheta_d2PV", "Rec_vtx_in_hemisEmin_d2PV","Rec_vtx_isPV"] #to change to all vars from B2inv yaml
decay_list = ["p8_ee_Zbb_ecm91"]  # to change to full list from yaml

#Add preliminary cut on ability to fit PV with non-MC seeded fitter
cuts = "Rec_PV_ntracks>1"

for decay in decay_list:
    print(f"Decay: {decay}")
    # Loop over each chunk of data for each decay
    for index in range(0, len(glob.glob(os.path.join(os.path.abspath(input), decay, "chunk_*.root"))), 1):
        
        # Read data and TParameters (eventsProcessed, eventsSelected) for each decay sample
        df, events_processed, events_selected, chunk_processed = read_root_data(inpath=input, folder=decay, varnames=vars, cut=cuts, chunk_index=index)

        matching_results = []
        # Loop over events and compare the jagged arrays
        for i in range(len(df)):
            comparison = np.equal(df["Rec_vtx_in_hemisEmin"][i][1:], df["Rec_vtx_in_hemisEmin_d2PV"][i][1:])
            matching_results.append(comparison.all())  # Check if all elements match
    
        # Count the events to remove (where comparison is False)
        count_false = matching_results.count(False)
        total_events = len(matching_results)
        
        # Update eventsProcessed and eventsSelected based on filtering
        events_processed_post_PVfit_cut = total_events
        events_selected_after_both_cuts = total_events - count_false

        print(f"Chunk: {chunk_processed}")
        print(f"Total events processed by tupling: {events_processed}")
        print(f"Total events pre cuts: {events_selected}")
        print(f"Total events post cuts (PV reconstriction and vertex pointing): {events_selected_after_both_cuts}")
        print(f"Fraction removed by two additional cuts: {events_selected_after_both_cuts/events_selected}")

        
        # Store the filtered DataFrame based on the matching condition
        df["vtx_assignment_agreement"] = matching_results
        filtered_df = df[df["vtx_assignment_agreement"] == True]

         # Save the filtered data and tracking parameters (eventsProcessed, eventsSelected) to a new ROOT file
    
        # Check if the folder exists
        output_folder_path = os.path.join(os.path.abspath(tupling_output_path), "prelim_and_PVreco_vtxassignment_cuts")
        
        if not os.path.exists(output_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(output_folder_path)

        if not os.path.exists(os.path.join(os.path.abspath(output_folder_path), decay)):
            # Create the folder if it doesn't exist
            os.makedirs(os.path.join(os.path.abspath(output_folder_path), decay))
            
        output_file = os.path.join(os.path.abspath(output_folder_path), decay, chunk_processed +".root")

        # Open a new ROOT file for writing
        with uproot.recreate(output_file) as f_out:
            # Create a tree with filtered data
            f_out["events"] = filtered_df
        
        # Create TParameters for event counts
        tparams = {
            "eventsProcessed": events_processed,
            "eventsSelectedbyTupling": events_selected,
            "eventsSelected": events_selected_after_both_cuts,
        }
        
        # Write the TParameters as separate objects to the ROOT file 
        # As far as I know Uproot cant do this so switching to pyroot
        rootfile = ROOT.TFile(output_file, 'UPDATE')
        for param_name, param_values in tparams.items():
            param = ROOT.TParameter('int')(param_name, param_values)
            param.Write() 
        
        # Close the file
        rootfile.Close()






















'''

## Now look at whole df where a row == an event
def as_np(inpath,folder, varnames, cut, nchunks):
    if nchunks is not None:
        files = glob.glob(os.path.join(os.path.abspath(inpath), folder, "*.root"))[:nchunks]
        path = [ f"{f}:events" for f in files ]
    else:
        path = os.path.join( os.path.abspath(inpath), folder, "*.root:events" )

    a = uproot.concatenate( path, expressions=varnames, cut=cut, library="np" )

    # Return jagged numpy array
    return a

input = "/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/prelim_cuts_noPV_ntracks_cut"
vars = ["Rec_vtx_thrustCosTheta", "Rec_vtx_in_hemisEmin","Rec_vtx_thrustCosTheta_d2PV", "Rec_vtx_in_hemisEmin_d2PV","Rec_vtx_isPV"]
decay_list = cfg.samples
cuts = "Rec_PV_ntracks>1"
df_dict={}
for decay in decay_list:
    df_dict[decay] = pd.DataFrame(as_np(inpath=input,folder=decay, varname=vars, cut=cuts, nchunks=5))


filtered_dict={}
for decay in decay_list:
    ddd = df_dict[decay]
    arry=[]
    for i in range(len(ddd)):
        comparison = np.equal(ddd["Rec_vtx_in_hemisEmin"][i][1:], ddd["Rec_vtx_in_hemisEmin_d2PV"][i][1:])
        arry.append(comparison.all())
    # Return False if any pair doesn't match
    
    count = arry.count(False)
    ddd["vtx_assignment_agreement"] = arry
    print(decay)
    print(f"Frac to remove {count/len(arry)}")
    print(len(arry))
    print(len(ddd))
    print(len(ddd[ddd["vtx_assignment_agreement"]==True]))
    filtered_dict[decay] = ddd[ddd["vtx_assignment_agreement"]==True]
        



evdict = {}
for dec, files in fdict.items():
    print(dec)
    totProc = 0
    totSele = 0
    for file in files:
        tf = ROOT.TFile( f"{path}/{dec}/{file}")
        evsProc = tf.Get("eventsProcessed").GetVal()
        evsSele = tf.Get("eventsSelected").GetVal()
        totProc += evsProc
        totSele += evsSele
        print( f"  {file:20s} - {evsSele:8d} / {evsProc:8d}" )
    print( f"TOTAL = {totSele:8d} / {totProc:8d}" )
    evdict[dec] = (totSele, totProc)

'''