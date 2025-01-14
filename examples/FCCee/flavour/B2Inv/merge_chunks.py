import ROOT
import os
import sys

configPath = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv'
sys.path.append(os.path.abspath(configPath))
import config as cfg

def merge_root_files(input_folder_path, output_file, tree_name='events', input_files_list=None):
    # Create a new ROOT file to store the combined data
    output_file = ROOT.TFile(output_file, "RECREATE")
    #list input files
    input_files = os.listdir(input_folder_path)
    if input_files_list:
        input_files = input_files_list

    # Create a TChain to combine TTrees from input files 
    chain = ROOT.TChain(tree_name)

    # Initialize counters 
    total_eventsProcessed = 0 
    total_eventsSelected = 0
    total_events=0

    for file_name in input_files:
        input_file_path = os.path.join(input_folder_path, file_name)
        if os.path.isfile(input_file_path): 
            chain.Add(input_file_path)

            # Open the input file to read the TParameters 
            input_file = ROOT.TFile.Open(input_file_path) 
            processed_events_param = input_file.Get("eventsProcessed") 
            selected_events_param = input_file.Get("eventsSelected") 
            # Update event counters 
            if processed_events_param: 
                total_eventsProcessed += processed_events_param.GetVal() 
            if selected_events_param: 
                total_eventsSelected += selected_events_param.GetVal()

    # Clone the structure of the chain into a new TTree 
    output_file.cd() 
    merged_tree = chain.CloneTree(0)

    # Loop over all entries and fill the merged tree 
    for entry in range(chain.GetEntries()): 
        chain.GetEntry(entry) 
        merged_tree.Fill() 
        total_events += 1
        
    # Write the merged TTree to the output file 
    merged_tree.Write() 

    # Store the event counts as TParameter objects 
    eventsProcessed_param = ROOT.TParameter(int)("eventsProcessed", total_eventsProcessed) 
    eventsSelected_param = ROOT.TParameter(int)("eventsSelected", total_eventsSelected) 
    eventsProcessed_param.Write() 
    eventsSelected_param.Write()

    output_file.Close()

    # Print the total events processed and selected 
    print(f"{sample}")
    print(f"Total events processed: {total_eventsProcessed}") 
    print(f"Total events selected: {total_eventsSelected}")
    print(f"Total events in new Ttree:{total_events}")

    if total_eventsSelected !=total_events:
        print('Error: Incorrect number of events in Ttree!!')



'''
### Test on data really dont care about...
data_folder_path = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training_old_and_incorrectBSC/'
list = ['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu','p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu','p8_ee_Zbb_ecm91','p8_ee_Zcc_ecm91','p8_ee_Zss_ecm91', 'p8_ee_Zud_ecm91']
for sample in list:
    output_file = data_folder_path + f'full_data_merged/{sample}_allchunks.root'
    folder_path = data_folder_path+ sample
    input_files = os.listdir(folder_path)
    merge_root_files(folder_path,output_file)
#n.b. for old_andOincorrectBSC added extra variables whilst still processing ud so some ud files have extra variables -> cant compile ud into one file
'''

#### Running on old data as a test
data_folder_path = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training_incorrectBSC/'

for sample in cfg.samples:
    output_file = data_folder_path + f'full_data_merged/{sample}_allchunks.root'
    folder_path = data_folder_path+ sample
    nput_files = os.listdir(folder_path)
    merge_root_files(folder_path,output_file)

