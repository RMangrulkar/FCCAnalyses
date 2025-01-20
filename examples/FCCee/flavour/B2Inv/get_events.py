#script to recover EventsProcessed and EventsSelected for ud files since two died meaning this wasn't calculated

import ROOT
import os
import sys
import re
import numpy as np

configPath = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))
import config as cfg

folder_path = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training/p8_ee_Zud_ecm91'

condor_job_file_path = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/BatchOutputs/2025-01-11_21-21-59/p8_ee_Zud_ecm91'
# find total_events_processed from condor error



def list_files(directory, extension,file_array):
    files = os.listdir(directory)
    for file in files:
        if file.endswith(extension):
            file_array.append(file)

#list all files to go through
error_logs=[]
list_files(condor_job_file_path, '.error',error_logs)

def find_number(file_path, search_string,array):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(f"{search_string} ([\d,]+)", content)
        if match:
            number = match.group(1).replace(',', '') # Remove commas from the number
            #print(f"Number found after '{search_string}': {number}")
            array.append(int(number))
        else:
            print(f"Number not found after '{search_string}'.")

events_selected_per_file=[]
events_processed_per_file=[]
#use to find number
for file in error_logs:
    find_number(os.path.join(condor_job_file_path, file), 'Total events processed: ', events_processed_per_file)
    find_number(os.path.join(condor_job_file_path, file), 'No. result events:      ', events_selected_per_file)
    



total_events_processed = np.sum(events_processed_per_file)
total_events_selected = np.sum(events_selected_per_file)

print(total_events_processed)
print(total_events_selected)


'''
### Now write these to all files for ud: /r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training/p8_ee_Zud_ecm91

outputs_folder = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training/p8_ee_Zud_ecm91'

def write_parameter(root_files,folder, parameter_name, parameter_value):
    for file in root_files:
        full_path = os.path.join(folder, file)
        file = ROOT.TFile.Open(full_path, 'UPDATE')
        if file:
            param = ROOT.TParameter(int)(parameter_name, int(parameter_value)) 
            param.Write() 
            file.Close()
        else:
            print(f"Could not open file: {full_path}")

# Example usage
root_files=[]
root_folder = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/outputs/stage2_training/test'
list_files(root_folder, '.root',root_files)

write_parameter(root_files, root_folder,"eventsProcessed", total_events_processed) #remeber need name in double quotations
write_parameter(root_files, root_folder,"eventsSelected", total_events_selected)
'''