# mother_averages.py
# LEGACY - Prints a summary of true MC properties of surviving events
import os
import uproot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import awkward as ak
import pandas as pd

from glob import glob
from argparse import ArgumentParser

import config as cfg

parser = ArgumentParser()
parser.add_argument("--cut",       type=str, required=False, default=None, help='Cut expression to pass to uproot')
parser.add_argument("--inputpath", type=str, default='/r01/lhcb/rrm42/fcc/stage1_postBDT', help='Path to files containing cfg.samples')
parser.add_argument("--nchunks",   nargs='*', default=None, help='Number of chunks to run over')
parser.add_argument("--eminonly",  default=False, action="store_const", const=True, help="Option to only use particles from Emin hemisphere")
args = parser.parse_args()

print('Initialising...')
print(f'Using {args.inputpath} with cut {args.cut}')
if args.eminonly:
    print('From reconstructed particles in the Emin hemisphere')
else:
    print('From all reconstructed particles')

if args.nchunks is not None:
    chunks = {sample: int(args.nchunks[i]) for i, sample in enumerate(cfg.samples)}

arr_list = [
    "Rec_true_PDG",
    "Rec_true_M1",
    "Rec_true_M2",
    "Rec_true_M1ofM1",
    "Rec_true_M2ofM1",
    "Rec_true_M1ofM2",
    "Rec_true_M2ofM2"
]
hists = {}

bins_meaning = ['none', 'quark', 'leptons', 'g/gamma', 'light uds', 'K', 'D', 'B', 'had']
bins = [0, 1, 10, 20, 100, 300, 400, 500, 600, 9999]

print(f'Column meanings:')
for i in range(len(bins_meaning)):
    print(f'{bins_meaning[i]}: PDG in [{bins[i]}, {bins[i+1]})')
print(f"{30*'-'}")
for sample in cfg.samples:
    outdict = {x: np.array([]) for x in arr_list}
    hists[sample] = pd.DataFrame(columns=bins_meaning, index=arr_list)

    path = os.path.join(args.inputpath, sample, '*.root')
    files = glob(path)
    n_events = 0

    if args.nchunks is not None:
        files = files[:chunks[sample]]

    for file in files:
        with uproot.open(file) as f:
            n_events += int(f['eventsSelected'])
            if args.eminonly:
                arr = f['events'].arrays(arr_list+["Rec_in_hemisEmin"], cut=args.cut, library='ak')

                for j in arr_list:
                    temp = arr[j][arr['Rec_in_hemisEmin'] == 1]
                    outdict[j] = np.append(outdict[j], np.abs(ak.flatten(temp).to_numpy()))

            else:
                arr = f['events'].arrays(arr_list, cut=args.cut, library = 'ak')

                for j in arr_list:
                    outdict[j] = np.append(outdict[j], np.abs(ak.flatten(arr[j]).to_numpy()))
    
    for j in arr_list:
        hists[sample].loc[j] = np.histogram(outdict[j], bins=bins, density=False)[0]/n_events

    print(f'{sample}:')
    #out_df = hists[sample].replace(0., '-')
    print(hists[sample].replace(0, '-'))
    print(f'\n{30*"-"}\n')
