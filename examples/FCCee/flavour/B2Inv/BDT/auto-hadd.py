# auto-hadd.py
# UNTESTED -- CURRENTLY LEADS TO MEMORY LEAK
# Probably best to instead implement this logic into a shell script
# Takes stage1 or stage2 files and merges to a smaller number of files
# Assumes POSIX system for glob syntax
import os
import sys
sys.path.append('/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/')
import subprocess

from shutil import which
from argparse import ArgumentParser
from glob import glob
from datetime import timedelta
from time import time

import config as cfg

# Arguments
parser = ArgumentParser(description='Merge stage1 or stage2 files')
parser.add_argument('--input-source', required=True, type=str, help='Source of input files, must be a valid key of config.fccana_opts["outputDir"]')
parser.add_argument('--samples',      required=True, nargs='+', type=str, help='Samples whose files are to be merged, pass `all` to apply to all samples or `background-only` to apply to all backgrounds')
parser.add_argument('--n-per-chunk',  required=False, default=250, type=int, help='Reduction factor. The number of files for each hadd command, default is 250')
args = parser.parse_args()

##############################
## INITIALISATION
##############################
start = time()
print(f"\n{30*'-'}")
print(f"MERGE FILES USING HADD")
print(f"{30*'-'}\n")

# If ROOT is not in PATH, source key4hep env
if which('hadd') is None:
    print(f"----> INFO: `hadd` not found in PATH. Sourcing key4hep")
    subprocess.run("source /cvmfs/sw.hsf.org/key4hep/setup.sh", check=True, shell=True)

# Get samples
if args.samples[0] == 'all':
    samples = cfg.samples
elif args.samples[0] == 'background-only':
    samples = cfg.sample_allocations['background']
elif not set(args.samples).issubset(set(cfg.samples)):
    raise ValueError(f'''----> ERROR: Invalid sample name(s) passed, must be in:
                     {15*' '}{cfg.samples}
                     ''')
else:
    samples = args.samples

print(f"Initialising...")
print(f"----> INFO: Merging files for samples:")
print(f"{15*' '}{samples}")

try:
    location = cfg.fccana_opts['outputDir'][args.input_source]
    print(f"----> INFO: Merging files found in:")
    print(f"{15*' '}{location}")
except KeyError:
    raise KeyError(f'''----> ERROR: Invalid input source: {args.input_source}
                   {15*' '}Must be one of {cfg.fccana_opts['outputDir'].keys()}
                   ''')

##############################
## MAIN BODY
##############################
print(f"\n{30*'-'}\n")
for sample in samples:
    sample_dir = os.path.abspath(os.path.join(location, sample))
    n_files = len(glob(os.path.join(sample_dir, '*.root')))
    if n_files == 0:
        print(f"----> WARNING: No ROOT files found at path, skipping...")
        print(f"{15*' '}{sample_dir}\n")
        continue

    elif n_files <= args.n_per_chunk:
        print(f"----> WARNING: File count <={args.n_per_chunk} for sample, skipping...")
        print(f"{15*' '}{sample_dir}\n")
        continue
    print(f"----> INFO: Found {n_files} files for sample:")
    print(f"{15*' '}{sample_dir}")

    # Create output directory
    temp_dir = os.path.abspath(os.path.join(location, sample+'_temp'))
    subprocess.run(f"mkdir {temp_dir}", check=True, shell=True)

    # Split files into chunks
    n_chunks = (n_files // args.n_per_chunk) + 1
    limits = [-1]
    count = -1  # Need to start at -1 because indexing starts at 0

    while count < n_files:
        count += args.n_per_chunk

        # Last chunk contains the remainder
        if count >= n_files:
            count = n_files - 1

        limits.append(count)

    for i in range(len(count) - 1):
        input_files = os.path.abspath(os.path.join(sample_dir, f"chunk_{{{count[i]+1}..{count[i+1]}}}.root"))
        output_file_name = os.path.abspath(os.path.join(temp_dir, f"chunk_{i}.root"))
        subprocess.run(f"hadd -v 1 -k -fk {output_file_name} {input_files}", check=True, shell=True)

    subprocess.run(["mv", sample_dir, os.path.abspath(os.path.join(location, sample+'_nohadd'))], check=True)
    subprocess.run(["mv", temp_dir, sample_dir], check=True)

    print(f"----> INFO: File merging complete, un-merged files moved to:")
    print(f"{15*' '}{os.path.join(location, sample+'_nohadd')}\n")
