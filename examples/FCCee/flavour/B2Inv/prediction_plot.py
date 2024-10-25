import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True

from efficiency_finder import get_efficiencies, get_sample_expectations
from argparse import ArgumentParser

import config as cfg

parser = ArgumentParser(description='Plots the prediction for EVT_MVA2')
parser.add_argument('--inputtype',  choices=['stage2', 'custom'], default='stage2', help='Sample input type, by default uses stage2 location in config.py')
parser.add_argument('--custompath', type=str,   help='Path used if INPUTTYPE is custom')
parser.add_argument('--signal-bf',  type=float, default=1e-5,  help='Signal branching fraction, default is 10^-5')
parser.add_argument('--bdt1cut',    type=float, default=0.99, help='BDT1 cut value to use, default is 0.99')
parser.add_argument('--bdt2lower',  type=float, default=0.95, help='Lower bound for the BDT2, default is 0.95')
parser.add_argument('--bdt2upper',  type=float, default=1,    help='Upper bound for the BDT2, default is 1')
parser.add_argument('--nbins',      type=int,   default=5,    help='Number of bins to use, default is 5')
parser.add_argument('--save',       type=str,   default=None, help='Path (directory, not file) to save the plot. If None uses the parent directory of the input')
parser.add_argument('--interactive', action='store_true', default=False, help='Show the plot interactively, default is False')
parser.add_argument('--no-save',     action='store_true', default=False, help='Skip saving the plots and only show')
args = parser.parse_args()

##############################
## Error handling
##############################
print(f"{30*'-'}")
print(f"BDT2 PREDICTION PLOTTER")
print(f"{'-'*30}\n")
if args.inputtype == 'custom' and args.custompath is None:
    raise ValueError("----> ERROR: Custom path is required if input type is custom")
elif args.inputtype == 'custom' and not os.path.exists(args.custompath):
    raise ValueError(f"----> ERROR: Custom path does not exist\n{15*' '}{args.custompath}")
elif args.inputtype == 'custom':
    inputpath = os.path.abspath(args.custompath)
elif args.inputtype == 'stage2':
    inputpath = cfg.fccana_opts['outputDir']['stage2']

print(f"----> INFO: Using input path")
print(f"{15*' '}{inputpath}")

if args.bdt2lower >= args.bdt2upper:
    raise ValueError("----> ERROR: Given lower and upper bounds for BDT2 are invalid")
if args.no_save:
    print(f"---> INFO: --no-save flag set, skipping saving")
else:
    if args.save is not None:
        if not os.path.exists(os.path.dirname(os.path.abspath(args.save))):
            outputpath = cfg.poststage2_opts['outputPath']
            print(f"----> WARNING: Specified save path does not exist, defaulting to")
            print(f"{' '*15}{outputpath}")
        else:
            outputpath = os.path.dirname(os.path.abspath(args.save))
            print(f"----> INFO: Saving to")
            print(f"{15*' '}{outputpath}")
    else:
        outputpath  = cfg.poststage2_opts['outputPath']
        print(f"----> INFO: No save path specified, defaulting to")
        print(f"{' '*15}{outputpath}")
    
##############################
## Setup
##############################
bins = np.linspace(args.bdt2lower, args.bdt2upper, args.nbins+1)  # For 5 bins need 6 edges
bin_centres = (bins[1:] + bins[:-1]) / 2

cut_expr = []
for i in range(len(bins)):
    if (i+1) == len(bins):
        break
    elif (i+2) == len(bins):
        cut_expr.append(f'(EVT_MVA1 >= {args.bdt1cut}) & (EVT_MVA2 >= {bins[i]}) & (EVT_MVA2 <= {bins[i+1]})')
    else:
        cut_expr.append(f'(EVT_MVA1 >= {args.bdt1cut}) & (EVT_MVA2 >= {bins[i]}) & (EVT_MVA2 < {bins[i+1]})')

##############################
## Calculations
##############################
effs = get_efficiencies('custom', custompath=inputpath, cut=cut_expr, raw=True, verbose=False)
nums = get_sample_expectations(effs, args.signal_bf, verbose=False, cut=cut_expr)

tot = np.zeros(len(cut_expr))
var = np.zeros(len(cut_expr))

for sample in cfg.samples:
    if sample in cfg.sample_allocations['background']:
        tot = tot + nums[sample+'_num']
        var = var + nums[sample+'_err']**2

tot_err = np.sqrt(var)
bottom = np.zeros(len(cut_expr))

##############################
## Plotting
##############################
reds = mpl.colormaps['Reds_r']
cols = reds(np.linspace(0, 1, len(cfg.sample_allocations['background'])+4)[2:-2])
fig, ax = plt.subplots(figsize=(10, 8))

# Background
for i, sample in enumerate(cfg.sample_allocations['background']):
    ax.bar(bin_centres, nums[sample+'_num'], width=np.diff(bins), align='center', color=cols[i], bottom=bottom, label=cfg.titles[sample])
    bottom = bottom + nums[sample+'_num']

# Second loop because we want the errorbars on top
errorbar_count = np.zeros(len(cut_expr))
for i, sample in enumerate(cfg.sample_allocations['background']):
    errorbar_count = errorbar_count + nums[sample+'_num']
    ax.errorbar(bin_centres, errorbar_count, xerr=0.5*np.diff(bins), yerr=nums[sample+'_err'], fmt='none', ecolor='black', elinewidth=0.8)

# Signal
for sample in cfg.sample_allocations['signal']:
    ax.bar(bin_centres, nums[sample+'_num'], width=np.diff(bins), align='center', bottom=tot, 
           edgecolor='blue', facecolor='none', hatch='////', linewidth=1.5, label=cfg.titles[sample])

# Total
ax.errorbar(bin_centres, tot, xerr=0.5*np.diff(bins), yerr=tot_err, fmt='None', ecolor='black', elinewidth=2)
ax.bar(bin_centres, 2*tot_err, width=np.diff(bins), align='center', color='gray', alpha=0.5, bottom=tot-tot_err, label='Total background')

ax.set_xlabel('BDT2 Score', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
pred1_str = '\n'.join((
    rf'$\mathcal{{B}}(B_s\to\nu\bar{{\nu}}) = {args.signal_bf:.1e}$', 
    rf'BDT1 $>= {args.bdt1cut}$',
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.27, 0.97, pred1_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
ax.legend(loc='upper left', fontsize=14)
fig.tight_layout()

if not args.no_save:
    print(f"----> INFO: Plot with individual backgrounds saved to")
    print(f"{15*' '}{os.path.join(outputpath, 'prediction_all_separate.pdf')}")
    plt.savefig(os.path.join(outputpath, 'prediction_all_separate.pdf'), dpi=600)
    
# Plot with just the total background and errorbars on signal+background
S = np.zeros(len(bin_centres))
S_err = np.zeros(len(bin_centres))

for sample in cfg.sample_allocations['signal']:
    S += nums[sample+'_num']
    S_err += nums[sample+'_err']**2

S_err = np.sqrt(S_err)

SplusB = tot + S
SplusB_err = np.sqrt(tot_err**2 + S_err**2 + SplusB)

# Separation between B and S+B in the last bin
sep_last_bin = (SplusB[-1] - tot[-1])/SplusB_err[-1]

fig2, ax2 = plt.subplots(figsize=(10, 8))
# Background, B
#ax2.bar(bin_centres, 2*tot_err, width=np.diff(bins), align='center', color='red', alpha=0.4, bottom=tot-tot_err)
B_bar_opts = {'align': 'center', 'alpha': 1, 'color': cols[-1], 'edgecolor': 'red', 'linewidth': 0.7}
B_err_opts = {'ecolor': 'red', 'elinewidth': 1.25}
ax2.bar(bin_centres, tot, width=np.diff(bins), xerr=0.5*np.diff(bins), yerr=tot_err, label=r'$B$', **B_bar_opts, error_kw=B_err_opts)
#ax2.errorbar(bin_centres, tot, xerr=0.5*np.diff(bins), yerr=tot_err, fmt='+', color='red', linewidth=1.25, label=r'$B$')

# Signal, S
S_bar_opts = {'align': 'center', 'alpha': 1, 'facecolor': 'none', 'hatch': '////', 'edgecolor': 'blue', 'linewidth': 0.7}
S_err_opts = {'ecolor': 'blue', 'elinewidth': 1.25}
ax2.bar(bin_centres, S, width=np.diff(bins), bottom=tot, xerr=0.5*np.diff(bins), yerr=S_err, label=r'$S$', **S_bar_opts, error_kw=S_err_opts)
#ax2.errorbar(bin_centres, S+tot, xerr=0.5*np.diff(bins), yerr=S_err, fmt='x', color='blue', linewidth=1.25, label=r'$S$')

# S+B
ax2.errorbar(bin_centres, SplusB, xerr=0.5*np.diff(bins), yerr=SplusB_err, fmt='o', color='black', linewidth=1.5, capsize=5, label=r'$S+B$')

ax2.set_xlabel('BDT2 Score', fontsize=14)
ax2.set_ylabel('Counts', fontsize=14)

# text box
SplusB_str = '\n'.join((
    rf'$\mathcal{{B}}(B_s\to\nu\bar{{\nu}}) = {args.signal_bf:.1e}$', 
    rf'BDT1 $>= {args.bdt1cut}$', 
    rf'${sep_last_bin:.1f}\sigma$ separation in last bin'
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax2.text(0.02, 0.83, SplusB_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
ax2.legend(loc='upper left', fontsize=14)
fig2.tight_layout()

if not args.no_save:
    print(f"----> INFO: Plot with total signal and background saved to")
    print(f"{15*' '}{os.path.join(outputpath, 'prediction_SplusB.pdf')}")
    plt.savefig(os.path.join(outputpath, 'prediction_SplusB.pdf'), dpi=600)

if args.interactive:
    plt.show()
