# thrust_bb_distributions.py
# LEGACY - for intermediate analysis

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from matplotlib.gridspec import GridSpec

import config as cfg
plt.style.use('fcc.mplstyle')

inputpath = '/r01/lhcb/mkenzie/fcc/B2Inv/stage0'

thrust_angle = '(EVT_Thrust_z)/((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'
b_angle = '(MC_q1_pz)/(MC_q1_p)'
#bbar_angle = '(MC_q2_pz)/(MC_q2_p)'

signal_path = os.path.join(inputpath, cfg.samples[0], '*.root')
bb_path = os.path.join(inputpath, cfg.samples[1], '*.root')

thrust_nocut = np.array([])
b_nocut = np.array([])
thrust_cut = np.array([])
b_cut = np.array([])

for tree in uproot.iterate(signal_path+':events', expressions=[thrust_angle, b_angle, 'EVT_Thrust_Emin_e']):
    thrust_nocut = np.append(thrust_nocut, ak.ravel(tree[thrust_angle]).to_numpy())
    b_nocut = np.append(b_nocut, ak.ravel(tree[b_angle]).to_numpy())
    thrust_cut = np.append(thrust_cut, ak.ravel(tree[thrust_angle][tree['EVT_Thrust_Emin_e'] < 20]).to_numpy())
    b_cut = np.append(b_cut, ak.ravel(tree[b_angle][tree['EVT_Thrust_Emin_e'] < 20]).to_numpy())

###### Plotting
fig = plt.figure(figsize=(12, 6))

# Create the main GridSpec
main_gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

# Create subgrids for the areas where we want no spacing
left_gs = main_gs[:2, :2].subgridspec(2, 2, wspace=0.01, hspace=0.01, width_ratios=[1, 1], height_ratios=[1, 1])
right_gs = main_gs[:2, 2:4].subgridspec(2, 2, wspace=0.01, hspace=0.01, width_ratios=[1, 1], height_ratios=[1, 1])

# Create the subplots
ax1 = fig.add_subplot(left_gs[0, 0])
ax2 = fig.add_subplot(left_gs[0, 1], sharey=ax1)
ax5 = fig.add_subplot(left_gs[1, 0], sharex=ax1)
ax6 = fig.add_subplot(left_gs[1, 1], sharex=ax2, sharey=ax5)

ax3 = fig.add_subplot(right_gs[0, 0])
ax4 = fig.add_subplot(right_gs[0, 1], sharey=ax3)
ax7 = fig.add_subplot(right_gs[1, 0], sharex=ax3)
ax8 = fig.add_subplot(right_gs[1, 1], sharex=ax4, sharey=ax7)

# Plotting commands remain unchanged
ax1.hist2d(x=thrust_nocut, y=b_nocut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax2.hist2d(x=thrust_nocut, y=b_nocut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax5.hist2d(x=thrust_nocut, y=b_nocut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax6.hist2d(x=thrust_nocut, y=b_nocut, bins=100, range=((-1, 1), (-1, 1)), density=True)

ax3.hist2d(x=thrust_cut, y=b_cut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax4.hist2d(x=thrust_cut, y=b_cut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax7.hist2d(x=thrust_cut, y=b_cut, bins=100, range=((-1, 1), (-1, 1)), density=True)
ax8.hist2d(x=thrust_cut, y=b_cut, bins=100, range=((-1, 1), (-1, 1)), density=True)

# Axis settings remain unchanged
ax1.set_xlim(-1, -0.75)
ax1.set_ylim(0.75, 1)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.yaxis.tick_left()
ax1.xaxis.tick_top()
ax1.tick_params(labelsize=8)

ax2.set_xlim(0.75, 1)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.yaxis.tick_right()
ax2.xaxis.tick_top()
ax2.tick_params(labelsize=8)

ax5.set_ylim(-1, -0.75)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.yaxis.tick_left()
ax5.xaxis.tick_bottom()
ax5.tick_params(labelsize=8)

ax6.spines['left'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.yaxis.tick_right()
ax6.xaxis.tick_bottom()
ax6.tick_params(labelsize=8)

ax3.set_xlim(-1, -0.75)
ax3.set_ylim(0.75, 1)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.yaxis.tick_left()
ax3.xaxis.tick_top()
ax3.tick_params(labelsize=8)

ax4.set_xlim(0.75, 1)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.yaxis.tick_right()
ax4.xaxis.tick_top()
ax4.tick_params(labelsize=8)

ax7.set_ylim(-1, -0.75)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.yaxis.tick_left()
ax7.xaxis.tick_bottom()
ax7.tick_params(labelsize=8)

ax8.spines['left'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.yaxis.tick_right()
ax8.xaxis.tick_bottom()
ax8.tick_params(labelsize=8)

fig.suptitle('Thrust axis vs b without cut (left) and with cut (right)')
fig.supxlabel(r'$\cos{\theta}$ of thrust axis')
fig.supylabel(r'$\cos{\theta}$ of b quark (MC\_q1)')
fig.tight_layout()

plt.savefig('/usera/rrm42/private/fcc-figures/sig_thrustvsb.png', dpi=300)
#plt.savefig('/usera/rrm42/private/fcc-figures/bb_thrustvsb.png', dpi=300)
plt.close()

############## Plotting 2
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


ax[0].hist2d(x=thrust_nocut, y=b_nocut, bins=50, range=((-1, 1), (-1, 1)), density=True)
ax[1].hist2d(x=thrust_cut, y=b_cut, bins=50, range=((-1, 1), (-1, 1)), density=True)

ax[0].set_xlabel(r'$\cos{\theta}$ of thrust axis')
ax[0].set_ylabel(r'$\cos{\theta}$ of MC\_q1')
ax[0].set_title('Without cut')

ax[1].set_xlabel(r'$\cos{\theta}$ of thrust axis')
ax[1].set_ylabel(r'$\cos{\theta}$ of b quark (MC\_q1)')
ax[1].set_title('With cut')

fig.tight_layout()

plt.savefig('/usera/rrm42/private/fcc-figures/sig_thrustvsb_nolim.png')
#plt.savefig('/usera/rrm42/private/fcc-figures/bb_thrustvsb_nolim.png')
