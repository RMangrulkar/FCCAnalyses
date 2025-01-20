# thrustmag_vs_bbthrustangle.py
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

thrust_angle = '(MC_q1_px*EVT_Thrust_x + MC_q1_py*EVT_Thrust_y + MC_q1_pz*EVT_Thrust_z)/((MC_q1_p)*(EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'

signal_path = os.path.join(inputpath, cfg.samples[0], '*.root')
bb_path = os.path.join(inputpath, cfg.samples[1], '*.root')

thrustmag_nocut = np.array([])
costheta_nocut = np.array([])
thrustmag_cut = np.array([])
costheta_cut = np.array([])

for tree in uproot.iterate(bb_path+':events', expressions=[thrust_angle, 'EVT_Thrust_mag', 'EVT_Thrust_Emin_e']):
    thrustmag_nocut = np.append(thrustmag_nocut, ak.ravel(tree['EVT_Thrust_mag']).to_numpy())
    costheta_nocut = np.append(costheta_nocut, ak.ravel(tree[thrust_angle]).to_numpy())
    thrustmag_cut = np.append(thrustmag_cut, ak.ravel(tree['EVT_Thrust_mag'][tree['EVT_Thrust_Emin_e'] < 20]).to_numpy())
    costheta_cut = np.append(costheta_cut, ak.ravel(tree[thrust_angle][tree['EVT_Thrust_Emin_e'] < 20]).to_numpy())

###### Plotting
fig = plt.figure(figsize=(12, 6))

# Create the main GridSpec
main_gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])

# Create subgrids for the areas where we want no spacing
left_gs = main_gs[0, :2].subgridspec(1, 2, wspace=0.01, hspace=0.01)
right_gs = main_gs[0, 2:4].subgridspec(1, 2, wspace=0.01, hspace=0.01)

# Create the subplots
ax1 = fig.add_subplot(left_gs[0, 0])
ax2 = fig.add_subplot(left_gs[0, 1], sharey=ax1)

ax3 = fig.add_subplot(right_gs[0, 0])
ax4 = fig.add_subplot(right_gs[0, 1], sharey=ax3)

# Plotting commands remain unchanged
ax1.hist2d(x=costheta_nocut, y=thrustmag_nocut, bins=100, range=((-1, 1), (0, 1)), density=True)
ax2.hist2d(x=costheta_nocut, y=thrustmag_nocut, bins=100, range=((-1, 1), (0, 1)),   density=True)

ax3.hist2d(x=costheta_cut, y=thrustmag_cut, bins=100, range=((-1, 1), (0, 1)), density=True)
ax4.hist2d(x=costheta_cut, y=thrustmag_cut, bins=100, range=((-1, 1), (0, 1)),   density=True)

# Axis settings remain unchanged
ax1.set_xlim(-1, -0.8)
ax1.set_ylim(0.6, 1)
ax1.spines['right'].set_visible(False)
ax1.yaxis.tick_left()
ax2.xaxis.tick_bottom()
ax1.tick_params(labelsize=8)
ax1.tick_params(axis='x', which='both', labeltop='off')

ax2.set_xlim(0.8, 1)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax2.xaxis.tick_bottom()
ax2.tick_params(labelsize=8)
ax2.tick_params(axis='x', which='both', labeltop='off')

ax3.set_xlim(-1, -0.8)
ax3.set_ylim(0.6, 1)
ax3.spines['right'].set_visible(False)
ax3.yaxis.tick_left()
ax3.xaxis.tick_bottom()
ax3.tick_params(labelsize=8)
ax3.tick_params(axis='x', which='both', labeltop='off')

ax4.set_xlim(0.8, 1)
ax4.spines['left'].set_visible(False)
ax4.yaxis.tick_right()
ax4.xaxis.tick_bottom()
ax4.tick_params(labelsize=8)
ax4.tick_params(axis='x', which='both', labeltop='off')

fig.suptitle('EVT\_Thrust\_mag as a measure of \"correctness\" of thrust')
fig.supxlabel(r'$\cos{\theta}$ between b and thrust')
fig.supylabel(r'EVT\_Thrust\_mag')
fig.tight_layout()
#plt.savefig('/usera/rrm42/private/fcc-figures/sig_costhetavsthrustmag.png', dpi=300)
plt.savefig('/usera/rrm42/private/fcc-figures/bb_costhetavsthrustmag.png', dpi=300)

#################### Plotting 2
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].hist2d(x=costheta_nocut, y=thrustmag_nocut, bins=75, density=True)
ax[1].hist2d(x=costheta_cut, y=thrustmag_cut, bins=75, density=True)

ax[0].set_xlabel(r'$|\cos{\theta}|$ between thrust and q1')
ax[0].set_ylabel('EVT\_Thrust\_mag')
ax[0].set_title('Without cut')

ax[1].set_xlabel(r'$|\cos{\theta}|$ between thrust and q1')
ax[1].set_ylabel('EVT\_Thrust\_mag')
ax[1].set_title('With cut')

#plt.savefig('/usera/rrm42/private/fcc-figures/sig_costhetavsthrustmag_nolim.png', dpi=300)
plt.savefig('/usera/rrm42/private/fcc-figures/bb_costhetavsthrustmag_nolim.png', dpi=300)
