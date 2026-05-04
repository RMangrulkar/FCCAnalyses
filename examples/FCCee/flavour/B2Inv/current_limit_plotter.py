import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 14
})

# --- Estimated Data ---
# Format: [B0 -> inv value, B0_s -> inv value] (None if no data for that category)
data = {
    'ALEPH': [1.4e-4, 5.6e-4],
    'BaBar': [2.4e-5, None],
    'Belle': [7.8e-5, None],
    'Belle II 50ab^{-1} Y(4S)': [1.5e-6, None],
    'Belle II 5ab^{-1} Y(5S)': [None, 1.1e-5]
}

# Colors matching the plot
c_aleph = '#413f68'  # Dark blue/purple
c_babar = '#448083'  # Teal
c_belle = '#79c07e'  # Light green

fig, ax = plt.subplots(figsize=(7, 6))

# X-axis positions for the two categories
x_b0 = 1
x_bs = 2

# --- Plotting Data Points ---
# ALEPH (Filled Square)
ax.scatter(x_b0, data['ALEPH'][0], marker='s', s=80, color=c_aleph, label='ALEPH', zorder=3)
ax.scatter(x_bs, data['ALEPH'][1], marker='s', s=80, color=c_aleph, zorder=3)

# BaBar (Filled Circle)
ax.scatter(x_b0, data['BaBar'][0], marker='o', s=80, color=c_babar, label='BaBar', zorder=3)

# Belle (Filled Upward Triangle)
ax.scatter(x_b0, data['Belle'][0], marker='^', s=80, color=c_belle, label='Belle', zorder=3)

# Belle II 50ab-1 (Empty Upward Triangle)
ax.scatter(x_b0, data['Belle II 50ab^{-1} Y(4S)'][0], marker='^', s=80, 
           facecolors='none', edgecolors=c_belle, linewidths=1.5, 
           label=r'Belle II $50\mathrm{ab}^{-1}$ $\Upsilon(4S)$', zorder=3)

# Belle II 5ab-1 (Empty Downward Triangle)
ax.scatter(x_bs, data['Belle II 5ab^{-1} Y(5S)'][1], marker='v', s=80, 
           facecolors='none', edgecolors=c_belle, linewidths=1.5, 
           label=r'Belle II $5\mathrm{ab}^{-1}$ $\Upsilon(5S)$', zorder=3)

# --- Formatting Axes ---
ax.set_yscale('log')
ax.set_ylim(5e-7, 5e-2) # Adjust limits to match image
ax.set_xlim(0.5, 2.5)

ax.set_ylabel(r'$\mathcal{B}$ limit at $90\%$ CL')

# X-axis labels
ax.set_xticks([x_b0, x_bs])
ax.set_xticklabels([r'$B^0 \to \mathrm{invisible}$', r'$B^0_s \to \mathrm{invisible}$'], rotation=0)

# Vertical grid line to separate the two categories
ax.axvline(1.5, color='lightgray', linestyle='--', linewidth=1, zorder=1)

# --- Tick Parameters ---
# Y-axis ticks pointing inwards, visible on top/right
ax.tick_params(axis='y', which='major', direction='in', right=True, length=6)
ax.tick_params(axis='y', which='minor', direction='in', right=True, length=3)

# X-axis ticks pointing outwards
ax.tick_params(axis='x', which='major', direction='out', top=False, length=6)

# --- Legend ---
# ncol=2 formats the legend into two columns 
ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.98))

plt.tight_layout()
plt.savefig("current_b2inv_limits.pdf")
plt.savefig("current_b2inv_limits.png")