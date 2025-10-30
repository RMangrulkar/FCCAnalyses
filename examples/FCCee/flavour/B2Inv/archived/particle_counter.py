# particle_counter.py
# LEGACY - for reference (plotting styles)

########################################################
# Plots histograms of particle num per event using 'datafromrecp' files
# Produces a plot of: e, mu, tau, kaon, pion count
# As it uses MC PIDs of reconstructed particles, only those are plotted
# For e.g. no K0, only K0L, no pi0, only pi+/-
########################################################
import os
import uproot
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import config as cfg

plt.style.use('fcc.mplstyle')

def particle_counter(file, Ecut, cosexpr, cut):
    min_k = np.array([])
    min_p = np.array([])
    min_e = np.array([])
    min_m = np.array([])
    min_t = np.array([])
    max_k = np.array([])
    max_p = np.array([])
    max_e = np.array([])
    max_m = np.array([])
    max_t = np.array([])
    min_lep = np.array([])
    max_lep = np.array([])
    
    if cut:

        for datamin in uproot.iterate(file+':events', expressions=['pid'], aliases={'pid':'abs(MC_PDG)'}, cut=f"({Ecut}) & ({cosexpr} > 0)"):
            
            # Remove empty events i.e. [] from data
            nonempty = datamin['pid'][ak.num(datamin['pid']) > 0]

            # Particle count in Emin hemisphere with Emin cut
            min_k = np.append( min_k , ak.count(nonempty[(nonempty == 321) | (nonempty == 130)], axis=-1).to_numpy() )
            min_p = np.append( min_p , ak.count(nonempty[(nonempty == 111) | (nonempty == 211)], axis=-1).to_numpy() )

            min_e = np.append( min_e , ak.count(nonempty[nonempty  == 11], axis=-1).to_numpy() )
            min_m = np.append( min_m , ak.count(nonempty[nonempty  == 13], axis=-1).to_numpy() )
            min_t = np.append( min_t , ak.count(nonempty[nonempty  == 15], axis=-1).to_numpy() )
        
            min_lep = np.append( min_lep , min_e + min_m + min_t )

        for datamax in uproot.iterate(file+':events', expressions=['pid'], aliases={'pid':'abs(MC_PDG)'}, cut=f"({Ecut}) & ({cosexpr} < 0)"):
            
            # Remove empty events i.e. [] from data
            nonempty = datamax['pid'][ak.num(datamax['pid']) > 0]

            # Particle count in Emin hemisphere with Emin cut
            max_k = np.append( max_k , ak.count(nonempty[(nonempty == 321) | (nonempty == 130)], axis=-1).to_numpy() )
            max_p = np.append( max_p , ak.count(nonempty[(nonempty == 111) | (nonempty == 211)], axis=-1).to_numpy() )

            max_e = np.append( max_e , ak.count(nonempty[nonempty  == 11], axis=-1).to_numpy() )
            max_m = np.append( max_m , ak.count(nonempty[nonempty  == 13], axis=-1).to_numpy() )
            max_t = np.append( max_t , ak.count(nonempty[nonempty  == 15], axis=-1).to_numpy() )
        
            max_lep = np.append( max_lep , max_e + max_m + max_t )

    else:
        for datamin in uproot.iterate(file+':events', expressions=['pid'], aliases={'pid':'abs(MC_PDG)'}, cut=f"{cosexpr} > 0"):
            
            # Remove empty events i.e. [] from data
            nonempty = datamin['pid'][ak.num(datamin['pid']) > 0]

            # Particle count in Emin hemisphere with Emin cut
            min_k = np.append( min_k , ak.count(nonempty[(nonempty == 321) | (nonempty == 130)], axis=-1).to_numpy() )
            min_p = np.append( min_p , ak.count(nonempty[(nonempty == 111) | (nonempty == 211)], axis=-1).to_numpy() )

            min_e = np.append( min_e , ak.count(nonempty[nonempty  == 11], axis=-1).to_numpy() )
            min_m = np.append( min_m , ak.count(nonempty[nonempty  == 13], axis=-1).to_numpy() )
            min_t = np.append( min_t , ak.count(nonempty[nonempty  == 15], axis=-1).to_numpy() )
        
            min_lep = np.append( min_lep , min_e + min_m + min_t )

        for datamax in uproot.iterate(file+':events', expressions=['pid'], aliases={'pid':'abs(MC_PDG)'}, cut=f"{cosexpr} < 0"):
            
            # Remove empty events i.e. [] from data
            nonempty = datamax['pid'][ak.num(datamax['pid']) > 0]

            # Particle count in Emin hemisphere with Emin cut
            max_k = np.append( max_k , ak.count(nonempty[(nonempty == 321) | (nonempty == 130)], axis=-1).to_numpy() )
            max_p = np.append( max_p , ak.count(nonempty[(nonempty == 111) | (nonempty == 211)], axis=-1).to_numpy() )

            max_e = np.append( max_e , ak.count(nonempty[nonempty  == 11], axis=-1).to_numpy() )
            max_m = np.append( max_m , ak.count(nonempty[nonempty  == 13], axis=-1).to_numpy() )
            max_t = np.append( max_t , ak.count(nonempty[nonempty  == 15], axis=-1).to_numpy() )
        
            max_lep = np.append( max_lep , max_e + max_m + max_t )

    return {'min_k': min_k, 
            'max_k': max_k, 
            'min_p': min_p, 
            'max_p': max_p, 
            'min_e': min_e, 
            'max_e': max_e, 
            'min_m': min_m, 
            'max_m': max_m, 
            'min_t': min_t, 
            'max_t': max_t, 
            'min_lep': min_lep, 
            'max_lep': max_lep}

# Signal arrays
#filepath = '/r01/lhcb/mkenzie/fcc/B2Inv/stage0/'
#cosexpr = '(MC_px*EVT_Thrust_x + MC_py*EVT_Thrust_y + MC_pz*EVT_Thrust_z)/(MC_p*(EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2)**0.5)'

filepath = '/r01/lhcb/rrm42/fcc/post_stage0/'
cosexpr = 'Rec_cosrel2thrust'
Ecut = 'Thrust_Emin < 20'

sigfile = os.path.join(filepath, 'Bs2NuNu_fromrecp.root')
sigdict_nocut = particle_counter(sigfile, Ecut, cosexpr, False)
print("Signal files without cut done")
sigdict_Emcut = particle_counter(sigfile, Ecut, cosexpr, True)
print("Signal files with cut done")

bbfile = os.path.join(filepath, 'bb_fromrecp.root')
bbdict_nocut = particle_counter(bbfile, Ecut, cosexpr, False)
print("bb files without cut done")
bbdict_Emcut = particle_counter(bbfile, Ecut, cosexpr, True)
print("bb files with cut done")

ccfile = os.path.join(filepath, 'cc_fromrecp.root')
ccdict_nocut = particle_counter(ccfile, Ecut, cosexpr, False)
print("cc files without cut done")
ccdict_Emcut = particle_counter(ccfile, Ecut, cosexpr, True)
print("cc files with cut done")

ssfile = os.path.join(filepath, 'ss_fromrecp.root')
ssdict_nocut = particle_counter(ssfile, Ecut, cosexpr, False)
print("ss files without cut done")
ssdict_Emcut = particle_counter(ssfile, Ecut, cosexpr, True)
print("ss files with cut done")

udfile = os.path.join(filepath, 'ud_fromrecp.root')
uddict_nocut = particle_counter(udfile, Ecut, cosexpr, False)
print("ud files without cut done")
uddict_Emcut = particle_counter(udfile, Ecut, cosexpr, True)
print("ud files with cut done")

###################### Plotting #############################

titledict = {'k': r"Number of Rec Kaons ($K^0_L$ or $K^\pm$)",
             'p': r"Number of Rec Pions ($\pi^\pm$)",
             'e': r"Number of Rec $e^\pm$",
             'm': r"Number of Rec $\mu^\pm$",
             #'t': r"Number of Rec $\tau^\pm$",
             'lep': r"Number of Rec leptons ($e/\mu$)"}

#min_histopts = {
#        'histtype': 'step',
#        'lw': 2,
#        'color': 'blue'}
#
#max_histopts = {
#        'histtype': 'step',
#        'lw': 1.5,
#        'color': 'red'}
#

bkmin_histopts = {
        'histtype': 'step',
        'lw': 2,
        'color': 'orange'}

bkmax_histopts = {
        'histtype': 'step',
        'lw': 2,
        'ls': '--',
        'color': 'green'}

min_histopts = {
        'histtype': 'step',
        'lw': 2,
        'color': 'red'}

max_histopts = {
        'histtype': 'step',
        'lw': 2,
        'ls': '--',
        'color': 'cornflowerblue'}

for i in titledict:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    _, bins, _ = ax1.hist(
            x = sigdict_nocut['min_'+i],
            bins=np.arange(min(sigdict_nocut['min_'+i]), max(sigdict_nocut['min_'+i]) + 1, 1),
            density = True,
            label = r"$B_s^0 \to \nu\bar{\nu}$: Emin",
            **min_histopts)

    ax1.hist(
            x = sigdict_nocut['max_'+i],
            bins=bins,
            density = True,
            label = r"$B_s^0 \to \nu\bar{\nu}$: Emax",
            **max_histopts)

    ax1.set_ylabel('Density')
    ax1.set_xlabel('Count')
    ax1.set_title('Without cut')
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='best')
    
    ax2.hist(
            x = sigdict_Emcut['min_'+i],
            bins=bins,
            density = True,
            label = r"$B_s^0 \to \nu\bar{\nu}$: Emin",
            **min_histopts)

    ax2.hist(
            x = sigdict_nocut['max_'+i],
            bins=bins,
            density = True,
            label = r"$B_s^0 \to \nu\bar{\nu}$: Emax",
            **max_histopts)

    ax2.set_ylabel('Density')
    ax2.set_xlabel('Count')
    ax2.set_title('With Emin cut')
    ax2.tick_params(labelsize=10)
    ax2.legend(loc='best')

    fig.suptitle(titledict[i])
    fig.tight_layout()
    plt.savefig('/r01/lhcb/rrm42/fcc/'+i+'signal.pdf', dpi=300)
    print(f"/r01/lhcb/rrm42/fcc/{i}signal.pdf plot saved")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    _, bins, _ = ax1.hist(
            x = bbdict_nocut['min_'+i],
            bins=np.arange(min(bbdict_nocut['min_'+i]), max(bbdict_nocut['min_'+i]) + 1, 1),
            density = True,
            label = r"$Z \to b\bar{b}$: Emin",
            **bkmin_histopts)

    ax1.hist(
            x = bbdict_nocut['max_'+i],
            bins=bins,
            density = True,
            label = r"$Z \to b\bar{b}$: Emax",
            **bkmax_histopts)

    ax1.set_ylabel('Density')
    ax1.set_xlabel('Count')
    ax1.set_title('Without cut')
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='best')
    
    ax2.hist(
            x = bbdict_Emcut['min_'+i],
            bins=bins,
            density = True,
            label = r"$Z \to b\bar{b}$: Emin",
            **bkmin_histopts)

    ax2.hist(
            x = bbdict_nocut['max_'+i],
            bins=bins,
            density = True,
            label = r"$Z \to b\bar{b}$: Emax",
            **bkmax_histopts)

    ax2.set_ylabel('Density')
    ax2.set_xlabel('Count')
    ax2.set_title('With Emin cut')
    ax2.tick_params(labelsize=10)
    ax2.legend(loc='best')

    fig.suptitle(titledict[i])
    fig.tight_layout()
    plt.savefig('/r01/lhcb/rrm42/fcc/'+i+'bb.pdf', dpi=300)
    print(f"/r01/lhcb/rrm42/fcc/{i}bb.pdf plot saved")

    bbweight = 0.1512
    ccweight = 0.1203
    ssweight = 0.1560
    udweight = 0.1560 + 0.1160
    
    totmin_nocut = np.concatenate([bbdict_nocut['min_'+i], 
                                  ccdict_nocut['min_'+i],
                                  ssdict_nocut['min_'+i],
                                  uddict_nocut['min_'+i]])
    totmin_nocutweight = np.concatenate([bbweight*np.ones_like(bbdict_nocut['min_'+i]),
                                        ccweight*np.ones_like(ccdict_nocut['min_'+i]),
                                        ssweight*np.ones_like(ssdict_nocut['min_'+i]),
                                        udweight*np.ones_like(uddict_nocut['min_'+i])])
    
    totmax_nocut = np.concatenate([bbdict_nocut['max_'+i], 
                                  ccdict_nocut['max_'+i],
                                  ssdict_nocut['max_'+i],
                                  uddict_nocut['max_'+i]])
    totmax_nocutweight = np.concatenate([bbweight*np.ones_like(bbdict_nocut['max_'+i]),
                                        ccweight*np.ones_like(ccdict_nocut['max_'+i]),
                                        ssweight*np.ones_like(ssdict_nocut['max_'+i]),
                                        udweight*np.ones_like(uddict_nocut['max_'+i])])
    
    totmin_Emcut = np.concatenate([bbdict_Emcut['min_'+i], 
                                  ccdict_Emcut['min_'+i],
                                  ssdict_Emcut['min_'+i],
                                  uddict_Emcut['min_'+i]])
    totmin_Emcutweight = np.concatenate([bbweight*np.ones_like(bbdict_Emcut['min_'+i]),
                                        ccweight*np.ones_like(ccdict_Emcut['min_'+i]),
                                        ssweight*np.ones_like(ssdict_Emcut['min_'+i]),
                                        udweight*np.ones_like(uddict_Emcut['min_'+i])])
    
    totmax_Emcut = np.concatenate([bbdict_Emcut['max_'+i], 
                                  ccdict_Emcut['max_'+i],
                                  ssdict_Emcut['max_'+i],
                                  uddict_Emcut['max_'+i]])
    totmax_Emcutweight = np.concatenate([bbweight*np.ones_like(bbdict_Emcut['max_'+i]),
                                        ccweight*np.ones_like(ccdict_Emcut['max_'+i]),
                                        ssweight*np.ones_like(ssdict_Emcut['max_'+i]),
                                        udweight*np.ones_like(uddict_Emcut['max_'+i])])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    _, bins, _ = ax1.hist(
            x = totmin_nocut,
            bins=np.arange(min(totmin_nocut), max(totmin_nocut), 1),
            density = True,
            weights = totmin_nocutweight,
            label = r"Total background: Emin",
            **bkmin_histopts)

    ax1.hist(
            x = totmax_nocut,
            bins=bins,
            density = True,
            weights = totmax_nocutweight,
            label = r"Total background: Emax",
            **bkmax_histopts)

    ax1.set_ylabel('Density')
    ax1.set_xlabel('Count')
    ax1.set_title('Without cut')
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='best')
    
    ax2.hist(
            x = totmin_Emcut,
            bins=bins,
            density = True,
            weights = totmin_Emcutweight,
            label = r"Total background: Emin",
            **bkmin_histopts)

    ax2.hist(
            x = totmax_Emcut,
            bins=bins,
            density = True,
            weights = totmax_Emcutweight,
            label = r"Total background: Emax",
            **bkmax_histopts)

    ax2.set_ylabel('Density')
    ax2.set_xlabel('Count')
    ax2.set_title('With Emin cut')
    ax2.tick_params(labelsize=10)
    ax2.legend(loc='best')

    fig.suptitle(titledict[i])
    fig.tight_layout()
    plt.savefig('/r01/lhcb/rrm42/fcc/'+i+'totbkg.pdf', dpi=300)
    print(f"/r01/lhcb/rrm42/fcc/{i}totbkg.pdf plot saved")
    
