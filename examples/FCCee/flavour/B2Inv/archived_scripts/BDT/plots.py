# plot.py
# LEGACY -- Julio's code
import numpy as np
import ROOT
from matplotlib import pyplot as plt
import pandas as pd

ROOT.EnableImplicitMT()


def plot_comparison(var):
    fig, ax = plt.subplots()
    _, bins, _ = plt.hist(df_signal_pd[var], bins=40,
                          histtype='step', label=r'$B_s \rightarrow \nu \bar{\nu}$')
    _, bins, _ = plt.hist(df_bb_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow b \bar{b}$')
    _, bins, _ = plt.hist(df_cc_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow c \bar{c}$')
    _, bins, _ = plt.hist(df_ss_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow s \bar{s}$')
    _, bins, _ = plt.hist(df_ud_pd[var], bins=bins,
                          histtype='step', label=r'$Z \rightarrow q \bar{q},q \in \{u,d\}$')
    plt.xlabel(var)
    #plt.xlim(0.40, 0.52)
    plt.xlim(bins[0], bins[-1])
    plt.legend(loc='upper right')

    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax.text(0.725, 0.4, transform=ax.transAxes, bbox=props)


path = 'root://eospublic.cern.ch//eos/experiment/fcc/ee/analyses_storage/flavour/B2Inv/'

listvars = {"EVT_ThrustEmin_E", 
            "EVT_ThrustEmax_E",
            "EVT_ThrustEmin_Echarged",
            "EVT_ThrustEmax_Echarged",
            "EVT_ThrustEmin_Eneutral",
            "EVT_ThrustEmax_Eneutral",
            "EVT_ThrustEmin_Ncharged",
            "EVT_ThrustEmax_Ncharged",
            "EVT_ThrustEmin_Nneutral",
            "EVT_ThrustEmax_Nneutral",
            "EVT_NtracksPV",
            "EVT_NVertex",
            "EVT_ThrustEmin_NDV",
            "EVT_ThrustEmax_NDV",
            "EVT_dPV2DVmin",
            "EVT_dPV2DVmax",
            "EVT_dPV2DVave",
            "EVT_Thrust_Mag",
            "EVT_Thrust_X", 
            "EVT_Thrust_Y", 
            "EVT_Thrust_Z"} 

df_signal = ROOT.RDataFrame('events',path+'Bs2NuNuSig.root',listvars)
df_bb = ROOT.RDataFrame('events',path+'ZbbBkg.root',listvars)
df_cc = ROOT.RDataFrame('events',path+'ZccBkg.root')
df_ss = ROOT.RDataFrame('events',path+'ZssBkg.root')
df_ud = ROOT.RDataFrame('events',path+'ZudBkg.root')

#cols = ['EVT_ThrustEmin_E', 'EVT_ThrustEmax_E']

cols = ["EVT_ThrustEmin_E",
        "EVT_ThrustEmax_E",
        "EVT_ThrustEmin_Echarged",
        "EVT_ThrustEmax_Echarged",
        "EVT_ThrustEmin_Eneutral",
        "EVT_ThrustEmax_Eneutral",
        "EVT_ThrustEmin_Ncharged",
        "EVT_ThrustEmax_Ncharged",
        "EVT_ThrustEmin_Nneutral",
        "EVT_ThrustEmax_Nneutral",
        "EVT_NtracksPV",
        "EVT_NVertex",
        "EVT_ThrustEmin_NDV",
        "EVT_ThrustEmax_NDV",
        "EVT_dPV2DVmin",
        "EVT_dPV2DVmax",
        "EVT_dPV2DVave",
        "EVT_Thrust_Mag",
        "EVT_Thrust_X",
        "EVT_Thrust_Y",
        "EVT_Thrust_Z"]

df_signal_np = df_signal.AsNumpy(columns=cols)
df_bb_np = df_bb.AsNumpy(columns=cols)
df_cc_np = df_cc.AsNumpy(columns=cols)
df_ss_np = df_ss.AsNumpy(columns=cols)
df_ud_np = df_ud.AsNumpy(columns=cols)

df_signal_pd = pd.DataFrame(df_signal_np)
df_bb_pd = pd.DataFrame(df_bb_np)
df_cc_pd = pd.DataFrame(df_cc_np)
df_ss_pd = pd.DataFrame(df_ss_np)
df_ud_pd = pd.DataFrame(df_ud_np)

plt.figure()
plot_comparison('EVT_ThrustEmin_E')
plt.savefig('./output/comp_emin_e.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_E')
plt.savefig('./output/comp_emax_e.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmin_Echarged')
plt.savefig('./output/comp_emin_echarged.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_Echarged')
plt.savefig('./output/comp_emax_echarged.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmin_Eneutral')
plt.savefig('./output/comp_emin_eneutral.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_Eneutral')
plt.savefig('./output/comp_emax_eneutral.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmin_Ncharged')
plt.savefig('./output/comp_emin_ncharged.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_Ncharged')
plt.savefig('./output/comp_emax_ncharged.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmin_Nneutral')
plt.savefig('./output/comp_emin_nneutral.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_Nneutral')
plt.savefig('./output/comp_emax_nneutral.pdf')

plt.figure()
plot_comparison('EVT_NtracksPV')
plt.savefig('./output/comp_ntrackspv.pdf')

plt.figure()
plot_comparison('EVT_NVertex')
plt.savefig('./output/comp_nvertex.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmin_NDV')
plt.savefig('./output/comp_emin_ndv.pdf')

plt.figure()
plot_comparison('EVT_ThrustEmax_NDV')
plt.savefig('./output/comp_emax_ndv.pdf')

plt.figure()
plot_comparison('EVT_dPV2DVmin')
plt.savefig('./output/comp_dpv2dvmin.pdf')

plt.figure()
plot_comparison('EVT_dPV2DVmax')
plt.savefig('./output/comp_dpv2dvmax.pdf')

plt.figure()
plot_comparison('EVT_dPV2DVave')
plt.savefig('./output/comp_dpv2dvave.pdf')

plt.figure()
plot_comparison('EVT_Thrust_Mag')
plt.savefig('./output/comp_mag.pdf')

plt.figure()
plot_comparison('EVT_Thrust_X')
plt.savefig('./output/comp_x.pdf')

plt.figure()
plot_comparison('EVT_Thrust_Y')
plt.savefig('./output/comp_y.pdf')

plt.figure()
plot_comparison('EVT_Thrust_Z')
plt.savefig('./output/comp_z.pdf')

#h = df_signal.Histo1D("EVT_ThrustEmin_E")
#h.Draw()
