import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotters.matt_paper_plotter import make_the_bloody_plot

path = "outputs/full_prelim_cuts_500k"
#presel_cuts = "(EVT_e < 85) & (EVT_hemisEmin_nCharged > 0) & (EVT_hemisEmin_nLept == 0) & (PV_Rec_vtx_m < 40) & (EVT_hemisEmax_n > 10)"

samples_heavy = ["combined_signal", "heavy_hadronic_background"]
samples_light = ["combined_signal", "light_hadronic_background"]

make_the_bloody_plot( var="EVT_hemisEmax_nDV", range=(0,8), bins=8, 
                      inputpath=path, samples=samples_heavy, 
                      xtitle='Number of displaced vertices in the non-signal hemisphere', 
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmax_nDV_heavy.pdf" )

make_the_bloody_plot( var="EVT_hemisEmax_nDV", range=(0,8), bins=8, 
                      inputpath=path, samples=samples_light, 
                      xtitle='Number of displaced vertices in the non-signal hemisphere', 
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmax_nDV_light.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nDV", range=(0,8), bins=8, 
                      inputpath=path, samples=samples_heavy, 
                      xtitle='Number of displaced vertices in the signal hemisphere', 
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmin_nDV_heavy.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nDV", range=(0,8), bins=8, 
                      inputpath=path, samples=samples_light, 
                      xtitle='Number of displaced vertices in the signal hemisphere', 
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmin_nDV_light.pdf" )

plt.show()




