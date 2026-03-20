import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotters.matt_paper_plotter import make_the_bloody_plot

path = "outputs/no_selection"#"outputs/no_selection"
samples = ["combined_signal", "hadronic_background"]
tau_samples = ["combined_signal", "tau_background"]

make_the_bloody_plot( var="EVT_hemisEmax_n", range=(0,60), bins=60, 
                      inputpath=path, samples=tau_samples, 
                      xtitle='Non-signal hemisphere particle multiplicity', 
                      plot_cutline=(10,"Preselection cut",">"),
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmax_n_tau_updatedlabel.pdf" )#plots/paper_plots/Using_Matts_plotter/March26

make_the_bloody_plot( var="EVT_e", range=(25,95), bins=70, 
                      inputpath=path, samples=samples, 
                      xtitle='Total event energy [GeV]', 
                      plot_cutline=(85,"Preselection cut","<"), 
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_e_prelim.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nCharged", range=(0,25), bins=25, 
                      inputpath=path, samples=samples, 
                      xtitle='Number of charged tracks in the signal hemisphere', 
                      plot_cutline=(1,"Preselection cut",">"),
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmin_nCharged_prelim.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nLept", range=(0,5), bins=5, 
                      inputpath=path, samples=samples, 
                      xtitle='Total number of $e^\pm$ and $\mu^\pm$ in the signal hemisphere', 
                      plot_cutline=(1,"Preselection cut","<"),
                      save="plots/paper_plots/JHEP_proofs_replies/EVT_hemisEmin_nLept_prelim.pdf" )

make_the_bloody_plot( var="PV_Rec_vtx_m", range=(0,90), bins=45, 
                      inputpath=path, samples=samples, 
                      xtitle='Reconstructed mass of the primary vertex [GeV]', 
                      plot_cutline=(40,"Preselection cut","<"),
                      save="plots/paper_plots/JHEP_proofs_replies/PV_Rec_vtx_m_prelim.pdf" )

plt.show()
