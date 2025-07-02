import matplotlib.pyplot as plt
from matt_paper_plotter import make_the_bloody_plot

path = "outputs/no_selection"
samples = ["combined_signal", "hadronic_background"]
tau_samples = ["combined_signal", "tau_background"]

make_the_bloody_plot( var="EVT_hemisEmax_n", range=(0,60), bins=60, 
                      inputpath=path, samples=tau_samples, 
                      xtitle='Non-Signal Hemisphere Particle Multiplicity', 
                      plot_cutline=(10,"Preselection cut",">"),
                      save="plots/paper_plots/Using_Matts_plotter/EVT_hemisEmax_n_tau_updatedlabel.pdf" )

make_the_bloody_plot( var="EVT_e", range=(25,95), bins=70, 
                      inputpath=path, samples=samples, 
                      xtitle='Total Event Energy [GeV]', 
                      plot_cutline=(85,"Preselection cut","<"), 
                      save="plots/paper_plots/Using_Matts_plotter/EVT_e_prelim.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nCharged", range=(0,25), bins=25, 
                      inputpath=path, samples=samples, 
                      xtitle='Number of Charged Tracks in the Signal Hemisphere', 
                      plot_cutline=(1,"Preselection cut",">"),
                      save="plots/paper_plots/Using_Matts_plotter/EVT_hemisEmin_nCharged_prelim.pdf" )

make_the_bloody_plot( var="EVT_hemisEmin_nLept", range=(0,5), bins=5, 
                      inputpath=path, samples=samples, 
                      xtitle='Total Number of $e^\pm$ and $\mu^\pm$ in the Signal Hemisphere', 
                      plot_cutline=(1,"Preselection cut","<"),
                      save="plots/paper_plots/Using_Matts_plotter/EVT_hemisEmin_nLept_prelim.pdf" )

make_the_bloody_plot( var="PV_Rec_vtx_m", range=(0,90), bins=45, 
                      inputpath=path, samples=samples, 
                      xtitle='Reconstructed Mass of the Primary Vertex [GeV]', 
                      plot_cutline=(40,"Preselection cut","<"),
                      save="plots/paper_plots/Using_Matts_plotter/PV_Rec_vtx_m_prelim.pdf" )

plt.show()
