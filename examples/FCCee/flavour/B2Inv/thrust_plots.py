import matplotlib.pyplot as plt
from matt_paper_plotter import make_the_bloody_plot

path = "/r01/lhcb/ejnw2/fcc/FCC_outputs_2025/outputs_28Jan25/no_selection_no_nlept_var/"
samples = ["Bdsignal", "bb_only"]

make_the_bloody_plot( "(180/3.141592653589793)*(arccos((EVT_Thrust_x * MCq1_px + EVT_Thrust_y * MCq1_py + EVT_Thrust_z * MCq1_pz) / sqrt((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2) * (MCq1_px**2 + MCq1_py**2 + MCq1_pz**2))))",
                      range=(0,40), bins=16,
                      inputpath=path, samples=samples, 
                      xtitle=r"$\Delta\phi$ [$^\circ$]",
                      save="plots/paper_plots/Using_Matts_plotter/EVT_ThrustAngle_min.pdf")

make_the_bloody_plot( "(180/3.141592653589793)*(arccos((EVT_Thrust_x * MCq1_px + EVT_Thrust_y * MCq1_py + EVT_Thrust_z * MCq1_pz) / sqrt((EVT_Thrust_x**2 + EVT_Thrust_y**2 + EVT_Thrust_z**2) * (MCq1_px**2 + MCq1_py**2 + MCq1_pz**2))))",
                      range=(140,180), bins=16,
                      inputpath=path, samples=samples, 
                      xtitle=r"$\Delta\phi$ [$^\circ$]",
                      save="plots/paper_plots/Using_Matt's_plotter/EVT_ThrustAngle_max.pdf")

plt.show()

