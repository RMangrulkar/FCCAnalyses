import config as cfg
import numpy as np
import matplotlib.pyplot as plt
from efficiency_finder import get_efficiencies, get_sample_expectations
import os
from glob import glob
import uproot
import awkward as ak
import matplotlib as mpl
from scipy.stats import poisson, norm
from iminuit import Minuit


#Defining key variables
signal_bf=1e-5
MVA1_cut = 0.994
MVA2_cut = 0.95
nbins = 5
xrange = (MVA2_cut, 1.00)
bins = np.linspace(xrange[0],xrange[1],nbins+1)
ntoys = 2#250

#defining coloiur scheme and binning
reds = mpl.colormaps['Reds_r']
cols = reds(np.linspace(0, 1, 8)[2:-2])
bin_centres = (bins[:-1]+bins[1:])/2

# Getting samples
folder = 'outputs/stage2'
files = {sample: glob(os.path.join(folder, sample, '*.root')) for sample in cfg.samples}

data = {}
for sample in cfg.samples:
    tree = [f'{f}:events' for f in files[sample]]
    data[sample] = uproot.concatenate(tree, expressions='EVT_MVA2', cut=f'(EVT_MVA1 >= {MVA1_cut}) & (EVT_MVA2 >= {MVA2_cut})')['EVT_MVA2']

# CReating MVA2 cut expressions per bin
cut_expr = []
for i in range(len(bins)):
    if (i+1) == len(bins):
        break
    elif (i+2) == len(bins):
        cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut}) & (EVT_MVA2 >= {bins[i]}) & (EVT_MVA2 <= {bins[i+1]})')
    else:
        cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut}) & (EVT_MVA2 >= {bins[i]}) & (EVT_MVA2 < {bins[i+1]})')

# getting sample expectations and their error
effs = get_efficiencies('stage2', cut=cut_expr, raw=True, verbose=False)
nums = get_sample_expectations(effs, signal_bf, verbose=False, cut=cut_expr)

a=0
#using to calculate total background expectation
B_tot = np.zeros_like(nums['p8_ee_Zbb_ecm91_num'])
B_var = np.zeros_like(nums['p8_ee_Zbb_ecm91_err'])

for sample in cfg.samples:
    if sample in cfg.sample_allocations['background']:
        if a==0:
            B_tot = B_tot + nums[sample+'_num']
            B_var = B_var + nums[sample+'_err']**2
            B = nums[sample+'_num']
            B_err = nums[sample+'_err']
        else:
            B_tot = B_tot + nums[sample+'_num']
            B_var = B_var + nums[sample+'_err']**2
            B_err = np.vstack((B_err,nums[sample+'_err']))
            B = np.vstack((B,nums[sample+'_num']))
        a+=1
#defining bkg errors
sig_B = np.sqrt(B_var)
average_background_error = np.mean( sig_B / B_tot )
#defining bkg labels
bkg_labels=cfg.sample_allocations['background']
n_bkgs=len(bkg_labels)


# Now define signal and poisson expectation
S = nums['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_num']
sig_S = nums['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_err']
poisson_expectation = B_tot + S


## define fit to toy (this is the negative log likelihood to minimize)
def poisson_likelihood(sc_b, sc_s):
    expectation = sc_b * B_tot + sc_s * S
    poiss_term = -np.sum( poisson.logpmf(toy_data, expectation)) #logpmf = Log of the probability mass function
    bkg_constraint_term = -norm.logpdf( sc_b, 1, average_background_error )
    return poiss_term + bkg_constraint_term

sc_s_arr=[]
sc_s_err_arr=[]
sc_b_arr=[]
sc_b_err_arr=[]
fval_arr=[]
mi_arr=[]

# throw and refit toys
for n in range(ntoys):
    toy_data = np.random.poisson(poisson_expectation) #throw toys
    mi = Minuit(poisson_likelihood, sc_b=1, sc_s=1 ) #fit toy
    mi.migrad()
    mi.hesse()
    sc_s_arr.append(mi.values['sc_s'])
    sc_b_arr.append(mi.values['sc_b'])
    sc_s_err_arr.append(mi.errors['sc_s'])
    sc_b_err_arr.append(mi.errors['sc_b'])
    fval_arr.append(mi.fval)
    mi_arr.append(mi)


sc_b_av = np.average(sc_b_arr)
sc_s_av = np.average(sc_s_arr)
sc_b_err_av = np.average(sc_b_err_arr)
sc_s_err_av = np.average(sc_s_err_arr)

sc_s_stdev = np.std(sc_s_arr) #if take enough toys expect to match sc_s_err_av
sc_b_stdev = np.std(sc_b_arr) #if take enough toys expect to match sc_b_err_av




 