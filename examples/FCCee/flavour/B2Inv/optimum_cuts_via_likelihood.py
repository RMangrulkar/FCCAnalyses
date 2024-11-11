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

def likelihood_model_builder(signal_bf=1e-5,
                            MVA1_cut = [0.994,0.995],#list st can do gridsearch
                            MVA2_cut = [0.95],#list st can do gridsearch
                            nbins = 5,
                            ntoys = 2, #250
                            stage='stage2',
                            save=False):


    """ 
    likelihood_model_builder(**opts ) will return optimum point from minimising signal error on fit to toys

    Parameters
    ----------
    signal_bf : float, optional
        signal branching fraction to use. Default: 1e-5
    MVA1_cut : list, optional
        MVA1 cuts to test. Default: [0.994,0.995]
    MVA2_cut : list, optional
        MVA2 cuts to test. Default: [0.95]
    nbins: int, optional
        number of bins to use in fit. Default:5
    ntoys: int, optional
        number of toys to use per test cut
    stage: str,optional
        'stage1' or 'stage2' data. Default: 'stage2'
    save : Bool, optional
        If True saves outputs in text file. Default: False

    """


    # Getting samples
    folder = f'outputs/{stage}'
    files = {sample: glob(os.path.join(folder, sample, '*.root')) for sample in cfg.samples}
            
    #Creating array to save sc_s_err in so that can save
    overall_sc_s_err_arr=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    overall_sc_s_arr=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    overall_sc_b_err_arr=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    overall_sc_b_arr=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    significancelike=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    significance=np.zeros((len(MVA1_cut),len(MVA2_cut)))
    
    S_exp_dict = {}
    B_tot_exp_dict = {}
    S_sig_dict = {}
    B_sig_dict = {}

    for i in range(len(MVA1_cut)):
        for j in range(len(MVA2_cut)):
            #defining key params
            xrange = (MVA2_cut[j], 1.00)
            bins = np.linspace(xrange[0],xrange[1],nbins+1)
            #defining bin centers
            bin_centres = (bins[:-1]+bins[1:])/2

            #defining bkg labels
            bkg_labels=cfg.sample_allocations['background']
            n_bkgs=len(bkg_labels)

            data = {}
            for sample in cfg.samples:
                tree = [f'{f}:events' for f in files[sample]]
                data[sample] = uproot.concatenate(tree, expressions='EVT_MVA2', cut=f'(EVT_MVA1 >= {MVA1_cut[i]}) & (EVT_MVA2 >= {MVA2_cut[j]})')['EVT_MVA2']
            
            # CReating MVA2 cut expressions per bin
            cut_expr = []
            for b in range(len(bins)):
                if (b+1) == len(bins):
                    break
                elif (b+2) == len(bins):
                    cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut[i]}) & (EVT_MVA2 >= {bins[b]}) & (EVT_MVA2 <= {bins[b+1]})')
                else:
                    cut_expr.append(f'(EVT_MVA1 >= {MVA1_cut[i]}) & (EVT_MVA2 >= {bins[b]}) & (EVT_MVA2 < {bins[b+1]})')
            
            # getting sample expectations and their error
            effs = get_efficiencies(stage, cut=cut_expr, raw=True, verbose=False)
            
            #Adding line to stop if cut any MC samples to zero to avoid further errors
            if not all(np.all(effs[sample+'_eff']) for sample in cfg.samples):
                print(f'At least one bkg MC sample has been reduced to zero. Use less tight cuts that {cut_expr}.')
                exit()
                
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
            
            
            
            # Now define signal and poisson expectation
            S = nums['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_num']
            sig_S = nums['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu_err']
            poisson_expectation = B_tot + S

            #adding S and Btot to array
            S_exp_dict[f'[{i}][{j}]'] = S
            B_tot_exp_dict[f'[{i}][{j}]'] = B_tot
            S_sig_dict[f'[{i}][{j}]'] = sig_S
            B_sig_dict[f'[{i}][{j}]'] = sig_B
            
            
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
            
            # throw and refit toys
            for n in range(ntoys):
                toy_data = np.random.poisson(poisson_expectation) #throw toys
                mi = Minuit(poisson_likelihood, sc_b=1, sc_s=1 ) #fit toy
                mi.migrad()
                mi.hesse()
                if mi.valid:
                    sc_s_arr.append(mi.values['sc_s'])
                    sc_b_arr.append(mi.values['sc_b'])
                    sc_s_err_arr.append(mi.errors['sc_s'])
                    sc_b_err_arr.append(mi.errors['sc_b'])
                    fval_arr.append(mi.fval)
                
            
            sc_b_av = np.average(sc_b_arr)
            sc_s_av = np.average(sc_s_arr)
            sc_b_err_av = np.average(sc_b_err_arr)
            sc_s_err_av = np.average(sc_s_err_arr)

            overall_sc_s_err_arr[i][j]=sc_s_err_av
            overall_sc_s_arr[i][j]=sc_s_av 
            overall_sc_b_arr[i][j]=sc_b_av 

            #significance calculation
            sigma_sq=0
            srootsb_sq=0
            for k in range(nbins):
                sigma_sq += (S[k]/np.sqrt(S[k]+B_tot[k]+sig_S[k]**2+sig_B[k]**2))**2
                srootsb_sq += (S[k]/np.sqrt(S[k]+B_tot[k]))**2
            sigma = np.sqrt(sigma_sq)
            srootsb = np.sqrt(srootsb_sq)
            significancelike[i][j]=sigma
            significance[i][j]=srootsb

            

    print(np.min(overall_sc_s_err_arr))
    #print(np.where(overall_sc_s_err_arr == np.min(overall_sc_s_err_arr))[1]) #this does spit out index - good!
    opt_i = np.where(overall_sc_s_err_arr == np.min(overall_sc_s_err_arr))[0]
    opt_i = opt_i[0]
    opt_j = np.where(overall_sc_s_err_arr == np.min(overall_sc_s_err_arr))[1]
    opt_j = opt_j[0]
    print(f'Optimum cuts: MVA1>={MVA1_cut[opt_i]} and MVA2>={MVA2_cut[opt_j]}')
    print(f'Significance_like figure at optimum: {significancelike[opt_i][opt_j]}')
    print(f'S/sqrt(S+B) at optimum: {significance[opt_i][opt_j]}')
    

    if save:
        # Open a file in write mode
        with open('Likelihood_optimisation.txt', 'w') as file:
        # Write some text to the file
            file.write(f'Optimum cuts: MVA1>={MVA1_cut[opt_i]} and MVA2>={MVA2_cut[opt_j]}\n')
            file.write(f'Significance_like figure at optimum:{significancelike[opt_i][opt_j]}\n') #incl. systematic
            file.write(f'S/sqrt(S+B) at optimum: {significance[opt_i][opt_j]}\n')
            file.write(f'MVA1 values tested: {MVA1_cut}')
            file.write(f'MVA2 values tested: {MVA2_cut}')
    

    



            
            
            
