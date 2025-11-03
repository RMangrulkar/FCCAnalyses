import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import glob
import sys
import ROOT
import math
from tabulate import tabulate
from yaml import safe_load, YAMLError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from itertools import combinations
from itertools import product
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg 
import bdt_lh_cut_opt_significance as cutopt
from efficiency_tools import efficiency_finder

plt.style.use('fcc.mplstyle')

ROOT.EnableImplicitMT()

def set_outputpath(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    return outputpath


#desired branches
flavtag_branches = ["Rec_e","Rec_p","Rec_px","Rec_py","Rec_pz","Rec_pt","Rec_true_PDG","Rec_in_hemisEmin","Rec_indvtx","Rec_vtx_isPV", "Rec_true_M1", "Rec_true_M2","Rec_true_M1ofM1","Rec_true_M2ofM1","Rec_true_M1ofM2","Rec_true_M2ofM2","decay" ] # would be good to add "Rec_track_absd0","Rec_track_absnormd0"
reco_space_branches = ["Rec_e","Rec_p","Rec_px","Rec_py","Rec_pz","Rec_pt","Rec_true_PDG","Rec_in_hemisEmin","Rec_indvtx","Rec_true_M1", "Rec_true_M2","Rec_true_M1ofM1","Rec_true_M2ofM1","Rec_true_M1ofM2","Rec_true_M2ofM2"] # would be good to add "Rec_track_absd0","Rec_track_absnormd0"

#############################################
#define function to select desired particles#
#############################################
def filter_df_on_IDs(dfbsfull,dfbdfull,mask_fn, reco_space_branches= reco_space_branches):
    #mask = (np.array(evt['Rec_in_hemisEmin']) == 1) & ((np.array(np.round(evt['Rec_true_PDG'])) == 321)| (np.array(np.round(evt['Rec_true_PDG'])) == -321)
    filtered_df_dict={}
    key = ['bs','bd']
    for s in range(2):
        df = [dfbsfull,dfbdfull][s]
       
        #create empty dfs to fill
        filtered_df = pd.DataFrame(columns= reco_space_branches )
        
        #loop through events
        for i in range(len(df[reco_space_branches[0]])):
            evt = df.iloc[i]
            mask = mask_fn(evt)
            for var in reco_space_branches:
                filtered_df.loc[i, var] = np.array(evt[var])[mask]
                
        filtered_df_dict[key[s]] = filtered_df
   
    return filtered_df_dict



def filter_df_on_fromPV(dfbsfull, dfbdfull, reco_space_branches= reco_space_branches):
    key = ['bs','bd']
    filtered_df_dict={}
    for s in range(2):
        df = [dfbsfull,dfbdfull][s]
        
        #create empty dfs to fill
        filtered_df=pd.DataFrame(columns= reco_space_branches )
        
        #loop through events
        for i in range(len(df[reco_space_branches[0]])):
            from_PV=[]
            for k in range(len(df['Rec_indvtx'][i])):
                if df['Rec_indvtx'][i][k] == -999:
                    from_PV.append(0)
                else:
                    from_PV.append(np.array(df['Rec_vtx_isPV'][i])[df['Rec_indvtx'][i][k]])
            
            frompv_mask = np.array(from_PV)==1
            
            for var in reco_space_branches:
                filtered_df.loc[i, var] = df.loc[i, var][frompv_mask]
    
        filtered_df_dict[key[s]] = filtered_df
    
    return filtered_df_dict


################################################
#define masks for selecting differnt types of K#
################################################
def charged_fsK_mask(evt):
    return ((np.array(evt['Rec_in_hemisEmin']) == 1) & ((np.round(evt['Rec_true_PDG']) == 321) | (np.round(evt['Rec_true_PDG']) == -321)))

def fsK_mask(evt):
    return ((np.array(evt['Rec_in_hemisEmin']) == 1) & ((np.array(np.round(evt['Rec_true_PDG'])) == 321)| (np.array(np.round(evt['Rec_true_PDG'])) == -321)| (np.array(np.round(evt['Rec_true_PDG'])) == 130)))

def fsKS_mask(evt):
    return ((np.array(evt['Rec_in_hemisEmin']) == 1) & ((np.array(np.round(evt['Rec_true_M1'])) == 310)|(np.array(np.round(evt['Rec_true_M2'])) == 310)|(np.array(np.round(evt['Rec_true_M1ofM1'])) == 310)|(np.array(np.round(evt['Rec_true_M2ofM1'])) == 310)|(np.array(np.round(evt['Rec_true_M1ofM2'])) == 310)|(np.array(np.round(evt['Rec_true_M2ofM2'])) == 310)))

def fsKL_mask(evt):
    return ((np.array(evt['Rec_in_hemisEmin']) == 1) & (np.array(np.round(evt['Rec_true_PDG'])) == 130))

def reco_pi0(evt):
    return (((np.round(evt['Rec_true_PDG']) == 111) | (np.round(evt['Rec_true_PDG']) == -111)))



##################
#define plotters #
################## 
#define histogram settings from config
hist_settings={'Bssignal': {}, 'Bdsignal':{} }

hist_settings['Bssignal']['histtype'] = 'step'
hist_settings['Bssignal']['lw'] = 2
hist_settings['Bssignal']['color'] = plt.cm.Blues( np.linspace(0, 1, len(cfg.sample_allocations['combined_signal'])+4)[3]) #'cornflowerblue'
hist_settings['Bssignal']['hatch'] = '////'

hist_settings['Bdsignal']['histtype'] = 'step'
hist_settings['Bdsignal']['lw'] = 2
hist_settings['Bdsignal']['color'] = plt.cm.Blues( np.linspace(0, 1, len(cfg.sample_allocations['combined_signal'])+4)[-1]) #-2#'mediumblue'#'royalblue'
hist_settings['Bdsignal']['hatch'] = r'\\\\'

"""
for key in ['Bssignal','Bdsignal']:
    hist_settings[key]['histtype'] = 'step'
    hist_settings[key]['lw'] = 2
    hist_settings[key]['color'] = cfg.sample_colors[key]
    hist_settings[key]['hatch'] = cfg.sample_hatches[key]
"""

def make_maxpK_efficiency_plot(filtered_dict,K_type = ' Signal Side Final State $K$ ($K^\pm$ or $K^0_L$) from PV',savepath=None,plottype=None):
    dfbs = filtered_dict['bs']
    dfbd = filtered_dict['bd']
    maxp_bs =[np.max(dfbs['Rec_p'][i]) if len(dfbs['Rec_p'][i]) != 0 else -1 for i in range(len(dfbs))]
    maxp_bd =[np.max(dfbd['Rec_p'][i]) if len(dfbd['Rec_p'][i]) != 0 else -1 for i in range(len(dfbd))]


    #plot efficinecy of p cut
    n_remaining_bs = []
    n_remaining_bd = []
    cut_arr = []
    for i in np.linspace(0,max(maxp_bs),400):
        filter_bs_arr = np.array(maxp_bs) >i
        filter_bd_arr= np.array(maxp_bd) >i
        n_remaining_bs.append(len(np.array(maxp_bs)[filter_bs_arr])/len(maxp_bs))
        n_remaining_bd.append(len(np.array(maxp_bd)[filter_bd_arr])/len(maxp_bd))
        cut_arr.append(i)

    n_remaining_bs_interp = interp1d(cut_arr, n_remaining_bs, kind = 'quadratic')
    n_remaining_bd_interp = interp1d(cut_arr, n_remaining_bd, kind = 'quadratic')

    cut_arr_interp = np.linspace(0,max(maxp_bs),1000)
    plt.figure()
    plt.plot(cut_arr,n_remaining_bd,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'], color = hist_settings['Bdsignal']['color'],lw=2,linestyle=(0,(3,1,1,1))) 
    plt.plot(cut_arr,n_remaining_bs,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'], color = hist_settings['Bssignal']['color'],lw=2) 
    plt.xlim(0,8)
    plt.legend()
    plt.xlabel('Cut on $p$/GeV of Highest $p$ '+f'{K_type}')
    plt.ylabel('Cut Efficiency')
    if savepath!=None:
        plt.savefig(os.path.join(set_outputpath(savepath),f'{plottype}_cut_eff.pdf'))
    plt.show()


def  make_nK_plot(filtered_dict,K_type = ' Signal Side Final State $K$ ($K^\pm$ or $K^0_L$) from PV',savepath=None,plottype=None):
    dfbs = filtered_dict['bs']
    dfbd = filtered_dict['bd']
    num_K_bs =[len(dfbs['Rec_p'][i]) for i in range(len(dfbs))]
    num_K_bd =[len(dfbd['Rec_p'][i]) for i in range(len(dfbd))]

    #plot hostogram
    plt.figure()
    plt.hist(num_K_bd,bins=6,density=True,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'],range=(0,6), **hist_settings['Bdsignal'])
    plt.hist(num_K_bs,bins=6,density=True,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'],range=(0,6), **hist_settings['Bssignal'])
    plt.xlabel('Number of'+f'{K_type}')
    plt.ylabel('Density')
    plt.legend()
    if savepath!=None:
        plt.savefig(os.path.join(set_outputpath(savepath),f'{plottype}_number.pdf'))
    plt.show()


def make_maxpKS_efficiency_plot(KS_dict,K_type = ' Signal Side $K_s^0$',savepath=None,plottype=None):  
    dfbs = KS_dict['bs']
    dfbd = KS_dict['bd']
    maxp_bs =[np.max(dfbs['KS_p'][i]) if len(dfbs['KS_p'][i]) != 0 else -1 for i in range(len(dfbs))]
    maxp_bd =[np.max(dfbd['KS_p'][i]) if len(dfbd['KS_p'][i]) != 0 else -1 for i in range(len(dfbd))]


    #plot efficinecy of p cut
    n_remaining_bs = []
    n_remaining_bd = []
    cut_arr = []
    for i in np.linspace(0,max(maxp_bs),1000):
        filter_bs_arr = np.array(maxp_bs) >i
        filter_bd_arr= np.array(maxp_bd) >i
        n_remaining_bs.append(len(np.array(maxp_bs)[filter_bs_arr])/len(maxp_bs))
        n_remaining_bd.append(len(np.array(maxp_bd)[filter_bd_arr])/len(maxp_bd))
        cut_arr.append(i)
    plt.figure()
    plt.plot(cut_arr,n_remaining_bd,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'], color = hist_settings['Bdsignal']['color'],lw=2,linestyle=(0,(3,1,1,1))) 
    plt.plot(cut_arr,n_remaining_bs,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'], color = hist_settings['Bssignal']['color'],lw=2) 
    plt.xlim(0,6)
    plt.legend()
    plt.xlabel('Cut on $p$/GeV of Highest $p$ '+f'{K_type}')
    plt.ylabel('Cut Efficiency')
    if savepath!=None:
        plt.savefig(os.path.join(set_outputpath(savepath),f'{plottype}_efficiency.pdf'))
    plt.show()


def make_nKS_plot(KS_dict,K_type = ' Signal Side $K_s^0$',savepath=None,plottype=None):
    dfbs = KS_dict['bs']
    dfbd = KS_dict['bd']
    num_K_bs =[len(dfbs['KS_p'][i]) for i in range(len(dfbs))]
    num_K_bd =[len(dfbd['KS_p'][i]) for i in range(len(dfbd))]

    #plot hostogram
    plt.figure()
    plt.hist(num_K_bd,bins=5,density=True,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bd2NuNu'],range=(0,5), **hist_settings['Bdsignal'])
    plt.hist(num_K_bs,bins=5,density=True,label=cfg.titles['p8_ee_Zbb_ecm91_EvtGen_Bs2NuNu'],range=(0,5), **hist_settings['Bssignal'])
    plt.xlabel('Number of'+f'{K_type}')
    plt.ylabel('Density')
    plt.legend()
    if savepath!=None:
        plt.savefig(os.path.join(set_outputpath(savepath),f'{plottype}_number.pdf'))
    plt.show()


##############################
#define reconstruction for Ks#
############################## 
def run_pairing_reconstruction(E,px,py,pz, parent_invariant_mass):

    n = len(E)
    indices = list(range(len(E)))

    # Step 1: Generate all index pairs (i < j)
    all_pairs = list(combinations(indices, 2))

    # Step 2: Filter for combinations of pairs that:
    # - cover all indices
    # - do not share any index (i.e., disjoint pairs)
    def is_disjoint(pairing):
        used = set()
        for i, j in pairing:
            if i in used or j in used:
                return False
            used.update([i, j])
        return True

    # Step 3: Generate all combinations of n//2 pairs
    valid_pairings = []
    for combo in combinations(all_pairs, n // 2):
        if is_disjoint(combo):
            valid_pairings.append(combo)


    # Step 4: Compute the sum of each valid pairing set
    results = []
    for pairing in valid_pairings:

        total_E = np.array([E[i] + E[j] for i, j in pairing])
        total_px =  np.array([px[i] + px[j] for i, j in pairing])
        total_py =  np.array([py[i] + py[j] for i, j in pairing])
        total_pz =  np.array([pz[i] + pz[j] for i, j in pairing])

        abs_total_inv_mass_diff_sum = sum(abs(np.sqrt(total_E**2 - (total_px**2+total_py**2+total_pz**2)) - parent_invariant_mass))
        results.append((pairing, abs_total_inv_mass_diff_sum))

    min_pairing, min_total = min(results, key=lambda x: x[1])


    return min_pairing


def run_charged_pairing_reconstruction(E_p,px_p,py_p,pz_p, E_m,px_m,py_m,pz_m, parent_invariant_mass):

    n_p = len(E_p)
    n_m = len(E_m)

    indicesp = list(range(len(E_p)))
    indicesm = list(range(len(E_m)))

    # Step 1: Generate all pairings (inexp,indexm)
    all_pairs = list(product(indicesp, indicesm))


    # Step 2: Filter for combinations of pairs that:
    # - cover all indices
    # - do not share any index (i.e., disjoint pairs)
    def is_disjoint(pairing):
        used = set()
        for i, j in pairing:
            if i in used or j+ len(E_p)  in used:
                return False
            used.update([i, j+ len(E_p)])
        return True

    # Step 3: Generate all combinations of n//2 pairs
    valid_pairings = []
    for combo in combinations(all_pairs, (min(n_p,n_m))): #(n_p+n_m) // 2
        if is_disjoint(combo):
            valid_pairings.append(combo)



    # Step 4: Compute the sum of each valid pairing set
    results = []
    for pairing in valid_pairings:
        total_E = np.array([E_p[i] + E_m[j] for i, j in pairing])
        total_px =  np.array([px_p[i] + px_m[j] for i, j in pairing])
        total_py =  np.array([py_p[i] + py_m[j] for i, j in pairing])
        total_pz =  np.array([pz_p[i] + pz_m[j] for i, j in pairing])

        abs_total_inv_mass_diff_sum = sum(abs(np.sqrt(total_E**2 - (total_px**2+total_py**2+total_pz**2)) - parent_invariant_mass))
        results.append((pairing, abs_total_inv_mass_diff_sum))

    min_pairing, min_total = min(results, key=lambda x: x[1])


    return min_pairing


if __name__ == "__main__":
    #################################
    #load and process data for plots#
    #################################


    #load data
    print('--> Loading data')
    data = pd.read_pickle('outputs/prelim_cuts_full_data/baseline_plus_bdtlh_dataframes/flavtag_dataframes/BF1e-06_selected_signal_flavtag_dataframe_inclmotherinfo.pkl')

    dfbsfull = data.head(20000)
    dfbdfull = data.tail(20000).reset_index(drop=True)

    #filter data to select different particles
    print('--> Filtering data on PID')
    charged_fsK_dict = filter_df_on_IDs(dfbsfull,dfbdfull,charged_fsK_mask)
    fsK_dict = filter_df_on_IDs(dfbsfull,dfbdfull,fsK_mask)
    fsKL_dict = filter_df_on_IDs(dfbsfull,dfbdfull,fsKL_mask)
    pi0dict = filter_df_on_IDs(dfbsfull,dfbdfull,reco_pi0)

    charged_fsK_dict_fromPV = filter_df_on_IDs(charged_fsK_dict['bs'],charged_fsK_dict['bd'],charged_fsK_mask)
    fsK_dict_fromPV = filter_df_on_IDs(fsK_dict['bs'],fsK_dict['bd'],fsK_mask)
    fsKL_dict_fromPV = filter_df_on_IDs(fsKL_dict['bs'],fsKL_dict['bd'],fsKL_mask)
    fsKS_dict = filter_df_on_IDs(dfbsfull,dfbdfull,fsKS_mask)

    print('--> Saving Charged K from PV dict')
    data_savepath = 'outputs/prelim_cuts_full_data/flavtag_dataframes/selected_kaons'
    with open(os.path.join(data_savepath,'flavtag_prompt_charged_K.pkl'), 'wb') as f:
        pickle.dump(charged_fsK_dict_fromPV, f)

    #make plots for charged kaons
    print('--> Saving Charged K plots')
    savepath = 'plots/paper_plots'
    #savepath = 'plots/BDTlh_baseline_plus_cut_optimisation/with_tau_veto/flavour_tagging/'
    make_maxpK_efficiency_plot(charged_fsK_dict_fromPV,K_type = ' Signal Side Final State $K^\pm$ from PV', savepath=savepath,plottype='chargedK')
    make_nK_plot(charged_fsK_dict_fromPV,K_type = ' Signal Side Final State $K^\pm$ from PV', savepath=savepath,plottype='chargedK')

    #################
    # reconstruct Ks#
    #################
    print('--> Running KS reconstruction')
    #remove empty rows for ease of figuring out reco
    filtered_KS={}
    for key in fsKS_dict.keys():
        indextoremover = fsKS_dict[key][[len(fsKS_dict[key]['Rec_p'][i]) == 0 for i in range(len(fsKS_dict[key]['Rec_p']))]].index
        filtered_KS[key] = fsKS_dict[key].drop(indextoremover, inplace=False)


    ## Create events containing fully reconstructed events only - for this must have even number of pi- and pi+ and have photons in multiples of 4
    KSreconstructed = {}
    KSpartreco = {}
    for key in filtered_KS.keys():

        part_reco=[]
        #loop through events
        for i in filtered_KS[key].index:
            evt = filtered_KS[key].loc[i]

            if (len(evt['Rec_true_PDG'][evt['Rec_true_PDG']==211]) != len(evt['Rec_true_PDG'][evt['Rec_true_PDG']==-211])):
                part_reco.append(i)
            elif (len(evt['Rec_true_PDG'][evt['Rec_true_PDG']==22])%4!=0):
                part_reco.append(i)

        KSreconstructed[key]= filtered_KS[key].drop(part_reco, inplace=False)
        KSpartreco[key]= filtered_KS[key].loc[part_reco]

    #check for multipl Ks in an event
    for key in KSpartreco.keys():

        multiple=[]
        #loop through events
        for i in KSpartreco[key].index:
            evt = KSpartreco[key].loc[i]

            if (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)])>2):
                multiple.append(i)
            elif (len(evt['Rec_true_PDG'][evt['Rec_true_PDG']==22])>4):
                multiple.append(i)


    #Run fake intermediate reconstruction of part reco events
    KS_PDG_m = cfg.mass_KS
    pi0_PDG_m = cfg.mass_pi0


    for key in KSpartreco.keys():

        #create KS_p colum
        KSpartreco[key]['KS_p'] = pd.Series(
            [np.array([]) for _ in range(len(KSpartreco[key]))],
            index=KSpartreco[key].index,dtype=object)
        

        for i in KSpartreco[key].index:
            evt = KSpartreco[key].loc[i]
            KS_p=[]


            if (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]))==2:
                px = evt['Rec_px'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                p =np.sqrt(sum(px)**2 + sum(py)**2+sum(pz)**2)
                KS_p.append(p)
                
                
            elif (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]))>2:
            
                #print(evt['Rec_true_PDG'])
                E_p = evt['Rec_e'][(evt['Rec_true_PDG']==211)]
                px_p = evt['Rec_px'][(evt['Rec_true_PDG']==211)]
                py_p = evt['Rec_py'][(evt['Rec_true_PDG']==211)]
                pz_p = evt['Rec_pz'][(evt['Rec_true_PDG']==211)]

                E_m = evt['Rec_e'][(evt['Rec_true_PDG']==-211)]
                px_m = evt['Rec_px'][(evt['Rec_true_PDG']==-211)]
                py_m = evt['Rec_py'][(evt['Rec_true_PDG']==-211)]
                pz_m = evt['Rec_pz'][(evt['Rec_true_PDG']==-211)]
                
                optimum_pairing = run_charged_pairing_reconstruction(E_p,px_p,py_p,pz_p, E_m,px_m,py_m,pz_m, KS_PDG_m)

                KS_px =  np.array([px_p[i] + px_m[j] for i, j in optimum_pairing])
                KS_py =  np.array([py_p[i] + py_m[j] for i, j in optimum_pairing])
                KS_pz =  np.array([pz_p[i] + pz_m[j] for i, j in optimum_pairing])

                p = np.sqrt(KS_px**2+KS_py**2+KS_pz**2)
                KS_p.extend(p)


            if (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==22)]))==4:
                px = evt['Rec_px'][(evt['Rec_true_PDG']==22)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==22)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==22)]
                p =np.sqrt(sum(px)**2 + sum(py)**2+sum(pz)**2)
                KS_p.append(p)

            elif (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==22)]))>4:
                E = evt['Rec_e'][(evt['Rec_true_PDG']==22)]
                px = evt['Rec_px'][(evt['Rec_true_PDG']==22)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==22)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==22)]
                #reconstruct photon pairs into neutral pions
                optimum_pairing_pi0 = run_pairing_reconstruction(E,px,py,pz, pi0_PDG_m)

                pi0_E =  np.array([E[i] + E[j] for i, j in optimum_pairing_pi0])
                pi0_px =  np.array([px[i] + px[j] for i, j in optimum_pairing_pi0])
                pi0_py =  np.array([py[i] + py[j] for i, j in optimum_pairing_pi0])
                pi0_pz =  np.array([pz[i] + pz[j] for i, j in optimum_pairing_pi0])
                #reconstruct pi0 pairs into KS
                optimum_pairing_KS = run_pairing_reconstruction(pi0_E,pi0_px,pi0_py,pi0_pz, KS_PDG_m)

                KS_px =  np.array([pi0_px[i] + pi0_px[j] for i, j in optimum_pairing_KS])
                KS_py =  np.array([pi0_py[i] + pi0_py[j] for i, j in optimum_pairing_KS])
                KS_pz =  np.array([pi0_pz[i] + pi0_pz[j] for i, j in optimum_pairing_KS])

                p = np.sqrt(KS_px**2+KS_py**2+KS_pz**2)

                KS_p.extend(p)


            # add back into initial df
            KSpartreco[key].at[i, 'KS_p']= np.array(KS_p)


    # Run fake intermediate reconstruction
    for key in KSreconstructed.keys():

        #create KS_p colum
        KSreconstructed[key]['KS_p'] = pd.Series(
            [np.array([]) for _ in range(len(KSreconstructed[key]))],
            index=KSreconstructed[key].index,dtype=object)

        for i in KSreconstructed[key].index:
            evt = KSreconstructed[key].loc[i]
            KS_p=[]


            if (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]))==2:
                px = evt['Rec_px'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]
                p =np.sqrt(sum(px)**2 + sum(py)**2+sum(pz)**2)
                KS_p.append(p)
                
                
            elif (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==211)|(evt['Rec_true_PDG']==-211)]))>2:

                E_p = evt['Rec_e'][(evt['Rec_true_PDG']==211)]
                px_p = evt['Rec_px'][(evt['Rec_true_PDG']==211)]
                py_p = evt['Rec_py'][(evt['Rec_true_PDG']==211)]
                pz_p = evt['Rec_pz'][(evt['Rec_true_PDG']==211)]

                E_m = evt['Rec_e'][(evt['Rec_true_PDG']==-211)]
                px_m = evt['Rec_px'][(evt['Rec_true_PDG']==-211)]
                py_m = evt['Rec_py'][(evt['Rec_true_PDG']==-211)]
                pz_m = evt['Rec_pz'][(evt['Rec_true_PDG']==-211)]


                #optimum_pairing = run_pairing_reconstruction(E,px,py,pz, KS_PDG_m)
                optimum_pairing = run_charged_pairing_reconstruction(E_p,px_p,py_p,pz_p, E_m,px_m,py_m,pz_m, KS_PDG_m)

                KS_px =  np.array([px_p[i] + px_m[j] for i, j in optimum_pairing])
                KS_py =  np.array([py_p[i] + py_m[j] for i, j in optimum_pairing])
                KS_pz =  np.array([pz_p[i] + pz_m[j] for i, j in optimum_pairing])

                p = np.sqrt(KS_px**2+KS_py**2+KS_pz**2)
                KS_p.extend(p)


            if (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==22)]))==4:
                px = evt['Rec_px'][(evt['Rec_true_PDG']==22)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==22)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==22)]
                p =np.sqrt(sum(px)**2 + sum(py)**2+sum(pz)**2)
                KS_p.append(p)

            elif (len(evt['Rec_true_PDG'][(evt['Rec_true_PDG']==22)]))>4:
                E = evt['Rec_e'][(evt['Rec_true_PDG']==22)]
                px = evt['Rec_px'][(evt['Rec_true_PDG']==22)]
                py = evt['Rec_py'][(evt['Rec_true_PDG']==22)]
                pz = evt['Rec_pz'][(evt['Rec_true_PDG']==22)]
                #reconstruct photon pairs into neutral pions
                optimum_pairing_pi0 = run_pairing_reconstruction(E,px,py,pz, pi0_PDG_m)

                pi0_E =  np.array([E[i] + E[j] for i, j in optimum_pairing_pi0])
                pi0_px =  np.array([px[i] + px[j] for i, j in optimum_pairing_pi0])
                pi0_py =  np.array([py[i] + py[j] for i, j in optimum_pairing_pi0])
                pi0_pz =  np.array([pz[i] + pz[j] for i, j in optimum_pairing_pi0])
                #reconstruct pi0 pairs into KS
                optimum_pairing_KS = run_pairing_reconstruction(pi0_E,pi0_px,pi0_py,pi0_pz, KS_PDG_m)

                KS_px =  np.array([pi0_px[i] + pi0_px[j] for i, j in optimum_pairing_KS])
                KS_py =  np.array([pi0_py[i] + pi0_py[j] for i, j in optimum_pairing_KS])
                KS_pz =  np.array([pi0_pz[i] + pi0_pz[j] for i, j in optimum_pairing_KS])

                p = np.sqrt(KS_px**2+KS_py**2+KS_pz**2)

                KS_p.extend(p)
            

            # add back into initial df
            KSreconstructed[key].at[i, 'KS_p']= np.array(KS_p)


    # add bakc into original df so that have- ones for non-KS
    print('--> Adding Reco KS back into df')
    for key in fsKS_dict.keys():
        fsKS_dict[key]['KS_p'] = pd.Series(
            [np.array([]) for _ in range(len(fsKS_dict[key]))],
            index=fsKS_dict[key].index,dtype=object)

        for i in KSreconstructed[key].index:
            fsKS_dict[key].at[i, 'KS_p'] = KSreconstructed[key]['KS_p'][i]

        for i in KSpartreco[key].index:
            fsKS_dict[key].at[i, 'KS_p'] = KSpartreco[key]['KS_p'][i]

    print('--> Saving KS plots')
    make_maxpKS_efficiency_plot(fsKS_dict , savepath=savepath,plottype='KS')
    make_nKS_plot(fsKS_dict, savepath=savepath,plottype='KS')

    print('--> Saving KS dict')
    data_savepath = 'outputs/prelim_cuts_full_data/flavtag_dataframes/selected_kaons'
    with open(os.path.join(data_savepath,'flavtag_KS.pkl'), 'wb') as f:
        pickle.dump(fsKS_dict, f)
