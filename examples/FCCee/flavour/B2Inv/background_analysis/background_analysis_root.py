import ROOT
import sys
from collections import Counter
import os
import glob
import numpy as np
import csv
import pdg
import re
api = pdg.connect()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basic_functions import vars_fromyaml, check_inputpath, set_outputpath, chunk_list
from df_makers.data_to_pickle_function import remove_dot_etc
import config as cfg



def pdg_name(pid):
    if pid in (None, 0, -999):
        return str(pid)
    try:
        return api.get_particle_by_mcid(int(pid)).name
    except Exception:
        return str(pid)

def safe_get(event, branch, idx):
    """Return event.<branch>[idx] or None if invalid/out-of-range."""
    # get attribute
    try:
        arr = getattr(event, branch)
    except Exception:
        return None
    # scalar branch
    if not hasattr(arr, "__len__"):
        return arr
    if idx is None:
        return None
    try:
        if idx < 0 or idx >= len(arr):
            return None
    except Exception:
        return None
    try:
        return arr[idx]
    except Exception:
        return None


def flatten(xss):
    #flatten arbitrarily nested list
    flat_list = []
    for item in xss:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def map_nested(func, data):
    """Recursively apply func to every non-list element in arbitrarily nested lists."""
    if isinstance(data, (list, tuple)):
        return [map_nested(func, x) for x in data]
    else:
        return func(data)


def not_empty(x):
    """Return True if there's any non-empty element in nested list/tuple."""
    if isinstance(x, (list, tuple)):
        return any(not_empty(i) for i in x)
    else:
        return x is not None



def calc_cos_theta(px,py,pz,p_tot,utx,uty,utz):
    cosTheta = (px*utx+py*uty+pz*utz)/p_tot
    return cosTheta


def build_decay_tree_Dgr2(event, sample_name, have_evt_ID=False):

    #extract type of event from Z
    qq = re.match(r"[A-Za-z]+", sample_name.split("Z", 1)[1]).group()

    if len(qq) ==2:
        q1 = qq[0]
        q2 = qq[1]

    else:
        raise ValueError("Should only have two quarks from Z")


    q1_number = cfg.quark_dictionary[q1]
    q2_number = cfg.quark_dictionary[q2]


    decay_chain = []
    decay_chain_hemis_sgn = []
    hemisEmin_particles =[]
    hemisEmax_particles=[]
    hemisEmin_particles_fs=[]
    particles = len(event.MC_PDG)

    if have_evt_ID==True:
        evt_ID = safe_get(event, "evt_id", 0)
    else:
        evt_ID = -999
        

    for p in range(particles):

      ID = safe_get(event, "MC_PDG", p)
      d1 = safe_get(event, "MC_D1", p) 
      d2 = safe_get(event, "MC_D2", p) 

      px = safe_get(event, "MC_px", p)
      py = safe_get(event, "MC_py", p)
      pz = safe_get(event, "MC_pz", p)
      p_tot = safe_get(event, "MC_p", p)

      unitThrust_x = safe_get(event, "EVT_unitThrust_x", 0)#event.EVT_unitThrust_x
      unitThrust_y = safe_get(event, "EVT_unitThrust_y", 0)#event.EVT_unitThrust_y
      unitThrust_z = safe_get(event, "EVT_unitThrust_z", 0)#event.EVT_unitThrust_z

      cosTheta = calc_cos_theta(px,py,pz,p_tot,unitThrust_x,unitThrust_y,unitThrust_z)

      if cosTheta>0: #ie. in hemisEmin

         hemisEmin_particles.append(p)

         if abs(ID)==5: # start with b quark
            #print(f"b or b bar in hemisEmin index: {p}")
            indx = p
            D1_index = safe_get(event, "MC_D1", indx)
            D2_index = safe_get(event, "MC_D2", indx)
            D1_ID = safe_get(event, "MC_PDG", D1_index) if D1_index is not None else -999
            D2_ID = safe_get(event, "MC_PDG", D2_index) if D2_index is not None else -999

            decay = [[ID]]  
            hemis_sgn = [[np.sign(cosTheta)]] #+1 for Emin, -1 gfor Emax                 
            
            # iterate until find b quark that actually decayed
            while abs(D1_ID)==5:
               indx = D1_index
               ID = D1_ID


               px = safe_get(event, "MC_px", indx) #event.MC_px[indx]
               py = safe_get(event, "MC_py", indx)#event.MC_py[indx]
               pz = safe_get(event, "MC_pz", indx)#event.MC_pz[indx]
               p_tot = safe_get(event, "MC_p", indx)#event.MC_p[indx]

               D1_index = safe_get(event, "MC_D1", indx)
               D2_index = safe_get(event, "MC_D2", indx)

               D1_ID = safe_get(event, "MC_PDG", D1_index) if D1_index is not None else -999
               D2_ID = safe_get(event, "MC_PDG", D2_index) if D2_index is not None else -999
            
            #start with b  hadronidsation
            daughter_indices =  [indx]
            decay.append([ID])
            m_costheta = calc_cos_theta(px,py,pz,p_tot,unitThrust_x,unitThrust_y,unitThrust_z)
            hemis_sgn.append([np.sign(m_costheta)]) #+1 for Emin, -1 gfor Emax

            #additionally see what is given b quark parent
            b_hadprod_IDs = []
            b_hadprod_indices = []
            b_hadprod_sgns = []

            for k in range(particles):
                if safe_get(event, "MC_M1",k) == indx or safe_get(event, "MC_M2",k) == indx: 
                    b_hadprod_indices.append(k)  
                    b_hadprod_IDs.append(event.MC_PDG[k])
                    b_hadprod_px = safe_get(event, "MC_px",k)#event.MC_px[k]
                    b_hadprod_py = safe_get(event, "MC_py",k)#event.MC_py[k]
                    b_hadprod_pz = safe_get(event, "MC_pz",k)#event.MC_pz[k]
                    b_hadprod_p_tot = safe_get(event, "MC_p",k)#event.MC_p[k]
                    b_hadprod_costheta = calc_cos_theta(b_hadprod_px,b_hadprod_py,b_hadprod_pz,b_hadprod_p_tot,unitThrust_x,unitThrust_y,unitThrust_z)
                    b_hadprod_sgns.append(np.sign(b_hadprod_costheta))


            #find daughter properties - use to study B meson decay chain
            if D1_index>=0:
                D1_px = safe_get(event, "MC_px",D1_index)#event.MC_px[D1_index]
                D1_py = safe_get(event, "MC_py",D1_index)#event.MC_py[D1_index]
                D1_pz = safe_get(event, "MC_pz",D1_index)#event.MC_pz[D1_index]
                D1_p_tot = safe_get(event, "MC_p",D1_index)#event.MC_p[D1_index]
                D1_costheta = calc_cos_theta(D1_px,D1_py,D1_pz,D1_p_tot,unitThrust_x,unitThrust_y,unitThrust_z)
            else:
                D1_costheta = 0

            if D2_index>=0:
                D2_px = safe_get(event, "MC_px",D2_index)#event.MC_px[D2_index]
                D2_py = safe_get(event, "MC_py",D2_index)#event.MC_py[D2_index]
                D2_pz = safe_get(event, "MC_pz",D2_index)#event.MC_pz[D2_index]
                D2_p_tot = safe_get(event, "MC_p",D2_index)#event.MC_p[D2_index]
                D2_costheta = calc_cos_theta(D2_px,D2_py,D2_pz,D2_p_tot,unitThrust_x,unitThrust_y,unitThrust_z)
            else:
                D2_costheta = 0

            
            daughter_IDs = [D1_ID,D2_ID]
            daughter_indices = [D1_index,D2_index]
            daughter_hemis_sgns = [np.sign(D1_costheta), np.sign(D2_costheta)]
            decay.append(daughter_IDs)
            hemis_sgn.append(daughter_hemis_sgns)


            while not_empty(flatten(daughter_indices)): 
                  
               gd_IDs = []
               gd_indices = []
               gd_hemis_sgns = []

               for d in dict.fromkeys(flatten(daughter_indices)): # ie. only does for unique daighters
                  if d>0:
                     d_D_IDs = []
                     d_D_indices = []
                     d_D_hemis_sgns = []
                     for k in range(particles):
                        if safe_get(event, "MC_M1",k) == d or safe_get(event, "MC_M2",k) == d:  
                           d_D_indices.append(k)  
                           d_D_IDs.append(event.MC_PDG[k])
                           d_D_px = safe_get(event,"MC_px",k)#event.MC_px[k]
                           d_D_py = safe_get(event,"MC_py",k)#event.MC_py[k]
                           d_D_pz = safe_get(event,"MC_pz",k)#event.MC_pz[k]
                           d_D_p_tot = safe_get(event,"MC_p",k)#event.MC_p[k]
                           d_D_costheta = calc_cos_theta(d_D_px,d_D_py,d_D_pz,d_D_p_tot,unitThrust_x,unitThrust_y,unitThrust_z)
                           d_D_hemis_sgns.append(np.sign(d_D_costheta))

                  gd_IDs.append(d_D_IDs)
                  gd_indices.append(d_D_indices)
                  gd_hemis_sgns.append(d_D_hemis_sgns)

               daughter_IDs = gd_IDs
               daughter_indices = gd_indices
               daughter_hemis_sgns = gd_hemis_sgns

               if not_empty(gd_IDs):
                  decay.append(gd_IDs)
                  hemis_sgn.append(daughter_hemis_sgns)

            decay_chain.extend(decay)
            decay_chain_hemis_sgn.extend(hemis_sgn)
            break

        
    # Convert PDG codes to particle names
    named_decay =  map_nested(pdg_name, decay_chain)
    #print(f"SS B decay chain: {named_decay}")
    #print(f"hemis sgn for decay chain:{decay_chain_hemis_sgn}")

    #print(f"SS b hadronisation prods: {map_nested(pdg_name, b_hadprod_IDs)}")
    #print(f"hemis sgn for SS b hadronisation prods: {b_hadprod_sgns}")


    #compare with reco info
    SS_fs = []
    OS_fs = []
    for k in range(len(event.Rec_in_hemisEmin)):
            particle_name = pdg_name(event.Rec_true_PDG[k])
            if event.Rec_in_hemisEmin[k] ==1:
                SS_fs.append(particle_name)
            else:
                OS_fs.append(particle_name)

    #print(f"SS reco final states:{SS_fs}")
    #print(f"OS reco final states:{OS_fs}")

    fs_particles = len(event.MCfinal_PDG)
    fs_SS_particles = []
    fs_OS_particles = []
    for fs_p in range(fs_particles):

        ID = event.MCfinal_PDG[fs_p]

        px = event.MCfinal_px[fs_p]
        py = event.MCfinal_py[fs_p]
        pz = event.MCfinal_pz[fs_p]
        p_tot = event.MCfinal_p[fs_p]

        unitThrust_x = event.EVT_unitThrust_x
        unitThrust_y = event.EVT_unitThrust_y
        unitThrust_z = event.EVT_unitThrust_z

        cosTheta = (px*unitThrust_x+py*unitThrust_y+pz*unitThrust_z)/p_tot

        if cosTheta>0: #ie. in hemisEmin
            particle = api.get_particle_by_mcid(ID)
            fs_SS_particles.append(particle.name)
        else:
            particle = api.get_particle_by_mcid(ID)
            fs_OS_particles.append(particle.name)

    
    #print(f"SS MC final states:{fs_SS_particles}")


    # Count multiplicities
    reco_counts = Counter(SS_fs)
    mc_counts = Counter(fs_SS_particles)

    # Missed particles (in MC but not in reco)
    missed_counts = mc_counts - reco_counts
    missed_particles = list(missed_counts.elements())

    #print(f"Missed particles in reco: {missed_particles}")

    # find reason for passing veto (based on four criteria)
    
    #1 tau decay to muon not on SS
    if not any(str(particle) in ['mu+', 'mu-'] for particle in flatten(named_decay)): #need str(particle) so that it doesn't complain for -99 float entreis
        reason = '1. No mu in SS decay chain'

    #2 muon missed (ie. in missing oarticle list - in true MC but not reco)
    elif any(str(particle) in ['mu+', 'mu-'] for particle in missed_particles):
        reason = "2. mu missed by reco"

    #3 muon reco on wrong side - ie. in OS_reco_fs and not SS_reco_fs
    elif any(str(particle) in ['mu+', 'mu-']  for particle in OS_fs if particle not in SS_fs):
            reason = "3. mu reconstructed on OS"
            mu_indices = [i for i, p in enumerate(flatten(named_decay)) if str(p) in ("mu+", "mu-")]

            if flatten(decay_chain_hemis_sgn)[mu_indices[0]] ==-1:
                reason += " AND true mu has OS mometum pointing"

    #4 muon decays in flight
    elif not any(str(particle) in ['mu+', 'mu-'] for particle in named_decay[-1]):
        reason = "4. mu decays in flight"
    
    else: 
        reason = "unknown"
        #print(f"SS B decay chain: {named_decay}")
    
    #print(reason)

    if 'b_hadprod_IDs' in locals():

        # Check if two B mesons have same parent b quark.... I dont feel like this should be possible but there seem to be quite a few....
        B_count = sum(1 for x in flatten(map_nested(pdg_name, b_hadprod_IDs)) if isinstance(x, str) and 'B' in x)
        if B_count > 1:
            #print("Funky Hadronisation - multiple SS B mesons")
            funky_had = True
        elif not map_nested(pdg_name, b_hadprod_IDs):
            #print("Funky Hadronisation - SS empty")
            funky_had = True
        else:
            funky_had = False


        #if named_decay ==[]:
            #print(f"MCZ_pz:{event.MCZ_pz}")
            #print(f"hemisEmin_e:{event.EVT_hemisEmin_e}")
            #print(f"SS reco final states:{SS_fs}")
            #print(f"OS reco final states:{OS_fs}")
            #print(f"SS MC final states:{fs_SS_particles}")

    else:
        b_hadprod_IDs = []
        b_hadprod_sgns = []
        funky_had = True


                
    return {
        "evt_id":evt_ID,
        "decay_chain": decay_chain,
        "decay_chain_hemis_sgn":decay_chain_hemis_sgn,
        "SS_b_had_prods": map_nested(pdg_name, b_hadprod_IDs),
        "SS_b_had_prods_hemis_sgn": b_hadprod_sgns,
        "funky_had": funky_had,
        "named_decay": named_decay,
        "SS_reco_final_states": SS_fs,
        "OS_reco_final_states": OS_fs,
        "SS_MC_final_states": fs_SS_particles,
        "OS_MC_final_states": fs_OS_particles,
        "missed_particles": missed_particles,
        "reason_passed_veto":reason
    }



if __name__ == "__main__":

    samples = cfg.sample_allocations["hadronic_background"]#["p8_ee_Zbb_ecm91"]#cfg.sample_allocations["Bc2lnu_background"] #["p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu"]##["p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTAUHADNU", "p8_ee_Zbb_ecm91_EvtGen_Bu2MuNu"] #["p8_ee_Zbb_ecm91_EvtGen_Bu2TauNuTau2MuNuNu"]#cfg.sample_allocations["Bu2lnu_background"]
    runmode = "process_with_MC_full_prelim" #"Bc2lnu_background_no_lepton_veto" #"Bu2lnu_background_no_lepton_veto"##"Bu2lnu_background_no_lepton_veto"
    bdtcut=0.99965

    inputpath_base  = check_inputpath(cfg.fccana_opts["outputDir"][runmode])
    inputpath   = check_inputpath(os.path.join(inputpath_base,"root_bdtscores"))
    bdtcut_name = "bdtlh_nocut"
    inputpath = check_inputpath(os.path.join(inputpath, bdtcut_name))

    for decay in samples:
        #load combined files!
        print(decay)
        decay_inputpath = check_inputpath(os.path.join(inputpath,f'{decay}'))
        filepath = glob.glob(os.path.join(decay_inputpath, f'combined_chunks_with_bdtcut{remove_dot_etc(str(bdtcut))}.root'))[0]

        #print(filepath)
        file = ROOT.TFile.Open(filepath)
        tree = file.Get("events")   # events is TTree name

        n_events = tree.GetEntries()
        print(n_events)


        results = []
        t = 0


        for i, event in enumerate(tree):
            #Any other cuts can be added here eg. lepton veto
            #if event.EVT_hemisEmin_nLept != 0:
            #    continue  # skip events with leptons

       
                info = build_decay_tree_Dgr2(event, have_evt_ID=True)
                #print("-----------------------------------------------------------") 
                
                results.append({
                    "Event": i,
                    "evt_id": info["evt_id"],
                    "DecayChain": info["named_decay"],
                    "DecayChain_hemis_sgn":info["decay_chain_hemis_sgn"],
                    "SS_HadronisationProducts": info["SS_b_had_prods"],
                    "SS_HadronisationProductshemis_sgn": info["SS_b_had_prods_hemis_sgn"],
                    "FunkyHaronisation_flag":info["funky_had"],
                    "SS_Reco_FS": info["SS_reco_final_states"],
                    "OS_Reco_FS": info["OS_reco_final_states"],
                    "SS_MC_FS": info["SS_MC_final_states"],
                    "OS_MC_FS": info["OS_MC_final_states"],
                    "SS_MissedParticles": info["missed_particles"],
                    "Reason passed mu veto":info["reason_passed_veto"]
                })
            
                
                t += 1
                #if t >= 15:
                #    break
                if t%100==0:
                    print(f"--> # Processed events; {t}")

        # Save all to CSV
        with open(os.path.join(decay_inputpath,f"background_analysis_{decay}_bdtcut{remove_dot_etc(str(bdtcut))}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"CSV file 'background_analysis_{decay}_bdtcut{remove_dot_etc(str(bdtcut))}.csv' created successfully!")
    



