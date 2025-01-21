########## INSTRUCTIONS ############
#
# Use this file to process B -> inv tuples
# Runs in a few different configurations
#   -no_selection:
#       Processes small test file with no cuts for bb bkg and signal
#
#   -prelim_cuts:
#       Produces ntuples to train 1st stage BDTs (BDTh and BDTl)
#       Includes all variables that I believe might be helpful for selection 
#       Applies preliminary cuts:
#               -"EVT_hasPV==1"                 #EVT must have a PV
#               - "EVT_hemisEmin_e < 40"       # Energy on the signal side must be < 40 GeV
#               - "EVT_hemisEmin_nCharged > 0"  # Signal side must have at least one charged reco particle
#               - "EVT_hemisEmin_nLept == 0"    # Remove events with a reconstructed lepton on the signal side -- removes a lot of semileptonic decays
#       
#
#BDTh - single hadronic BDT, used to separate signal from all hadronic bkgs in one go
#BDTl - BDT to discriminate against light hadronic bkgs (u,d,s)
#BDTmE - BDT to look for missing energy events in events that pass BDTl
#
####################################

import os
import sys

# Config and yaml file must be in this directory by default
# Absolute path must be supplied for the script to work in batch mode
configPath = '/r02/lhcb/ejnw2/fcc_2025/FCCAnalyses/examples/FCCee/flavour/B2Inv/'
sys.path.append(os.path.abspath(configPath))


import ROOT
from yaml import safe_load
from math import sqrt

import config as cfg

#Mandatory: List of processes
processList = cfg.processList[ cfg.run_mode ]

#Mandatory: Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics
prodTag = cfg.fccana_opts['prodTag']

#Optional: output directory, default is local running directory
outputDir = cfg.fccana_opts['outputDir'][cfg.run_mode]

#Optional: analysisName, default is ""
analysisName = cfg.fccana_opts['analysisName']

#Optional: ncpus, default is 4
nCPUS = cfg.fccana_opts['nCPUS']

#Optional running on HTCondor, default is False
runBatch = cfg.fccana_opts['runBatch']

#Optional test file
testFile = cfg.fccana_opts['testFile']['Bs']

print("----> INFO: Using config.py file from:")
print(f"{15*' '}{os.path.abspath(configPath)}")
print("----> INFO: Using branch names from:")
print(f"{15*' '}{cfg.fccana_opts['yamlPath']}")


class RDFanalysis():

    #__________________________________________________________
    def analysers(df):
        
        #BSC for vertexing
        #bsc = [ 6, 25e-3, 400 ]
        bsc = cfg.BSC_opts['winter2023'] # list of sigmax,sigmay,sigmaz


        df2 = (
            df
            #############################################
            ##          Aliases for # in python        ##
            #############################################
            .Alias("MCRecoAssociationsRec", "MCRecoAssociations#0.index")  # points to ReconstructedParticles
            .Alias("MCRecoAssociationsGen", "MCRecoAssociations#1.index")  # points to Particle
            .Alias("ParticleParents",       "Particle#0.index")            # gen particle parents
            .Alias("ParticleChildren",      "Particle#1.index")            # gen particle children
            

            ##################################################
            ## MC variavles to help with understanding event##
            ##################################################
            # Pythia8 generatorStatus
            # 21 - incoming particles of hardest process (e+ e- beams)
            # 22 - intermediate particles of hardest process (Z)
            # 23 - outgoing particles of hardest process (quark pair produced from Z)
            #  1 - final-state particles
            .Define("MC_ee",          "MCParticle::sel_genStatus(21)(Particle)")   # INTERMEDIATE
            .Define("MC_Z",           "MCParticle::sel_genStatus(22)(Particle)")   # INTERMEDIATE
            .Define("MC_qq",          "MCParticle::sel_genStatus(23)(Particle)")   # INTERMEDIATE
            .Define("MCem_p",         "(MCParticle::get_p(MC_ee)).at(0)")
            .Define("MCep_p",         "(MCParticle::get_p(MC_ee)).at(1)")
            .Define("MCZ_p",          "(MCParticle::get_p(MC_Z)).at(0)")
            .Define("MCq1_p",         "(MCParticle::get_p(MC_qq)).at(0)")
            .Define("MCq1_pz",        "(MCParticle::get_pz(MC_qq)).at(0)")
            
            #############################################
            ##         Perform vertex fitting          ##
            #############################################
            # Get collection of tracks consistent with a PV (i.e. not downstream Ks, Lb etc. tracks)
            # using the get_PrimaryTracks() method with a beam spot constraint under the following parameters
            # bsc_sigma(x,y,z) = (6, 25e-3, 400)
            # bsc_(x,y,z) = (0,0,0)
            
            # First the PV - select tracks reconstructed as primaries
            .Define("Rec_PrimaryTracks",       f"VertexFitterSimple::get_PrimaryTracks( EFlowTrack_1, true, {bsc[0]}, {bsc[1]}, {bsc[2]}, 0., 0., 0.)")
            .Define("Rec_n_primary_tracks",     "ReconstructedParticle2Track::getTK_n( Rec_PrimaryTracks )")
            # Then fit the PV using these tracks
            .Define("Rec_PrimaryVertexObject", f"VertexFitterSimple::VertexFitter_Tk( 1, Rec_PrimaryTracks, true, {bsc[0]}, {bsc[1]}, {bsc[2]} )")
            .Define("Rec_PrimaryVertex",        "Rec_PrimaryVertexObject.vertex")
            # Get secondary tracks
            .Define("Rec_SecondaryTracks",      "VertexFitterSimple::get_NonPrimaryTracks( EFlowTrack_1, Rec_PrimaryTracks )")
            .Define("Rec_n_secondary_tracks",   "ReconstructedParticle2Track::getTK_n( Rec_SecondaryTracks )")
            # We don't actually do anything with the secondary tracks

            # get all MC vertices
            .Define("MC_VertexObject",          "myUtils::get_MCVertexObject(Particle, ParticleParents)")
            # use this to seed the Rec vertexing
            .Define("Rec_VertexObject",        f"myUtils::get_VertexObject(MC_VertexObject, ReconstructedParticles, EFlowTrack_1, MCRecoAssociationsRec, MCRecoAssociationsGen, {bsc[0]}, {bsc[1]}, {bsc[2]})")

            # add the PID hypothesis info to the RecParticles (based on MC truth - ie assume perfect PID here)
            .Define("RecoParticlesPID",          "myUtils::PID(ReconstructedParticles, MCRecoAssociationsRec, MCRecoAssociationsGen, Particle)")
            # now update reco momentum based on the rec vertex position
            .Define("RecoParticlesPIDAtVertex",  "myUtils::get_RP_atVertex(RecoParticlesPID, Rec_VertexObject)")

            #############################################
            ##         Define vertex variables         ##
            #############################################
            # PV
            .Define("Rec_PV_ntracks",  "float(Rec_PrimaryTracks.size())")
            .Define("Rec_PV_x",        "Rec_PrimaryVertex.position.x")
            .Define("Rec_PV_y",        "Rec_PrimaryVertex.position.y")
            .Define("Rec_PV_z",        "Rec_PrimaryVertex.position.z")
            # All Rec Vertices
            .Define("Rec_vtx_n",               "float(Rec_VertexObject.size())")
            .Define("Rec_vtx_indRP",           "myUtils::get_Vertex_ind(Rec_VertexObject)")
            .Define("Rec_vtx_chi2",            "myUtils::get_Vertex_chi2(Rec_VertexObject)")
            .Define("Rec_vtx_isPV",            "myUtils::get_Vertex_isPV(Rec_VertexObject)")
            .Define("Rec_vtx_ntracks",         "myUtils::get_Vertex_ntracks(Rec_VertexObject)")
            .Define("Rec_vtx_m",               "myUtils::get_Vertex_mass(Rec_VertexObject, RecoParticlesPIDAtVertex)")
            .Define("Rec_vtx_x",               "myUtils::get_Vertex_x(Rec_VertexObject)")
            .Define("Rec_vtx_y",               "myUtils::get_Vertex_y(Rec_VertexObject)")
            .Define("Rec_vtx_z",               "myUtils::get_Vertex_z(Rec_VertexObject)")
            .Define("Rec_vtx_xerr",            "myUtils::get_Vertex_xErr(Rec_VertexObject)")
            .Define("Rec_vtx_yerr",            "myUtils::get_Vertex_yErr(Rec_VertexObject)")
            .Define("Rec_vtx_zerr",            "myUtils::get_Vertex_zErr(Rec_VertexObject)")

            #############################################
            ##       Define reco particle variables    ##
            #############################################
            .Define("Rec_n",         "ReconstructedParticle::get_n(RecoParticlesPIDAtVertex)")
            .Define("Rec_type",      "ReconstructedParticle::get_type(RecoParticlesPIDAtVertex)")
            .Define("Rec_indvtx",    "myUtils::get_Vertex_fromRP(RecoParticlesPIDAtVertex, Rec_VertexObject)")
            .Define("Rec_customid",  "ReconstructedParticle::get_customid(RecoParticlesPIDAtVertex)")
            .Define("Rec_e",         "ReconstructedParticle::get_e(RecoParticlesPIDAtVertex)")
            .Define("Rec_m",         "ReconstructedParticle::get_mass(RecoParticlesPIDAtVertex)")
            .Define("Rec_q",         "ReconstructedParticle::get_charge(RecoParticlesPIDAtVertex)")
            .Define("Rec_p",         "ReconstructedParticle::get_p(RecoParticlesPIDAtVertex)")
            .Define("Rec_pt",        "ReconstructedParticle::get_pt(RecoParticlesPIDAtVertex)")
            .Define("Rec_px",        "ReconstructedParticle::get_px(RecoParticlesPIDAtVertex)")
            .Define("Rec_py",        "ReconstructedParticle::get_py(RecoParticlesPIDAtVertex)")
            .Define("Rec_pz",        "ReconstructedParticle::get_pz(RecoParticlesPIDAtVertex)")
            .Define("Rec_eta",       "ReconstructedParticle::get_eta(RecoParticlesPIDAtVertex)")
            .Define("Rec_phi",       "ReconstructedParticle::get_phi(RecoParticlesPIDAtVertex)")
            
            # Do MC association of reco particle to true MC particle
            .Define("MC_fromRP",           "myUtils::get_MCObject_fromRP(MCRecoAssociationsRec, MCRecoAssociationsGen, RecoParticlesPIDAtVertex, Particle)")
            .Define("Rec_true_PDG",        "MCParticle::get_pdg(MC_fromRP)")
            .Define("Rec_true_e",          "MCParticle::get_e(MC_fromRP)")
            .Define("Rec_true_m",          "MCParticle::get_mass(MC_fromRP)")
            .Define("Rec_true_q",          "MCParticle::get_charge(MC_fromRP)")
            .Define("Rec_true_p",          "MCParticle::get_p(MC_fromRP)")
            .Define("Rec_true_pt",         "MCParticle::get_pt(MC_fromRP)")
            .Define("Rec_true_px",         "MCParticle::get_px(MC_fromRP)")
            .Define("Rec_true_py",         "MCParticle::get_py(MC_fromRP)")
            .Define("Rec_true_pz",         "MCParticle::get_pz(MC_fromRP)")
            .Define("Rec_true_eta",        "MCParticle::get_eta(MC_fromRP)")
            .Define("Rec_true_phi",        "MCParticle::get_phi(MC_fromRP)")
            .Define("Rec_true_orivtx_x",   "MCParticle::get_vertex_x(MC_fromRP)")
            .Define("Rec_true_orivtx_y",   "MCParticle::get_vertex_y(MC_fromRP)")
            .Define("Rec_true_orivtx_z",   "MCParticle::get_vertex_z(MC_fromRP)")

            # RecoP true history (mothers and gmothers)
            .Define("True_ParentInfo",     "myUtils::get_MCParentandGParent_fromRP(MCRecoAssociationsRec, MCRecoAssociationsGen, ParticleParents, RecoParticlesPIDAtVertex, Particle)")   # INTERMEDIATE
            .Define("Rec_true_M1",         "True_ParentInfo.at(0)")
            .Define("Rec_true_M2",         "True_ParentInfo.at(1)")
            .Define("Rec_true_M1ofM1",     "True_ParentInfo.at(2)")
            .Define("Rec_true_M2ofM1",     "True_ParentInfo.at(3)")
            .Define("Rec_true_M1ofM2",     "True_ParentInfo.at(4)")
            .Define("Rec_true_M2ofM2",     "True_ParentInfo.at(5)")
            
            # Store total number of tracks
            .Define("Rec_track_n",       "float(ReconstructedParticle2Track::getTK_n(EFlowTrack_1))")

            #############################################
            ##      Construct the Thrust Axis          ##
            #############################################
            .Define("EVT_ThrustInfoNoPointing",     'Algorithms::minimize_thrust("Minuit2","Migrad")(Rec_px, Rec_py, Rec_pz)') 
            .Define("EVT_ThrustCosThetaNoPointing", "Algorithms::getAxisCosTheta(EVT_ThrustInfoNoPointing, Rec_px, Rec_py, Rec_pz)")
            .Define("EVT_ThrustInfo",               "Algorithms::getThrustPointing(1.)(EVT_ThrustCosThetaNoPointing, Rec_e, EVT_ThrustInfoNoPointing)")
            .Define("Rec_thrustCosTheta",           "Algorithms::getAxisCosTheta(EVT_ThrustInfo, Rec_px, Rec_py, Rec_pz)")
            .Define("Rec_in_hemisEmin",             "myUtils::get_RP_inHemis(1)(Rec_thrustCosTheta)")
            .Define("Rec_in_hemisEmax",             "myUtils::get_RP_inHemis(0)(Rec_thrustCosTheta)")

            .Define("EVT_ThrustInfoMax_N",     "Algorithms::getAxisN(0)(Rec_thrustCosTheta, Rec_q)")
            .Define("EVT_ThrustInfoMin_N",     "Algorithms::getAxisN(1)(Rec_thrustCosTheta, Rec_q)")
            .Define("EVT_ThrustInfoMax_E",     "Algorithms::getAxisEnergy(0)(Rec_thrustCosTheta, Rec_q, Rec_e)")
            .Define("EVT_ThrustInfoMin_E",     "Algorithms::getAxisEnergy(1)(Rec_thrustCosTheta, Rec_q, Rec_e)")

            .Define("EVT_hemisEmin_e",         "EVT_ThrustInfoMin_E.at(0)")
            .Define("EVT_hemisEmin_eCharged",  "EVT_ThrustInfoMin_E.at(1)")
            .Define("EVT_hemisEmin_eNeutral",  "EVT_ThrustInfoMin_E.at(2)")
            .Define("EVT_hemisEmin_n",         "float(EVT_ThrustInfoMin_N.at(0))")
            .Define("EVT_hemisEmin_nCharged",  "float(EVT_ThrustInfoMin_N.at(1))")
            .Define("EVT_hemisEmin_nNeutral",  "float(EVT_ThrustInfoMin_N.at(2))")


            #############################################
            ##           Remaining Thrust Vars         ##
            #############################################
            .Define("EVT_Thrust_mag",          "EVT_ThrustInfo.at(0)")
            .Define("EVT_Thrust_x",            "EVT_ThrustInfo.at(1)")
            .Define("EVT_Thrust_xerr",         "EVT_ThrustInfo.at(2)")
            .Define("EVT_Thrust_y",            "EVT_ThrustInfo.at(3)")
            .Define("EVT_Thrust_yerr",         "EVT_ThrustInfo.at(4)")
            .Define("EVT_Thrust_z",            "EVT_ThrustInfo.at(5)")
            .Define("EVT_Thrust_zerr",         "EVT_ThrustInfo.at(6)")

            .Define("EVT_unitThrust_x",            "myUtils::norm_RVec_x(EVT_ThrustInfo.at(1),EVT_ThrustInfo.at(3),EVT_ThrustInfo.at(5))")
            .Define("EVT_unitThrust_y",            "myUtils::norm_RVec_x(EVT_ThrustInfo.at(3),EVT_ThrustInfo.at(1),EVT_ThrustInfo.at(5))")
            .Define("EVT_unitThrust_z",            "myUtils::norm_RVec_x(EVT_ThrustInfo.at(5),EVT_ThrustInfo.at(3),EVT_ThrustInfo.at(1))")

            .Define("EVT_hemisEmax_e",         "EVT_ThrustInfoMax_E.at(0)")
            .Define("EVT_hemisEmax_eCharged",  "EVT_ThrustInfoMax_E.at(1)")
            .Define("EVT_hemisEmax_eNeutral",  "EVT_ThrustInfoMax_E.at(2)")
            .Define("EVT_hemisEmax_n",         "float(EVT_ThrustInfoMax_N.at(0))")
            .Define("EVT_hemisEmax_nCharged",  "float(EVT_ThrustInfoMax_N.at(1))")
            .Define("EVT_hemisEmax_nNeutral",  "float(EVT_ThrustInfoMax_N.at(2))")

            .Define("EVT_e", "(EVT_hemisEmin_e)+(EVT_hemisEmax_e)")

            # Count secondary vertices in each hemisphere
            .Define("SecondaryVertexThrustAngle",  "myUtils::get_DVertex_thrusthemis_angle(Rec_VertexObject, RecoParticlesPIDAtVertex, EVT_ThrustInfo)")
            .Define("EVT_hemisEmin_nDV",           "float(myUtils::get_Npos(SecondaryVertexThrustAngle))")
            .Define("EVT_hemisEmax_nDV",           "float(myUtils::get_Nneg(SecondaryVertexThrustAngle))")

            # Hemisphere energy differences
            .Define("EVT_Thrust_deltaE",            "(EVT_hemisEmax_e) - (EVT_hemisEmin_e)")
            .Define("EVT_hemisEmin_Emiss",          f"{0.5*cfg.mass_Z} - EVT_hemisEmin_e")
            .Define("EVT_hemisEmax_Emiss",          f"{0.5*cfg.mass_Z} - EVT_hemisEmax_e")

            # Gather info on charged lepons, kaons and pions in each hemisphere
            .Define("EVT_EminPartInfo",    "myUtils::get_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmin)")
            .Define("EVT_EmaxPartInfo",    "myUtils::get_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmax)")
            .Define("EVT_hemisEmin_nLept", "(EVT_EminPartInfo.at(0)).num")
            
            #################################################
            ## Ella extra variables to add for S2 Training ##
            #################################################
            .Define("PV_Rec_vtx_m_vec", "myUtils::filter_vtx_variable_onisPV(Rec_vtx_isPV, Rec_vtx_m)") #intermediate
            .Define("PV_Rec_vtx_m","PV_Rec_vtx_m_vec.at(0)") 
            

            #Equivalent for different hemispheres
            # Vertex relations to thrust
            .Define("Rec_vtx_thrustCosTheta",  "myUtils::get_Vertex_thrusthemis_angle(Rec_VertexObject, RecoParticlesPIDAtVertex, EVT_ThrustInfo)")
            # Flag vertex in max or min hemisphere
            .Define("Rec_vtx_in_hemisEmin",    "myUtils::get_Vertex_thrusthemis(Rec_vtx_thrustCosTheta, 1)")
            .Define("Rec_vtx_in_hemisEmax",    "myUtils::get_Vertex_thrusthemis(Rec_vtx_thrustCosTheta, 0)")  # FLAG - NOT SAVED

            #Sum over vertices in given hemis that are not PV
            .Define("EVT_hemisEmin_sum_Rec_vtx_ntracks_exclPV_vec", "myUtils::sum_RVec_with2cond(1-(Rec_vtx_isPV), Rec_vtx_in_hemisEmin, Rec_vtx_ntracks)") #Intermediate
            .Define("EVT_hemisEmax_sum_Rec_vtx_ntracks_exclPV_vec", "myUtils::sum_RVec_with2cond(1-(Rec_vtx_isPV), Rec_vtx_in_hemisEmax, Rec_vtx_ntracks)") #Intermediate
            
            #Overall three sum_Rec_vtx_exclPV vars
            .Define("EVT_sum_Rec_vtx_ntracks_exclPV", "myUtils::sum_RVec_withcond(1-(Rec_vtx_isPV), Rec_vtx_ntracks)")
            .Define("EVT_hemisEmin_sum_Rec_vtx_ntracks_exclPV", "EVT_hemisEmin_sum_Rec_vtx_ntracks_exclPV_vec.at(0)") 
            .Define("EVT_hemisEmax_sum_Rec_vtx_ntracks_exclPV", "EVT_hemisEmax_sum_Rec_vtx_ntracks_exclPV_vec.at(0)") 
            

            #Sum all particle momenta over given hemisphere
            .Define("EVT_hemisEmin_sum_Rec_p",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmin,Rec_p)")
            .Define("EVT_hemisEmax_sum_Rec_p",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmax,Rec_p)")

            .Define("EVT_hemisEmin_sum_Rec_px",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmin,Rec_px)")
            .Define("EVT_hemisEmin_sum_Rec_py",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmin,Rec_py)")
            .Define("EVT_hemisEmin_sum_Rec_pz",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmin,Rec_pz)")
            .Define("EVT_hemisEmin_p",  "sqrt(EVT_hemisEmin_sum_Rec_px*EVT_hemisEmin_sum_Rec_px+EVT_hemisEmin_sum_Rec_py*EVT_hemisEmin_sum_Rec_py+EVT_hemisEmin_sum_Rec_pz*EVT_hemisEmin_sum_Rec_pz)")
            
            .Define("EVT_hemisEmax_sum_Rec_px",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmax,Rec_px)")
            .Define("EVT_hemisEmax_sum_Rec_py",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmax,Rec_py)")
            .Define("EVT_hemisEmax_sum_Rec_pz",  "myUtils::sum_RVec_withcond(Rec_in_hemisEmax,Rec_pz)")
            .Define("EVT_hemisEmax_p",  "sqrt(EVT_hemisEmax_sum_Rec_px*EVT_hemisEmax_sum_Rec_px+EVT_hemisEmax_sum_Rec_py*EVT_hemisEmax_sum_Rec_py+EVT_hemisEmax_sum_Rec_pz*EVT_hemisEmax_sum_Rec_pz)")
            
            .Define("EVT_sum_Rec_px",  "EVT_hemisEmin_sum_Rec_px+EVT_hemisEmax_sum_Rec_px")
            .Define("EVT_sum_Rec_py",  "EVT_hemisEmin_sum_Rec_py+EVT_hemisEmax_sum_Rec_py")
            .Define("EVT_sum_Rec_pz",  "EVT_hemisEmin_sum_Rec_pz+EVT_hemisEmax_sum_Rec_pz")
            .Define("EVT_p",  "sqrt(EVT_sum_Rec_px*EVT_sum_Rec_px+EVT_sum_Rec_py*EVT_sum_Rec_py+EVT_sum_Rec_pz*EVT_sum_Rec_pz)")

            

            #############################################
            ##           IP-like track vars            ##
            #############################################
            .Define("Rec_track_d0",      "ReconstructedParticle2Track::getRP2TRK_D0(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_normd0",  "ReconstructedParticle2Track::getRP2TRK_D0_sig(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_z0",      "ReconstructedParticle2Track::getRP2TRK_Z0(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_normz0",  "ReconstructedParticle2Track::getRP2TRK_Z0_sig(RecoParticlesPIDAtVertex, EFlowTrack_1)")


            .Define("Rec_track_absd0",      "myUtils::abs_RVec(Rec_track_d0)")
            .Define("Rec_track_absnormd0",  "myUtils::abs_RVec(Rec_track_normd0)")
            .Define("Rec_track_absz0",      "myUtils::abs_RVec(Rec_track_z0)")
            .Define("Rec_track_absnormz0",  "myUtils::abs_RVec(Rec_track_normz0)")


            # Reco track stats
            .Define("RecoP_inhemisEminAndCharged",   "myUtils::remove_Neutrals_fromTrackStats(Rec_in_hemisEmin, Rec_q)")        # FLAG - NOT SAVED
            .Define("RecoP_inhemisEmaxAndCharged",   "myUtils::remove_Neutrals_fromTrackStats(Rec_in_hemisEmax, Rec_q)")        # FLAG - NOT SAVED
            .Define("Rec_track_absd0StatsEmin",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEminAndCharged, Rec_track_absd0)")  # INTERMEDIATE
            .Define("Rec_track_absd0StatsEmax",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEmaxAndCharged, Rec_track_absd0)")  # INTERMEDIATE
            .Define("Rec_track_absnormd0StatsEmin",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEminAndCharged, Rec_track_absnormd0)")  # INTERMEDIATE
            .Define("Rec_track_absnormd0StatsEmax",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEmaxAndCharged, Rec_track_absnormd0)")  # INTERMEDIATE
            .Define("Rec_track_absz0StatsEmin",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEminAndCharged, Rec_track_absz0)")  # INTERMEDIATE
            .Define("Rec_track_absz0StatsEmax",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEmaxAndCharged, Rec_track_absz0)")  # INTERMEDIATE
            .Define("Rec_track_absnormz0StatsEmin",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEminAndCharged, Rec_track_absnormz0)")  # INTERMEDIATE
            .Define("Rec_track_absnormz0StatsEmax",         "myUtils::get_Stats_fromRVec(RecoP_inhemisEmaxAndCharged, Rec_track_absnormz0)")  # INTERMEDIATE


            .Define("Rec_track_absd0_min_hemisEmin",    "Rec_track_absd0StatsEmin.at(0)")
            .Define("Rec_track_absd0_max_hemisEmin",    "Rec_track_absd0StatsEmin.at(1)")
            .Define("Rec_track_absd0_ave_hemisEmin",    "Rec_track_absd0StatsEmin.at(2)")

            .Define("Rec_track_absd0_min_hemisEmax",    "Rec_track_absd0StatsEmax.at(0)")
            .Define("Rec_track_absd0_max_hemisEmax",    "Rec_track_absd0StatsEmax.at(1)")
            .Define("Rec_track_absd0_ave_hemisEmax",    "Rec_track_absd0StatsEmax.at(2)")
            
            .Define("Rec_track_absd0chi2_min_hemisEmin",    "Rec_track_absnormd0StatsEmin.at(0)")
            .Define("Rec_track_absd0chi2_max_hemisEmin",    "Rec_track_absnormd0StatsEmin.at(1)")
            .Define("Rec_track_absd0chi2_ave_hemisEmin",    "Rec_track_absnormd0StatsEmin.at(2)")

            .Define("Rec_track_absd0chi2_min_hemisEmax",    "Rec_track_absnormd0StatsEmax.at(0)")
            .Define("Rec_track_absd0chi2_max_hemisEmax",    "Rec_track_absnormd0StatsEmax.at(1)")
            .Define("Rec_track_absd0chi2_ave_hemisEmax",    "Rec_track_absnormd0StatsEmax.at(2)")

            .Define("Rec_track_absz0_min_hemisEmin",    "Rec_track_absz0StatsEmin.at(0)")
            .Define("Rec_track_absz0_max_hemisEmin",    "Rec_track_absz0StatsEmin.at(1)")
            .Define("Rec_track_absz0_ave_hemisEmin",    "Rec_track_absz0StatsEmin.at(2)")

            .Define("Rec_track_absz0_min_hemisEmax",    "Rec_track_absz0StatsEmax.at(0)")
            .Define("Rec_track_absz0_max_hemisEmax",    "Rec_track_absz0StatsEmax.at(1)")
            .Define("Rec_track_absz0_ave_hemisEmax",    "Rec_track_absz0StatsEmax.at(2)")
            
            .Define("Rec_track_absz0chi2_min_hemisEmin",    "Rec_track_absnormz0StatsEmin.at(0)")
            .Define("Rec_track_absz0chi2_max_hemisEmin",    "Rec_track_absnormz0StatsEmin.at(1)")
            .Define("Rec_track_absz0chi2_ave_hemisEmin",    "Rec_track_absnormz0StatsEmin.at(2)")

            .Define("Rec_track_absz0chi2_min_hemisEmax",    "Rec_track_absnormz0StatsEmax.at(0)")
            .Define("Rec_track_absz0chi2_max_hemisEmax",    "Rec_track_absnormz0StatsEmax.at(1)")
            .Define("Rec_track_absz0chi2_ave_hemisEmax",    "Rec_track_absnormz0StatsEmax.at(2)")

            #############################################
            ##     for max P charged RP vars           ##
            #############################################

            .Define("EVT_hemisEmin_maxpChargedRPInfo",    "myUtils::get_maxp_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmin)")  # INTERMEDIATE
            .Define("EVT_hemisEmax_maxpChargedRPInfo",    "myUtils::get_maxp_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmax)")  # INTERMEDIATE
            
            .Define("EVT_hemisEmin_maxpChargedRP_p",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).maxp")
            .Define("EVT_hemisEmin_maxpChargedRP_e",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).energy")
            .Define("EVT_hemisEmin_maxpChargedRP_PDG",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).PDG")
            .Define("EVT_hemisEmin_maxpChargedRP_q",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).charge")
            .Define("EVT_hemisEmin_maxpChargedRP_px",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).px")
            .Define("EVT_hemisEmin_maxpChargedRP_py",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).py")
            .Define("EVT_hemisEmin_maxpChargedRP_pz",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).pz")
            .Define("EVT_hemisEmin_maxpChargedRP_fromPV",             "(EVT_hemisEmin_maxpChargedRPInfo.at(0)).fromPV")
            
            .Define("EVT_hemisEmax_maxpChargedRP_p",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).maxp")
            .Define("EVT_hemisEmax_maxpChargedRP_e",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).energy")
            .Define("EVT_hemisEmax_maxpChargedRP_PDG",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).PDG")
            .Define("EVT_hemisEmax_maxpChargedRP_q",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).charge")
            .Define("EVT_hemisEmax_maxpChargedRP_px",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).px")
            .Define("EVT_hemisEmax_maxpChargedRP_py",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).py")
            .Define("EVT_hemisEmax_maxpChargedRP_pz",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).pz")
            .Define("EVT_hemisEmax_maxpChargedRP_fromPV",             "(EVT_hemisEmax_maxpChargedRPInfo.at(0)).fromPV")


            ##################################################################
            ##     Variables for position-based assignment of vtx to hemis  ##        
            ##################################################################

            #d2PV variable - want for BDT2 and vtx assignment
            .Define("Rec_vtx_d2PV_x",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 0)")
            .Define("Rec_vtx_d2PV_y",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 1)")
            .Define("Rec_vtx_d2PV_z",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 2)")
            
            #calculate costheta for thrust to d2pv vector - returns 0 if vertex is a PV - not minus signs infront of d2pv variabls as d2PV defined in source code as PV-SV [the vector we want is SV-PV]
            .Define("Rec_vtx_thrustCosTheta_d2PV",           "myUtils::getAxisCosTheta_withcond(EVT_ThrustInfo, (-Rec_vtx_d2PV_x), (-Rec_vtx_d2PV_y), (-Rec_vtx_d2PV_z),1-(Rec_vtx_isPV))")

            # Flag vertex in max or min hemisphere - use get_RP_inHemis as gives '-1' if costheta==0
            .Define("Rec_vtx_in_hemisEmin_d2PV",             "myUtils::get_RP_inHemis(1)(Rec_vtx_thrustCosTheta_d2PV)")
            .Define("Rec_vtx_in_hemisEmax_d2PV",             "myUtils::get_RP_inHemis(0)(Rec_vtx_thrustCosTheta_d2PV)")

            ###########################################################
            ## others of Ritwiks S1 variables                       ###
            ###########################################################

            ###########################
            ##  thrustcostheta stats ##
            ###########################

            .Define("Rec_thrustCosThetaEminStats", "myUtils::get_Stats_fromRVec(Rec_in_hemisEmin, Rec_thrustCosTheta)")                                              # INTERMEDIATE
            .Define("Rec_thrustCosThetaEmaxStats", "myUtils::get_Stats_fromRVec(Rec_in_hemisEmax, Rec_thrustCosTheta)")                                              # INTERMEDIATE

            .Define("Rec_thrustCosTheta_min_hemisEmin", "Rec_thrustCosThetaEminStats.at(0)")
            .Define("Rec_thrustCosTheta_max_hemisEmin", "Rec_thrustCosThetaEminStats.at(1)")
            .Define("Rec_thrustCosTheta_ave_hemisEmin", "Rec_thrustCosThetaEminStats.at(2)")
            .Define("Rec_thrustCosTheta_min_hemisEmax", "Rec_thrustCosThetaEmaxStats.at(0)")
            .Define("Rec_thrustCosTheta_max_hemisEmax", "Rec_thrustCosThetaEmaxStats.at(1)")
            .Define("Rec_thrustCosTheta_ave_hemisEmax", "Rec_thrustCosThetaEmaxStats.at(2)")

            #############################################
            ##        Remaining reco vertex vars       ##
            #############################################

            .Define("Rec_vtx_d2PV",            "myUtils::get_Vertex_d2PV(Rec_VertexObject,-1)")   # INTERMEDIATE
            .Define("Rec_vtx_d2PV_err",        "myUtils::get_Vertex_d2PVError(Rec_VertexObject,-1)")
            .Define("Rec_vtx_d2PV_xerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 0)")
            .Define("Rec_vtx_d2PV_yerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 1)")
            .Define("Rec_vtx_d2PV_zerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 2)")
            .Define("Rec_vtx_normd2PV",        "Rec_vtx_d2PV / Rec_vtx_d2PV_err")   # INTERMEDIATE
            .Define("Rec_vtx_normd2PV_x",      "Rec_vtx_d2PV_x / Rec_vtx_d2PV_xerr")
            .Define("Rec_vtx_normd2PV_y",      "Rec_vtx_d2PV_y / Rec_vtx_d2PV_yerr")
            .Define("Rec_vtx_normd2PV_z",      "Rec_vtx_d2PV_z / Rec_vtx_d2PV_zerr")

            # Reco vertex stats
            .Define("Rec_vtx_in_hemisEmin_andNotPV",   "myUtils::remove_PV_fromVertexStats(Rec_vtx_in_hemisEmin, Rec_VertexObject)")            # FLAG - NOT SAVED
            .Define("Rec_vtx_in_hemisEmax_andNotPV",   "myUtils::remove_PV_fromVertexStats(Rec_vtx_in_hemisEmax, Rec_VertexObject)")            # FLAG - NOT SAVED
            .Define("Rec_vtx_d2PV_signed",             "myUtils::get_VertexFeature_signed(Rec_in_hemisEmin, Rec_vtx_d2PV)")
            .Define("Rec_vtx_normd2PV_signed",         "myUtils::get_VertexFeature_signed(Rec_in_hemisEmin, Rec_vtx_normd2PV)")

            .Define("Rec_vtx_ntracksStatsEmin",               "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmin_andNotPV, Rec_vtx_ntracks)")  # INTERMEDIATE
            .Define("Rec_vtx_ntracksStatsEmax",               "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmax_andNotPV, Rec_vtx_ntracks)")  # INTERMEDIATE
            .Define("Rec_vtx_ntracks_max_hemisEmin",          "Rec_vtx_ntracksStatsEmin.at(1)")
            .Define("Rec_vtx_ntracks_max_hemisEmax",          "Rec_vtx_ntracksStatsEmax.at(1)")
      
            .Define("Rec_vtx_d2PVStatsEmin",                  "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmin_andNotPV, Rec_vtx_d2PV)")  # INTERMEDIATE
            .Define("Rec_vtx_d2PVStatsEmax",                  "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmax_andNotPV, Rec_vtx_d2PV)")  # INTERMEDIATE
            .Define("Rec_vtx_d2PV_min_hemisEmin",             "Rec_vtx_d2PVStatsEmin.at(0)")
            .Define("Rec_vtx_d2PV_max_hemisEmin",             "Rec_vtx_d2PVStatsEmin.at(1)")
            .Define("Rec_vtx_d2PV_ave_hemisEmin",             "Rec_vtx_d2PVStatsEmin.at(2)")
            .Define("Rec_vtx_d2PV_min_hemisEmax",             "Rec_vtx_d2PVStatsEmax.at(0)")
            .Define("Rec_vtx_d2PV_max_hemisEmax",             "Rec_vtx_d2PVStatsEmax.at(1)")
            .Define("Rec_vtx_d2PV_ave_hemisEmax",             "Rec_vtx_d2PVStatsEmax.at(2)")

            .Define("Rec_vtx_thrustCosThetaStatsEmin",        "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmin_andNotPV, Rec_vtx_thrustCosTheta)")  # INTERMEDIATE
            .Define("Rec_vtx_thrustCosThetaStatsEmax",        "myUtils::get_Stats_fromRVec(Rec_vtx_in_hemisEmax_andNotPV, Rec_vtx_thrustCosTheta)")  # INTERMEDIATE
            .Define("Rec_vtx_thrustCosTheta_min_hemisEmin",   "Rec_vtx_thrustCosThetaStatsEmin.at(0)")
            .Define("Rec_vtx_thrustCosTheta_max_hemisEmin",   "Rec_vtx_thrustCosThetaStatsEmin.at(1)")
            .Define("Rec_vtx_thrustCosTheta_ave_hemisEmin",   "Rec_vtx_thrustCosThetaStatsEmin.at(2)")
            .Define("Rec_vtx_thrustCosTheta_min_hemisEmax",   "Rec_vtx_thrustCosThetaStatsEmax.at(0)")
            .Define("Rec_vtx_thrustCosTheta_max_hemisEmax",   "Rec_vtx_thrustCosThetaStatsEmax.at(1)")
            .Define("Rec_vtx_thrustCosTheta_ave_hemisEmax",   "Rec_vtx_thrustCosThetaStatsEmax.at(2)")
        
        )


        # If producing raw_tuples we are done
        if cfg.run_mode == 'no_selection':
            return df2 


        #Add prelim cuts if running in this mode    
        df3 = (
            df2
            #############################################
            ##         Filter events with no PV        ##
            #############################################
            .Define("EVT_hasPV",                "myUtils::hasPV(Rec_VertexObject)")
            .Filter("EVT_hasPV==1")

            #############################################
            ##                  Filters                ##
            #############################################
            .Filter("EVT_hemisEmin_e < 40")        # Energy on the signal side must be < 40 GeV
            .Filter("EVT_hemisEmin_nCharged > 0")  # Signal side must have at least one charged reco particle
            .Filter("EVT_hemisEmin_nLept == 0")    # Remove events with a reconstructed lepton on the signal side -- removes a lot of semileptonic decays
            #.Filter("ROOT::VecOps::Any(Rec_vtx_isPV > 0)")  # Remove events that fail to reconstruct a PV - I domt think this is needed due to above filter

        )

        # If producing files for training BDTh/l then we are done
        if cfg.run_mode == 'prelim_cuts':
            return df3            
        
        ##########################################################################################
        #This section needs changing once trained BDTl/h
        ##########################################################################################
        # Otherwise we evaluate the Stage 1 BDT
        else:
            # Read list of feature names used in the BDT from the config YAML file
            with open(cfg.fccana_opts['yamlPath']) as stream:
                yaml = safe_load(stream)
                BDT1branchList = yaml[cfg.bdt1_opts['mvaBranchList']]

            ROOT.gInterpreter.ProcessLine(f'''
            TMVA::Experimental::RBDT bdt1("{cfg.bdt1_opts['mvaRBDTName']}", "{cfg.bdt1_opts['mvaPath']}");
            auto computeModel1 = TMVA::Experimental::Compute<{len(BDT1branchList)}, float> (bdt1);
            ''')

            df3 = (
                df2
                #############################################
                ##                Build BDT                ##
                #############################################
                .Define("MVAVec",    ROOT.computeModel1, BDT1branchList)
                .Define("EVT_MVA1",  "MVAVec.at(0)")
            )

            # If the cut value is given filter on it else return the entire DataFrame
            if cfg.bdt1_opts['mvaCut'] is not None:
                return df3.Filter(f"EVT_MVA1 > {cfg.bdt1_opts['mvaCut']}")
            else:
                return df3

    def output():
        # Get the output branchList from the config YAML file
        with open(cfg.fccana_opts['yamlPath']) as stream:
            yaml = safe_load(stream)
            branchList = yaml[cfg.fccana_opts['outputBranches'][cfg.run_mode]]
            print(f"----> INFO:")
            print(f"            Output branch list used = {cfg.fccana_opts['outputBranches'][cfg.run_mode]}")

        return branchList