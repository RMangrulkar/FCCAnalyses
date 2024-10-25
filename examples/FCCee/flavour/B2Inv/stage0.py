import os
import sys
from yaml import safe_load

# Config and yaml file must be in this directory by default
# Absolute path must be supplied for the script to work in batch mode
configPath = '/r02/lhcb/rrm42/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv'
sys.path.append(os.path.abspath(configPath))

import config as cfg

#Mandatory: List of processes
processList = cfg.processList['stage0']

#Mandatory: Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics
prodTag     = cfg.fccana_opts['prodTag']

#Optional: output directory, default is local running directory
outputDir   = cfg.fccana_opts['outputDir']['stage0']

#Optional: analysisName, default is ""
analysisName = cfg.fccana_opts['analysisName']

#Optional: ncpus, default is 4
nCPUS       = cfg.fccana_opts['nCPUs']

#Optional running on HTCondor, default is False
runBatch    = cfg.fccana_opts['runBatch']

#Optional batch queue name when running on HTCondor, default is workday
batchQueue = cfg.fccana_opts['runBatch']

#Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
compGroup = cfg.fccana_opts['compGroup']

#Optional test file
testFile = cfg.fccana_opts['testFile']['bb']

print("----> INFO: Using config.py file from:")
print(f"{15*' '}{os.path.abspath(configPath)}")
print("----> INFO: Using branch names from:")
print(f"{15*' '}{cfg.fccana_opts['yamlPath']}")

#Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis():

    #__________________________________________________________
    #Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
        df2 = (
            df
            #############################################
            ##          Aliases for # in python        ##
            #############################################
            .Alias("MCRecoAssociationsRec", "MCRecoAssociations#0.index") # points to ReconstructedParticles
            .Alias("MCRecoAssociationsGen", "MCRecoAssociations#1.index") # points to Particle
            .Alias("ParticleParents",       "Particle#0.index") # gen particle parents
            .Alias("ParticleChildren",      "Particle#1.index") # gen particle children

            #############################################
            ##             MC IDs and Status           ##
            #############################################
            .Define("MC_n",           "MCParticle::get_n(Particle)")
            .Define("MC_genStatus",   "MCParticle::get_genStatus(Particle)")
            .Define("MC_PDG",         "MCParticle::get_pdg(Particle)")
            .Define("MC_M1",          "myUtils::getMC_parent(0,Particle,ParticleParents)")
            .Define("MC_M2",          "myUtils::getMC_parent(1,Particle,ParticleParents)")
            .Define("MC_D1",          "myUtils::getMC_daughter(0,Particle,ParticleChildren)")
            .Define("MC_D2",          "myUtils::getMC_daughter(1,Particle,ParticleChildren)")
            .Define("MC_D3",          "myUtils::getMC_daughter(2,Particle,ParticleChildren)")
            .Define("MC_D4",          "myUtils::getMC_daughter(3,Particle,ParticleChildren)")


            #############################################
            ##               MC Particles              ##
            #############################################
            .Define("MC_e",           "MCParticle::get_e(Particle)")
            .Define("MC_m",           "MCParticle::get_mass(Particle)")
            .Define("MC_q",           "MCParticle::get_charge(Particle)")
            .Define("MC_p",           "MCParticle::get_p(Particle)")
            .Define("MC_pt",          "MCParticle::get_pt(Particle)")
            .Define("MC_px",          "MCParticle::get_px(Particle)")
            .Define("MC_py",          "MCParticle::get_py(Particle)")
            .Define("MC_pz",          "MCParticle::get_pz(Particle)")
            .Define("MC_eta",         "MCParticle::get_eta(Particle)")
            .Define("MC_phi",         "MCParticle::get_phi(Particle)")
            .Define("MC_orivtx_x",    "MCParticle::get_vertex_x(Particle)")
            .Define("MC_orivtx_y",    "MCParticle::get_vertex_y(Particle)")
            .Define("MC_orivtx_z",    "MCParticle::get_vertex_z(Particle)")    


            #############################################
            ##Special MC Particles ---> moved to stage1##
            #############################################
            # --------------------------------------- #
            #             MC intermediates            #
            # --------------------------------------- #
            # Pythia8 generatorStatus
            # 21 - incoming particles of hardest process (e+ e- beams)
            # 22 - intermediate particles of hardest process (Z)
            # 23 - outgoing particles of hardest process (quark pair produced from Z)
            #  1 - final-state particles
            .Define("MC_ee",          "MCParticle::sel_genStatus(21)(Particle)")
            .Define("MC_Z",           "MCParticle::sel_genStatus(22)(Particle)")
            .Define("MC_qq",          "MCParticle::sel_genStatus(23)(Particle)")
            .Define("MC_FS",          "MCParticle::sel_genStatus(1)(Particle)")

            # --------------------------------------- #
            #           MC e+ e- variables            #
            # --------------------------------------- #
            .Define("MCem_e",         "(MCParticle::get_e(MC_ee)).at(0)")
            .Define("MCem_m",         "(MCParticle::get_mass(MC_ee)).at(0)")
            .Define("MCem_q",         "(MCParticle::get_charge(MC_ee)).at(0)")
            .Define("MCem_p",         "(MCParticle::get_p(MC_ee)).at(0)")
            .Define("MCem_pt",        "(MCParticle::get_pt(MC_ee)).at(0)")
            .Define("MCem_px",        "(MCParticle::get_px(MC_ee)).at(0)")
            .Define("MCem_py",        "(MCParticle::get_py(MC_ee)).at(0)")
            .Define("MCem_pz",        "(MCParticle::get_pz(MC_ee)).at(0)")
            .Define("MCem_eta",       "(MCParticle::get_eta(MC_ee)).at(0)")
            .Define("MCem_phi",       "(MCParticle::get_phi(MC_ee)).at(0)")
            .Define("MCem_orivtx_x",  "(MCParticle::get_vertex_x(MC_ee)).at(0)")
            .Define("MCem_orivtx_y",  "(MCParticle::get_vertex_y(MC_ee)).at(0)")
            .Define("MCem_orivtx_z",  "(MCParticle::get_vertex_z(MC_ee)).at(0)")
            .Define("MCep_e",         "(MCParticle::get_e(MC_ee)).at(1)")
            .Define("MCep_m",         "(MCParticle::get_mass(MC_ee)).at(1)")
            .Define("MCep_q",         "(MCParticle::get_charge(MC_ee)).at(1)")
            .Define("MCep_p",         "(MCParticle::get_p(MC_ee)).at(1)")
            .Define("MCep_pt",        "(MCParticle::get_pt(MC_ee)).at(1)")
            .Define("MCep_px",        "(MCParticle::get_px(MC_ee)).at(1)")
            .Define("MCep_py",        "(MCParticle::get_py(MC_ee)).at(1)")
            .Define("MCep_pz",        "(MCParticle::get_pz(MC_ee)).at(1)")
            .Define("MCep_eta",       "(MCParticle::get_eta(MC_ee)).at(1)")
            .Define("MCep_phi",       "(MCParticle::get_phi(MC_ee)).at(1)")
            .Define("MCep_orivtx_x",  "(MCParticle::get_vertex_x(MC_ee)).at(1)")
            .Define("MCep_orivtx_y",  "(MCParticle::get_vertex_y(MC_ee)).at(1)")
            .Define("MCep_orivtx_z",  "(MCParticle::get_vertex_z(MC_ee)).at(1)")

            # --------------------------------------- #
            #           MC Z boson variables          #
            # --------------------------------------- #
            .Define("MCZ_e",          "(MCParticle::get_e(MC_Z)).at(0)")
            .Define("MCZ_m",          "(MCParticle::get_mass(MC_Z)).at(0)")
            .Define("MCZ_q",          "(MCParticle::get_charge(MC_Z)).at(0)")
            .Define("MCZ_p",          "(MCParticle::get_p(MC_Z)).at(0)")
            .Define("MCZ_pt",         "(MCParticle::get_pt(MC_Z)).at(0)")
            .Define("MCZ_px",         "(MCParticle::get_px(MC_Z)).at(0)")
            .Define("MCZ_py",         "(MCParticle::get_py(MC_Z)).at(0)")
            .Define("MCZ_pz",         "(MCParticle::get_pz(MC_Z)).at(0)")
            .Define("MCZ_eta",        "(MCParticle::get_eta(MC_Z)).at(0)")
            .Define("MCZ_phi",        "(MCParticle::get_phi(MC_Z)).at(0)")
            .Define("MCZ_orivtx_x",   "(MCParticle::get_vertex_x(MC_Z)).at(0)")
            .Define("MCZ_orivtx_y",   "(MCParticle::get_vertex_y(MC_Z)).at(0)")
            .Define("MCZ_orivtx_z",   "(MCParticle::get_vertex_z(MC_Z)).at(0)")

            # --------------------------------------- #
            #            MC qqbar variables           #
            # --------------------------------------- #
            .Define("MCq1_PDG",       "(MCParticle::get_pdg(MC_qq)).at(0)")
            .Define("MCq1_e",         "(MCParticle::get_e(MC_qq)).at(0)")
            .Define("MCq1_m",         "(MCParticle::get_mass(MC_qq)).at(0)")
            .Define("MCq1_q",         "(MCParticle::get_charge(MC_qq)).at(0)")
            .Define("MCq1_p",         "(MCParticle::get_p(MC_qq)).at(0)")
            .Define("MCq1_pt",        "(MCParticle::get_pt(MC_qq)).at(0)")
            .Define("MCq1_px",        "(MCParticle::get_px(MC_qq)).at(0)")
            .Define("MCq1_py",        "(MCParticle::get_py(MC_qq)).at(0)")
            .Define("MCq1_pz",        "(MCParticle::get_pz(MC_qq)).at(0)")
            .Define("MCq1_eta",       "(MCParticle::get_eta(MC_qq)).at(0)")
            .Define("MCq1_phi",       "(MCParticle::get_phi(MC_qq)).at(0)")
            .Define("MCq1_orivtx_x",  "(MCParticle::get_vertex_x(MC_qq)).at(0)")
            .Define("MCq1_orivtx_y",  "(MCParticle::get_vertex_y(MC_qq)).at(0)")
            .Define("MCq1_orivtx_z",  "(MCParticle::get_vertex_z(MC_qq)).at(0)")
            .Define("MCq2_PDG",       "(MCParticle::get_pdg(MC_qq)).at(1)")
            .Define("MCq2_e",         "(MCParticle::get_e(MC_qq)).at(1)")
            .Define("MCq2_m",         "(MCParticle::get_mass(MC_qq)).at(1)")
            .Define("MCq2_q",         "(MCParticle::get_charge(MC_qq)).at(1)")
            .Define("MCq2_p",         "(MCParticle::get_p(MC_qq)).at(1)")
            .Define("MCq2_pt",        "(MCParticle::get_pt(MC_qq)).at(1)")
            .Define("MCq2_px",        "(MCParticle::get_px(MC_qq)).at(1)")
            .Define("MCq2_py",        "(MCParticle::get_py(MC_qq)).at(1)")
            .Define("MCq2_pz",        "(MCParticle::get_pz(MC_qq)).at(1)")
            .Define("MCq2_eta",       "(MCParticle::get_eta(MC_qq)).at(1)")
            .Define("MCq2_phi",       "(MCParticle::get_phi(MC_qq)).at(1)")
            .Define("MCq2_orivtx_x",  "(MCParticle::get_vertex_x(MC_qq)).at(1)")
            .Define("MCq2_orivtx_y",  "(MCParticle::get_vertex_y(MC_qq)).at(1)")
            .Define("MCq2_orivtx_z",  "(MCParticle::get_vertex_z(MC_qq)).at(1)")

            # --------------------------------------- #
            #    MC final-state particle variables    #
            # --------------------------------------- #
            .Define("MCfinal_PDG",       "MCParticle::get_pdg(MC_FS)")
            .Define("MCfinal_e",         "MCParticle::get_e(MC_FS)")
            .Define("MCfinal_m",         "MCParticle::get_mass(MC_FS)")
            .Define("MCfinal_q",         "MCParticle::get_charge(MC_FS)")
            .Define("MCfinal_p",         "MCParticle::get_p(MC_FS)")
            .Define("MCfinal_pt",        "MCParticle::get_pt(MC_FS)")
            .Define("MCfinal_px",        "MCParticle::get_px(MC_FS)")
            .Define("MCfinal_py",        "MCParticle::get_py(MC_FS)")
            .Define("MCfinal_pz",        "MCParticle::get_pz(MC_FS)")
            .Define("MCfinal_eta",       "MCParticle::get_eta(MC_FS)")
            .Define("MCfinal_phi",       "MCParticle::get_phi(MC_FS)")
            .Define("MCfinal_orivtx_x",  "MCParticle::get_vertex_x(MC_FS)")
            .Define("MCfinal_orivtx_y",  "MCParticle::get_vertex_y(MC_FS)")
            .Define("MCfinal_orivtx_z",  "MCParticle::get_vertex_z(MC_FS)")


            #############################################
            ##            MC PrimaryVertex             ##
            #############################################
            # --------------------------------------- #
            #             MC_PV intermediate          #
            # --------------------------------------- #
            .Define("MC_PrimaryVertex",  "MCParticle::get_EventPrimaryVertex(21)(Particle)") 
            
            .Define("MC_PV_x",  "MC_PrimaryVertex.X()") 
            .Define("MC_PV_y",  "MC_PrimaryVertex.Y()") 
            .Define("MC_PV_z",  "MC_PrimaryVertex.Z()")


            #############################################
            ##           Find MC Vertices              ##
            #############################################
            # --------------------------------------- #
            #            MC_vtx intermediate          #
            # --------------------------------------- #
            .Define("MC_VertexObject",   "myUtils::get_MCVertexObject(Particle, ParticleParents)")
            
            .Define("MC_vtx_n",        "int(MC_VertexObject.size())")
            .Define("MC_vtx_ntracks",  "myUtils::get_NTracksMCVertex(MC_VertexObject)")
            .Define("MC_vtx_indMC",    "myUtils::get_MCindMCVertex(MC_VertexObject)")
            .Define("MC_vtx_x",        "myUtils::get_MCVertex_x(MC_VertexObject)")
            .Define("MC_vtx_y",        "myUtils::get_MCVertex_y(MC_VertexObject)")
            .Define("MC_vtx_z",        "myUtils::get_MCVertex_z(MC_VertexObject)")
            
            # MCParticle variable that needed MC_VertexObject
            .Define("MC_orivtx_ind",  "myUtils::get_MCVertex_fromMC(Particle, MC_VertexObject)")
            
            #############################################
            ##       Crucial Intermediate Objects      ##
            #############################################
            # Get collection of tracks consistent with a PV (i.e. not downstream Ks, Lb etc. tracks)
            # using the get_PrimaryTracks() method with a beam spot constraint under the following parameters
            # bsc_sigma(x,y,z) = (4.5, 20e-3, 300)
            # bsc_(x,y,z) = (0,0,0) 
            .Define("Rec_PrimaryTracks",        "VertexFitterSimple::get_PrimaryTracks( EFlowTrack_1, true, 4.5, 20e-3, 300, 0., 0., 0. )")
            # Run the vertex fit using only the primary tracks with the same beamspot constraint
            .Define("Rec_PrimaryVertexObject",  "VertexFitterSimple::VertexFitter_Tk( 1, Rec_PrimaryTracks, true, 4.5, 20e-3, 300 )")
            .Define("Rec_PrimaryVertex",        "Rec_PrimaryVertexObject.vertex")
            # function to get all reco vertices (uses MC vertex to seed the vertexing)
            .Define("Rec_VertexObject",         "myUtils::get_VertexObject(MC_VertexObject, ReconstructedParticles, EFlowTrack_1, MCRecoAssociationsRec, MCRecoAssociationsGen)")

            # actually add the PID hypothesis info to the RecParticles (based on MC truth)
            # ie we assume perfect PID here
            .Define("RecoParticlesPID",          "myUtils::PID(ReconstructedParticles, MCRecoAssociationsRec, MCRecoAssociationsGen, Particle)")
            # now update reco momentum based on the rec vertex
            .Define("RecoParticlesPIDAtVertex",  "myUtils::get_RP_atVertex(RecoParticlesPID, Rec_VertexObject)")

            #############################################
            ##         Reconstructed Particles         ##
            #############################################
            # MC associated with RecoP
            .Define("MC_fromRP",           "myUtils::get_MCObject_fromRP(MCRecoAssociationsRec, MCRecoAssociationsGen, RecoParticlesPIDAtVertex, Particle)")  # INTERMEDIATE

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
            
            # ReconstructedParticle variables
            .Define("Rec_n",         "ReconstructedParticle::get_n(RecoParticlesPIDAtVertex)")
            .Define("Rec_type",      "ReconstructedParticle::get_type(RecoParticlesPIDAtVertex)")
            .Define("Rec_indMC",     "ReconstructedParticle2MC::getRP2MC_index(MCRecoAssociationsRec,MCRecoAssociationsGen,RecoParticlesPIDAtVertex)")
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

            #############################################
            ##   EVT -- Needed for angular variables   ##
            #############################################
            .Define("EVT_ThrustInfoNoPointing",     'Algorithms::minimize_thrust("Minuit2","Migrad")(Rec_px, Rec_py, Rec_pz)')
            .Define("EVT_ThrustCosThetaNoPointing", "Algorithms::getAxisCosTheta(EVT_ThrustInfoNoPointing, Rec_px, Rec_py, Rec_pz)")
            .Define("EVT_ThrustInfo",               "Algorithms::getThrustPointing(1.)(EVT_ThrustCosThetaNoPointing, Rec_e, EVT_ThrustInfoNoPointing)")
            
            # Remaining Rec variables
            .Define("Rec_thrustCosTheta",  "Algorithms::getAxisCosTheta(EVT_ThrustInfo, Rec_px, Rec_py, Rec_pz)")
            .Define("Rec_in_hemisEmin",    "myUtils::get_RP_inHemis(1)(Rec_thrustCosTheta)")
            .Define("Rec_in_hemisEmax",    "myUtils::get_RP_inHemis(0)(Rec_thrustCosTheta)") # Not saved because redundant but used for other variables

            #############################################
            ##       Reconstructed PrimaryVertex       ##
            #############################################
            # Get collection of all tracks and use this to reconstruct the PV
            .Define("Rec_PV_ntracks",  "Rec_PrimaryTracks.size()")
            .Define("Rec_PV_x",        "Rec_PrimaryVertex.position.x")
            .Define("Rec_PV_y",        "Rec_PrimaryVertex.position.y")
            .Define("Rec_PV_z",        "Rec_PrimaryVertex.position.z")

            # Track information
            .Define("Rec_track_n",       "ReconstructedParticle2Track::getTK_n(EFlowTrack_1)")
            .Define("Rec_track_d0",      "ReconstructedParticle2Track::getRP2TRK_D0(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_normd0",  "ReconstructedParticle2Track::getRP2TRK_D0_sig(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_z0",      "ReconstructedParticle2Track::getRP2TRK_Z0(RecoParticlesPIDAtVertex, EFlowTrack_1)")
            .Define("Rec_track_normz0",  "ReconstructedParticle2Track::getRP2TRK_Z0_sig(RecoParticlesPIDAtVertex, EFlowTrack_1)")

            #############################################
            ##           Reconstructed Vertex          ##
            #############################################
            .Define("Rec_vtx_n",               "int(Rec_VertexObject.size())")
            .Define("Rec_vtx_indMCvtx",        "myUtils::get_Vertex_indMC(Rec_VertexObject)")
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

            .Define("Rec_vtx_d2PV",            "myUtils::get_Vertex_d2PV(Rec_VertexObject,-1)")
            .Define("Rec_vtx_d2PV_min",        "myUtils::get_dPV2DV_min(Rec_vtx_d2PV)")
            .Define("Rec_vtx_d2PV_max",        "myUtils::get_dPV2DV_max(Rec_vtx_d2PV)")
            .Define("Rec_vtx_d2PV_ave",        "myUtils::get_dPV2DV_ave(Rec_vtx_d2PV)")
            .Define("Rec_vtx_d2PV_x",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 0)")
            .Define("Rec_vtx_d2PV_y",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 1)")
            .Define("Rec_vtx_d2PV_z",          "myUtils::get_Vertex_d2PV(Rec_VertexObject, 2)")
            .Define("Rec_vtx_d2PV_err",        "myUtils::get_Vertex_d2PVError(Rec_VertexObject,-1)")
            .Define("Rec_vtx_d2PV_xerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 0)")
            .Define("Rec_vtx_d2PV_yerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 1)")
            .Define("Rec_vtx_d2PV_zerr",       "myUtils::get_Vertex_d2PVError(Rec_VertexObject, 2)")
            .Define("Rec_vtx_normd2PV",        "Rec_vtx_d2PV / Rec_vtx_d2PV_err")
            .Define("Rec_vtx_normd2PV_min",    "myUtils::get_dPV2DV_min(Rec_vtx_normd2PV)")
            .Define("Rec_vtx_normd2PV_max",    "myUtils::get_dPV2DV_max(Rec_vtx_normd2PV)")
            .Define("Rec_vtx_normd2PV_ave",    "myUtils::get_dPV2DV_ave(Rec_vtx_normd2PV)")
            .Define("Rec_vtx_normd2PV_x",      "Rec_vtx_d2PV_x / Rec_vtx_d2PV_xerr")
            .Define("Rec_vtx_normd2PV_y",      "Rec_vtx_d2PV_y / Rec_vtx_d2PV_yerr")
            .Define("Rec_vtx_normd2PV_z",      "Rec_vtx_d2PV_z / Rec_vtx_d2PV_zerr")
            # Vertex relations to thrust 
            .Define("Rec_vtx_thrustCosTheta",  "myUtils::get_Vertex_thrusthemis_angle(Rec_VertexObject, RecoParticlesPIDAtVertex, EVT_ThrustInfo)") 
            # Flag vertex in max or min hemisphere
            .Define("Rec_vtx_in_hemisEmin",    "myUtils::get_Vertex_thrusthemis(Rec_vtx_thrustCosTheta, 1)")
            .Define("Rec_vtx_in_hemisEmax",    "myUtils::get_Vertex_thrusthemis(Rec_vtx_thrustCosTheta, 0)")

            #############################################
            ##               Reco Thrust               ##
            #############################################
            .Define("EVT_Thrust_mag",          "EVT_ThrustInfo.at(0)")
            .Define("EVT_Thrust_x",            "EVT_ThrustInfo.at(1)")
            .Define("EVT_Thrust_xerr",         "EVT_ThrustInfo.at(2)")
            .Define("EVT_Thrust_y",            "EVT_ThrustInfo.at(3)")
            .Define("EVT_Thrust_yerr",         "EVT_ThrustInfo.at(4)")
            .Define("EVT_Thrust_z",            "EVT_ThrustInfo.at(5)")
            .Define("EVT_Thrust_zerr",         "EVT_ThrustInfo.at(6)")
            
            #############################################
            ##         Emin and Emax Hemispheres       ##
            #############################################
            .Define("EVT_ThrustInfoMax_N",     "Algorithms::getAxisN(0)(Rec_thrustCosTheta, Rec_q)")
            .Define("EVT_ThrustInfoMin_N",     "Algorithms::getAxisN(1)(Rec_thrustCosTheta, Rec_q)")
            .Define("EVT_ThrustInfoMax_E",     "Algorithms::getAxisEnergy(0)(Rec_thrustCosTheta, Rec_q, Rec_e)")
            .Define("EVT_ThrustInfoMin_E",     "Algorithms::getAxisEnergy(1)(Rec_thrustCosTheta, Rec_q, Rec_e)")
            # Count secondary vertices in each hemisphere
            .Define("SecondaryVertexThrustAngle",  "myUtils::get_DVertex_thrusthemis_angle(Rec_VertexObject, RecoParticlesPIDAtVertex, EVT_ThrustInfo)")
            
            .Define("EVT_hemisEmax_e",         "EVT_ThrustInfoMax_E.at(0)")
            .Define("EVT_hemisEmax_eCharged",  "EVT_ThrustInfoMax_E.at(1)")
            .Define("EVT_hemisEmax_eNeutral",  "EVT_ThrustInfoMax_E.at(2)")
            .Define("EVT_hemisEmax_n",         "EVT_ThrustInfoMax_N.at(0)")
            .Define("EVT_hemisEmax_nCharged",  "EVT_ThrustInfoMax_N.at(1)")
            .Define("EVT_hemisEmax_nNeutral",  "EVT_ThrustInfoMax_N.at(2)")
            .Define("EVT_hemisEmin_e",         "EVT_ThrustInfoMin_E.at(0)")
            .Define("EVT_hemisEmin_eCharged",  "EVT_ThrustInfoMin_E.at(1)")
            .Define("EVT_hemisEmin_eNeutral",  "EVT_ThrustInfoMin_E.at(2)")
            .Define("EVT_hemisEmin_n",         "EVT_ThrustInfoMin_N.at(0)")
            .Define("EVT_hemisEmin_nCharged",  "EVT_ThrustInfoMin_N.at(1)")
            .Define("EVT_hemisEmin_nNeutral",  "EVT_ThrustInfoMin_N.at(2)")
            
            .Define("EVT_hemisEmin_nDV",     "myUtils::get_Npos(SecondaryVertexThrustAngle)")
            .Define("EVT_hemisEmax_nDV",     "myUtils::get_Nneg(SecondaryVertexThrustAngle)")
            
            .Define("EVT_hemisEmin_Emiss",  f"{0.5*cfg.mass_Z} - (EVT_hemisEmin_e)")
            .Define("EVT_hemisEmax_Emiss",  f"{0.5*cfg.mass_Z} - (EVT_hemisEmax_e)")
            .Define("EVT_Thrust_deltaE",     "(EVT_hemisEmax_e) - (EVT_hemisEmin_e)")

            #############################################
            ##      Hemisphere Particle variables      ##
            #############################################
            # --------------------------------------- #
            #           Hemis intermediates           #
            # --------------------------------------- #
            .Define("EVT_EminPartInfo",  "myUtils::get_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmin)")
            .Define("EVT_EmaxPartInfo",  "myUtils::get_RP_HemisInfo(RecoParticlesPIDAtVertex, Rec_VertexObject, Rec_in_hemisEmax)")
            
            .Define("EVT_hemisEmin_nLept",                "(EVT_EminPartInfo.at(0)).num")
            .Define("EVT_hemisEmin_nKaon",                "(EVT_EminPartInfo.at(1)).num")
            .Define("EVT_hemisEmin_nPion",                "(EVT_EminPartInfo.at(2)).num")
            .Define("EVT_hemisEmin_maxeLept",             "(EVT_EminPartInfo.at(0)).maxE")
            .Define("EVT_hemisEmin_maxeKaon",             "(EVT_EminPartInfo.at(1)).maxE")
            .Define("EVT_hemisEmin_maxePion",             "(EVT_EminPartInfo.at(2)).maxE")
            .Define("EVT_hemisEmin_maxeLept_fromtruePV",  "(EVT_EminPartInfo.at(0)).fromPV")
            .Define("EVT_hemisEmin_maxeKaon_fromtruePV",  "(EVT_EminPartInfo.at(1)).fromPV")
            .Define("EVT_hemisEmin_maxePion_fromtruePV",  "(EVT_EminPartInfo.at(2)).fromPV")
            .Define("EVT_hemisEmin_maxeLept_ind",         "(EVT_EminPartInfo.at(0)).index")
            .Define("EVT_hemisEmin_maxeKaon_ind",         "(EVT_EminPartInfo.at(1)).index")
            .Define("EVT_hemisEmin_maxePion_ind",         "(EVT_EminPartInfo.at(2)).index")
            
            .Define("EVT_hemisEmax_nLept",                "(EVT_EmaxPartInfo.at(0)).num")
            .Define("EVT_hemisEmax_nKaon",                "(EVT_EmaxPartInfo.at(1)).num")
            .Define("EVT_hemisEmax_nPion",                "(EVT_EmaxPartInfo.at(2)).num")
            .Define("EVT_hemisEmax_maxeLept",             "(EVT_EmaxPartInfo.at(0)).maxE")
            .Define("EVT_hemisEmax_maxeKaon",             "(EVT_EmaxPartInfo.at(1)).maxE")
            .Define("EVT_hemisEmax_maxePion",             "(EVT_EmaxPartInfo.at(2)).maxE")
            .Define("EVT_hemisEmax_maxeLept_fromtruePV",  "(EVT_EmaxPartInfo.at(0)).fromPV")
            .Define("EVT_hemisEmax_maxeKaon_fromtruePV",  "(EVT_EmaxPartInfo.at(1)).fromPV")
            .Define("EVT_hemisEmax_maxePion_fromtruePV",  "(EVT_EmaxPartInfo.at(2)).fromPV")
            .Define("EVT_hemisEmax_maxeLept_ind",         "(EVT_EmaxPartInfo.at(0)).index")
            .Define("EVT_hemisEmax_maxeKaon_ind",         "(EVT_EmaxPartInfo.at(1)).index")
            .Define("EVT_hemisEmax_maxePion_ind",         "(EVT_EmaxPartInfo.at(2)).index")
            
            #############################################
            ##  Thrust hemispheres energy difference   ##
            #############################################
            .Define("EVT_hemisEmin_containsTau23Pi", "myUtils::get_hemis_containstau23pi(Rec_in_hemisEmin, Rec_true_PDG, Rec_true_M1, Rec_indvtx, Rec_vtx_ntracks)")
        )
        
        return df2

    #__________________________________________________________
    #Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        # Get the output branchList from the config YAML file
        with open(cfg.fccana_opts['yamlPath']) as stream:
            yaml = safe_load(stream)
            branchList = yaml[cfg.fccana_opts['outBranchList0']]
            print(f"----> INFO:")
            print(f"            Output branch list used = {cfg.fccana_opts['outBranchList0']}")

        return branchList
