########## INSTRUCTIONS ############
#
# Use this file to process B -> inv tuples
# Runs in a few different configurations
#   - stage1_training 
#           produce ntuples to train BDT1
#           these have pre-selection cuts on
#           min hemis E and nCharged
#   - stage2_training: 
#           produce ntuples to train BDT2
#           also have a loose cut on BDT1
#   - stage2:          
#           produce final tuples with both BDTs
#           and some loose cuts
#
####################################

import os
import sys

# Config and yaml file must be in this directory by default
# Absolute path must be supplied for the script to work in batch mode
configPath = os.getcwd() 
sys.path.append(os.path.abspath(configPath))

import ROOT
from yaml import safe_load

import config as cfg

#Mandatory: List of processes
processList = cfg.processList[ cfg.run_mode ]

#Mandatory: Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics
prodTag = cfg.fccana_opts['prodTag']

#Mandatory: BSC for vertexing
bsc = cfg.BSC_opts['winter2023'] # list of sigmax,sigmay,sigmaz

#Optional: output directory, default is local running directory
outputDir = cfg.fccana_opts['outputDir'][cfg.run_mode]

#Optional: analysisName, default is ""
analysisName = cfg.fccana_opts['analysisName']

#Optional: ncpus, default is 4
nCPUs = cfg.fccana_opts['nCPUs']

#Optional running on HTCondor, default is False
runBatch = cfg.fccana_opts['runBatch']

#Optional test file
testFile = cfg.fccana_opts['testFile']['bb']

print("----> INFO: Using config.py file from:")
print(f"{15*' '}{os.path.abspath(configPath)}")
print("----> INFO: Using branch names from:")
print(f"{15*' '}{cfg.fccana_opts['yamlPath']}")


class RDFanalysis():

    #__________________________________________________________
    def analysers(df):
        #bsc = [ 6, 25e-3, 400 ]

        df2 = (
            df
            #############################################
            ##          Aliases for # in python        ##
            #############################################
            .Alias("MCRecoAssociationsRec", "MCRecoAssociations#0.index")  # points to ReconstructedParticles
            .Alias("MCRecoAssociationsGen", "MCRecoAssociations#1.index")  # points to Particle
            .Alias("ParticleParents",       "Particle#0.index")            # gen particle parents
            .Alias("ParticleChildren",      "Particle#1.index")            # gen particle children

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
            ##         Filter events with no PV        ##
            #############################################
            .Define("EVT_hasPV",                "myUtils::hasPV(Rec_VertexObject)")
            .Filter("EVT_hasPV==1")
            
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

            .Define("EVT_hemisEmax_e",         "EVT_ThrustInfoMax_E.at(0)")
            .Define("EVT_hemisEmax_eCharged",  "EVT_ThrustInfoMax_E.at(1)")
            .Define("EVT_hemisEmax_eNeutral",  "EVT_ThrustInfoMax_E.at(2)")
            .Define("EVT_hemisEmax_n",         "float(EVT_ThrustInfoMax_N.at(0))")
            .Define("EVT_hemisEmax_nCharged",  "float(EVT_ThrustInfoMax_N.at(1))")
            .Define("EVT_hemisEmax_nNeutral",  "float(EVT_ThrustInfoMax_N.at(2))")

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
            
            #############################################
            ##                  Filters                ##
            #############################################
            .Filter("EVT_hemisEmin_e < 40")        # Energy on the signal side must be < 40 GeV
            .Filter("EVT_hemisEmin_nCharged > 0")  # Signal side must have at least one charged reco particle
            .Filter("EVT_hemisEmin_nLept == 0")    # Remove events with a reconstructed lepton on the signal side -- removes a lot of semileptonic decays

        )

        # If producing files for stage1 training then we are done
        if cfg.run_mode == 'stage1_training':
            return df2            
        
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