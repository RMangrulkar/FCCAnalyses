import numpy as np
from efficiency_tools.efficiency_finder import get_eventsProcessed
import config as cfg


get_eventsProcessed("process_with_MC_full_prelim", 
                     samples=cfg.sample_allocations["hadronic_background"],
                     custompath=None,#if runmode is custom
                     save=None, # ie. whether to save log file back to runmode folder
                     )