# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:51:45 2025

@author: sarah
"""


import sys
import os
from simnibs import mni2subject_coords
sys.path.append('.../functions')
import APSim

apsim = APSim.APSim()
apsim.subpath = 'XXXXXXXXX'
apsim.fnamecoil = 'XXX/simnibs/4.0.0/resources/coil_models/Drakaki_BrainStim_2022/MagMore_PMD70.ccd'
apsim.target = mni2subject_coords([-38,44,26], tms_opt.subpath)

apsim.pathfem = 'APSim_test'
apsim.below_skin_distance = 2
apsim.search_radius = 10
apsim.above_skin_distance = 10
apsim.angle_sampling = 1
apsim.spatial_resolution = 2
apsim.method = 'ADM'
apsim.run()



