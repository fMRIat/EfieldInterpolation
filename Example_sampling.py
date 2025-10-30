# -*- coding: utf-8 -*-
"""
This example shows how to run the a-priori simulation needed for E-field interpolation. 
The code should be run in a SimNIBS Python environment and requires a charm-generated headmesh as input. 
Prior to running, ensure all paths have been adjusted accordingly.

% Sarah Grosshagauer, Oct 2025
"""


import sys
import os
from simnibs import mni2subject_coords
sys.path.append('.../functions')
import APSim

apsim = APSim.APSim()
apsim.subpath = 'XXXXXXXXX'
apsim.fnamecoil = 'XXX/simnibs/4.0.0/resources/coil_models/Drakaki_BrainStim_2022/MagMore_PMD70.ccd'
apsim.target = mni2subject_coords([-38,44,26], apsim.subpath)

apsim.pathfem = 'APSim_test'
apsim.below_skin_distance = 2
apsim.above_skin_distance = 10
apsim.search_radius = 10
apsim.angle_sampling = 1
apsim.spatial_resolution = 2
apsim.method = 'ADM'
apsim.run()




