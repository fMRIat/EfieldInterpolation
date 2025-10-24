#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example to test the interpolation performance with data provided in the GitHub Repository.
This script can be run in any Python environment which includes the required packages. 
The interpolation code can be integrated in any other Python software, 
e.g. allowing for combining input of a neuronavigation camera directly with stimulator communication.
Prior to running, ensure all paths have been defined accordingly. 

% Sarah Grosshagauer, Oct 2025
"""
import sys
sys.path.append('.../functions')
import numpy as np
from se3_interpolator_trees import load_sampler_from_files, RigidTransform
import re
from pathlib import Path
import time


def create_interpolator(simfolder):
    
    start = time.time()
    geo_path = simfolder.joinpath("coil_positions.geo")
    mat_path = simfolder.joinpath("coil_matrices.txt")
    efield_interpolator = load_sampler_from_files(geo_path, mat_path)
    end = time.time()
    t=end-start
    return efield_interpolator,t


datafolder = Path('.../exampledata') #replace with actual path to datafolder

#Read the ground truth files
coilfile = datafolder.joinpath('neuronavigation_tracking.npy')  #this was derived from actual neuronavigation tracking data
coils = np.load(coilfile)
ground_truth_file =  datafolder.joinpath('neuronavigation_simulations.geo') #simulation results for the neuronavigation tracking data
ground_truth = np.array(list(map(float, re.findall("\\{(\d+\.\d+)\\}", ground_truth_file.open('r').read()))))

interpolator,t = create_interpolator(datafolder)
interp_efield = []
extrapolation = []

start = time.time()
for cci in range(coils.shape[0]):
    coil = coils[cci,:,:]
    coil_val, extrap = interpolator.interpolate(RigidTransform.from_matrix(coil))
    interp_efield.append(coil_val)
    extrapolation.append(extrap)
end = time.time()


mre = np.mean(abs(interp_efield - ground_truth)/ground_truth)
print(f'Time per sample: {np.round((end-start)/coils.shape[0],4)} s')
print(f'Mean relative error of interpolation: {np.round(100*mre,2)}%')
