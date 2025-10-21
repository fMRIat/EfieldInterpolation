# -*- coding: utf-8 -*-
"""
Interpolation algorithm for E-field guided motion compensation

This algorithm and the corresponding code was developed by Michael Woletz. 
To use it, one requires a .txt file containing all 4x4 coil matrices in a stacked/flattened array of shape Nx4 - where N corresponds to the number of matrices times 4.
The second input file is the .geo file.
Both files are saved in the respective simulation folder by APSim. 

Code written by Michael Woletz 

"""

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
import time
import numba
from pathlib import Path
import re

class SE3Interpolator:
    def __init__(self, rigid_transforms, values, euclidean_search_radius: float = None, quaternion_search_radius: float = None, distance_weighting_power: float = 2.0):
        self._rigid_transforms = rigid_transforms
        self._values = np.asarray(values)
        self._euclidean_search_radius  = euclidean_search_radius
        self._quaternion_search_radius = quaternion_search_radius
        self._distance_weighting_power = distance_weighting_power

        self._build_trees()

    @classmethod
    def from_matrices(cls, matrices, values):
        rigid_transforms = [RigidTransform.from_matrix(mat)
                                for mat in matrices]
        return cls(rigid_transforms, values)
    
    @property
    def rigid_transforms(self):
        return self._rigid_transforms

    @property
    def values(self):
        return self._values
    
    @property
    def euclidean_search_radius(self) -> float:
        return self._euclidean_search_radius
    
    @euclidean_search_radius.setter
    def euclidean_search_radius(self, v: float):
        self._euclidean_search_radius = v
        
    @property
    def quaternion_search_radius(self) -> float:
        return self._quaternion_search_radius
    
    @quaternion_search_radius.setter
    def quaternion_search_radius(self, v: float):
        self._quaternion_search_radius = v
        for quaternion_tree in self.quaternion_trees:
            quaternion_tree.search_radius = v

    @property
    def distance_weighting_power(self) -> float:
        return self._distance_weighting_power
    
    @distance_weighting_power.setter
    def distance_weighting_power(self, v: float):
        self._distance_weighting_power = float(v)

        for quaternion_tree in self.quaternion_trees:
            quaternion_tree.distance_weighting_power = self._distance_weighting_power

    def _build_trees(self):
        # extract the translations from all transforms
        translations = [rt.t for rt in self.rigid_transforms]

        # get the unique translations
        u_translations, u_indices = np.unique(translations,
                                        return_inverse=True,
                                        axis=0)
        
        self._unique_translations = u_translations

        # build a search tree for finding the closest points in euclidean space
        t_start = time.time()
        self._translation_tree = BallTree(u_translations)
        print(f"Building translation tree took {time.time()-t_start} seconds.")
        
        self._quaternion_trees = []
        
        if self.euclidean_search_radius == None:
            # if the search radius is not set, get one from the data
            # get the nearest neighbours for each point and compute a sensible search radius
            d, i = self._translation_tree.query(u_translations, 2)
            d = d[:,1]
            
            self.euclidean_search_radius = d.max() + 0.5 * np.median(d)
            
        print(f"{len(u_translations)} unique translations.")
        t_start = time.time()
        # for each unique position, build a tree of the rotations using the arclength between the rotations as distance metric
        for i in range(len(u_translations)):
            js = np.flatnonzero(u_indices == i)

            # get the quaternions from the transformations at this position
            quaternions = [self.rigid_transforms[j].q for j in js]
            values_js   = [self.values[j] for j in js]

            # build a search tree for the quaternions and the corresponding values
            self._quaternion_trees.append(
                QuaternionInterpolationTree(quaternions, values_js, self.quaternion_search_radius))
        print(f"Building quaternion trees took {time.time()-t_start} seconds.")
    
    @property
    def unique_translations(self):
        return self._unique_translations

    @property
    def translation_tree(self):
        return self._translation_tree
    
    @property
    def quaternion_trees(self):
        return self._quaternion_trees
    
    def interpolate(self, rt) -> tuple[float, bool]:
        # search for the nearest positions in euclidean space
        i, d = self.translation_tree.query_radius(
                        [rt.t],
                        self.euclidean_search_radius,
                        return_distance=True)
                        
        # since we only accept one point at the time, take the single result
        d = d[0]
        i = i[0]
        
        extrapolated = False
        
        if len(d) == 0:
            # if no sample was within the search radius, get the closest sample
            extrapolated = True
            d, i = self.translation_tree.query([rt.t])
            d = d[0]
            i = i[0]
            
        # interpolate the quaternions at the neighbours
        values = []
        extrapolated_quaternions = []
        for ind in i:
            v, extrapolated_quaternion = self.quaternion_trees[ind](rt.q)
            values.append(v)
            extrapolated_quaternions.append(extrapolated_quaternion)
            
        values                   = np.array(values)
        extrapolated_quaternions = np.array(extrapolated_quaternions)

        # only use the closest not extrapolated results if possible
        if np.any(extrapolated_quaternions) and (not np.all(extrapolated_quaternions) and not extrapolated_quaternions[0]):
            index = np.flatnonzero(extrapolated_quaternions)
            d = d[:index[0]]
            i = i[:index[0]]
            values = values[:index[0]]
            extrapolated_quaternions = extrapolated_quaternions[:index[0]]
        
        v = inverse_distance_weighting(d, values, power = self.distance_weighting_power)
            
        return (v, extrapolated or np.any(extrapolated_quaternions))
        
class QuaternionInterpolationTree:
    def __init__(self, qs, values, search_radius, distance_weighting_power: float = 2.0):
        self._qs = np.array(qs)
        self._values = np.array(values)
        self._search_radius = search_radius
        self._distance_weighting_power = distance_weighting_power

        self._tree = BallTree(self.qs, metric=get_quaternion_metric())

        if search_radius is None:
            d, i = self._tree.query(self.qs, 2)
            d = d[:,1]
                
            self.search_radius = d.max() + 0.5 * np.median(d)

    @property
    def qs(self):
        return self._qs

    @property
    def values(self):
        return self._values
    
    @property
    def search_radius(self):
        return self._search_radius
    
    @search_radius.setter
    def search_radius(self, v: float):
        self._search_radius = v

    @property
    def distance_weighting_power(self):
        return self._distance_weighting_power
    
    @distance_weighting_power.setter
    def distance_weighting_power(self, v):
        self._distance_weighting_power = float(v)

    @property
    def tree(self):
        return self._tree
    
    def interpolate(self, q) -> tuple[float, bool]:
        # search for the nearest quaternions within the search radius
        q = np.asarray(q)
        if q[3] < 0.0:
            q = -q
            
        i, d = self.tree.query_radius(
                        [q], 
                        self.search_radius,
                        return_distance=True)
        
        # since we only accept one point at the time, take the single result
        d = d[0]
        i = i[0]
        
        extrapolated = False
        
        if len(d) == 0:
            # if no sample was within the search radius, get the closest sample
            extrapolated = True
            d, i = self.tree.query([q])
            d = d[0]
            i = i[0]
            
        if len(d) == 1:
            # if only one sample was within the search radius/was extrapolated, return the corresponding value
            v = self.values[i[0]]
            return (v, extrapolated)
        
        v = inverse_distance_weighting(d, self.values[i], power = self.distance_weighting_power)
            
        return (v, extrapolated)
    
    def __call__(self, q) -> tuple[float, bool]:
        return self.interpolate(q)

class RigidTransform:
    def __init__(self, q, t):
        # TODO: assert q is unit quaternion and t is vector of dim 3
        self._q = - q if q[3] < 0.0 else q
        self._t = t

    @property
    def q(self):
        return self._q
    
    @property
    def t(self):
        return self._t

    def as_matrix(self):
        M = np.eye(4)
        M[:3,:3] = self.get_rotation().as_matrix()
        M[:3, 3] = self.t

    @property
    def M(self):
        return self.as_matrix()
    
    def get_rotation(self):
        return Rotation.from_quat(self.q)

    @classmethod
    def from_matrix(cls, M):
        rotation = Rotation.from_matrix(M[:3,:3])
        q = rotation.as_quat()

        t = M[:3, 3]

        return cls(q, t)
    
@numba.njit
def arc_angle_quaternions(x, y):
    return np.arccos(np.minimum(np.abs(np.asarray(x.dot(y))), 1.))

def get_quaternion_metric():
    return DistanceMetric.get_metric('pyfunc', func=arc_angle_quaternions)

def inverse_distance_weighting(distances, values, power=2):
    weights = 1. / np.power(distances + 1e-15, power, dtype=np.float64)
    weight_sum = weights.sum()

    return np.sum((weights * values) / weight_sum)

def load_sampler_from_files(geo_path, coil_matrices_path, **kwargs):
    geo_path = Path(geo_path)
    mat_path = Path(coil_matrices_path)

    value_data = np.array(list(map(float, re.findall("\\{(\d+\.\d+)\\}", geo_path.open('r').read()))))

    mats = np.loadtxt(mat_path).reshape((-1, 4, 4))

    rigids = [RigidTransform.from_matrix(mat) for mat in mats]

    return SE3Interpolator(rigids, value_data, **kwargs)

if __name__ == '__main__':
    pass
   