# -*- coding: utf-8 -*-
"""
A-priori sampling for E-field guided motion compensation

-) New functions for sampling
The functions within this file enable detailed a-priori sampling which can be combined with
E-field interpolation. Functions are partly based on SimNIBS content.

-) class APSim 
The class is mainly identical to opt_struct.TMSoptimize, just adding new init parameters and a new gridding-function within the _ADM_optimize function.
Besides that, code remains identical to the code published for SimNIBS 4.0.0

Sampling code written by Sarah Grosshagauer.

"""


import gc
import os
import numpy as np
import csv

from simnibs.optimization import ADMlib
from simnibs.simulation import fem
from simnibs.mesh_tools import mesh_io
from simnibs.utils.simnibs_logger import logger
from simnibs import opt_struct

from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from concurrent.futures import ThreadPoolExecutor
from simnibs.optimization import optimize_tms


def _fibonacci_sampling_disk(resolution, radius):
    """
    This function samples a single circular disk with specific radius in the 
    spatial resolution defined in a Fibonacci pattern. 

    Parameters
    ----------
    resolution : int
        distance in mm between individual points.
    radius : int
        total radius of disk in mm.

    Returns
    -------
    numpy array 
        array of sampling coordinates.

    """
    area = radius**2 * np.pi
    nb_samples = round(area/resolution**2)
    golden_angle = np.pi * (3 - np.sqrt(5))  # approximately 2.39996
    points = []

    for i in range(nb_samples):
        theta = i * golden_angle
        r = radius*np.sqrt(i / nb_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x, y))
    
    return np.array(points)

def _rotate_system_Hopf(resolution_angle): 
    """
    This function reads in the respective quaternions depending on the angular resolution selected.
    Quaternion-files were created using C++ code for an SO3 grid provided by Yershova et al. (2010) https://lavalle.pl/software/so3/so3.html
    r refers to the resolution, with r0 presenting the base resolution (72 points) and r1 corresponding to 576 points. 
    
    Parameters
    ----------
    resolution_angle : int
        0: low angular resolution
        1: high angular resolution

    Returns
    -------
    numeric_quats : list of unit quaternions 

    """
    quats = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if resolution_angle == 0:
        with open(os.path.join(script_dir,'data_r0.qua')) as file:
            tsv_file = csv.reader(file, delimiter='\t')
            for row in tsv_file:
                quats.append(row)
    elif resolution_angle == 1:
        with open(os.path.join(script_dir,'data_r1.qua')) as file:
           tsv_file = csv.reader(file, delimiter='\t')
           for row in tsv_file:
               quats.append(row)
    
    numeric_quats = [[float(item) for item in row] for row in quats]
    
    return numeric_quats

def _create_3d_grid(mesh: mesh_io.Msh, target, below_skin_distance, above_skin_distance, radius, resolution_pos, scalp_normals_smoothing_steps=20):
    """
    This function samples a cylindrical volume of interest centered above the skin element closest to position "pos".
    It includes translations as well as multiple different rotations per position. 
    Code is partly based on the function "_create_grid" from optimize_tms
    
    Parameters
    ----------
    mesh : mesh_io.Msh
        Input mesh file.
    target : np array
        [x,y,z] coordinates of target.
    below_skin_distance : int
        sampling below skin.
    above_skin_distance : int
        sampling above skin.
    radius : int
        radius of cylindrical VOI.
    resolution_pos : int
        within- and across-plane spatial resolution of positions.
    scalp_normals_smoothing_steps : int
        number of smoothing steps for scalp surface (Default: 20)

    Returns
    -------
    coil positions, normals for each position, the initial rotation per position, whether its below skin and the layer number

    """
    msh_surf = mesh.crop_mesh(elm_type=2)
    msh_skin = msh_surf.crop_mesh([5, 1005])
    target_skin = msh_skin.find_closest_element(target) #find skin element closest to desired stimulation position
    elm_center = msh_skin.elements_baricenters()[:]
    elm_mask_roi = np.linalg.norm(elm_center - target_skin, axis=1) < 1.2 * radius
    elm_center_zeromean = (
        elm_center[elm_mask_roi] -
        np.mean(elm_center[elm_mask_roi], axis=0)
    )
    msh_roi = msh_skin.crop_mesh(elements=msh_skin.elm.elm_number[elm_mask_roi])
    # tangential plane of target_skin point = "lowest plane of interest"
    u, s, vh = np.linalg.svd(elm_center_zeromean)
    vh = vh.transpose()
    
    # define starting structure
    coords_plane = _fibonacci_sampling_disk(resolution_pos, radius)
    coords_plane = np.dot(coords_plane, vh[:, :2].transpose()) + target_skin 
    normals_roi = msh_roi.triangle_normals(smooth=scalp_normals_smoothing_steps)
    extend_dist = max(below_skin_distance, above_skin_distance)+10*resolution_pos
    q1 = coords_plane + extend_dist * vh[:, 2]  #vh[:,2] is normal direction -> q1, q1 are points used to define line segments extending in opposite directions along normal
    q2 = coords_plane - extend_dist * vh[:, 2] 
    idx, pos = msh_roi.intersect_segment(q1, q2) #find segments through which line between q1 and q2 passes 
    
    # project grid-points to skin surface
    coords_mapped = []
    coords_normals = []
   
    for i, c in enumerate(coords_plane):
        intersections = idx[:, 0] == i   
        if np.any(intersections):
            intersect_pos = pos[intersections]
            intersect_triangles = idx[intersections, 1]
            dist = np.linalg.norm(c[None, :] - intersect_pos, axis=1)
            closest = np.argmin(dist) 
            coords_normals.append(normals_roi[intersect_triangles[closest]])
            coords_mapped.append(intersect_pos[closest]) #plane coord has been mapped to intersecting skin element/position -> all points are on skin
    
    skin_coords = np.array(coords_mapped)
    skin_normals = np.array(coords_normals)
    rvs_skin = [(np.arccos(np.dot(cn, [0,0,1])))*np.cross(cn, [0,0,1]) for cn in skin_normals] #calc rotation vectors
    
    
    #above skin points: get starting normal 
    if np.dot(vh[:,2],np.mean(skin_normals,axis=0)) < 0:
        maxplanevec = -vh[:,2]
    else:
        maxplanevec = vh[:,2]
    maxplane_normals = np.tile(maxplanevec, (len(skin_coords), 1))
  
    
    # calculate rotation vector: determine target orientation/direction as z axis of image
    rax = np.cross(maxplanevec, [0,0,1])
    rax /= np.linalg.norm(rax)
    cos_theta = np.dot(maxplanevec, [0,0,1])
    theta = np.arccos(cos_theta)
    maxplane_rots = [R.from_rotvec(theta*rax) for _ in maxplane_normals]
    skin_rots = [R.from_rotvec(rv) for rv in rvs_skin]
    
    #interpolation between rotations: adjust the normal vector based on the distance to skin using SLERP
    interp_rots = []
    interp_coords = []
    interp_normals = []
    below_skin = []
    layer_position = []
    interp_steps = int(np.ceil((above_skin_distance + below_skin_distance)/resolution_pos))
    
    for cm, cn, r_max, r_skin in zip(skin_coords, skin_normals, maxplane_rots, skin_rots):
        
        layer_positions = np.linspace(-below_skin_distance, above_skin_distance, interp_steps + 1)
        coords_layers = cm + cn * layer_positions[:, np.newaxis]
        below_skin.extend(layer_positions <= 0)
        # Prepare storage for rotations per layer
        rots_per_layer = []
        normals_per_layer = []
        
        # SLERP setup for layers >= 0
        rots = R.from_quat([r_max.as_quat(), r_skin.as_quat()])
        slerp = Slerp([0, interp_steps], rots)
        steps = np.linspace(0, interp_steps, interp_steps + 1)
        
        for pos_idx, pos_val in enumerate(layer_positions):
            if pos_val < 0:
                # Below skin: use skin rotation directly
                rot = r_skin
            else:
                # Above skin or on skin: interpolate
                interp_step = steps[pos_idx]
                rot = slerp(interp_step)
            
            rots_per_layer.append(rot.as_matrix())
            n = np.dot([0,0,1], rot.as_matrix())
            normals_per_layer.append(n)
    
        interp_coords.append(coords_layers)
        interp_rots.append(np.array(rots_per_layer))
        interp_normals.append(np.array(normals_per_layer))
        layer_position.extend(layer_positions)
    
    coords = np.vstack(interp_coords)
    interp_startrots = np.vstack(interp_rots)
    interp_normals = np.vstack(interp_normals)
    
    
    return coords, interp_normals, interp_startrots, below_skin, layer_position

def get_opt_3d_grid_ADM(mesh: mesh_io.Msh, pos, distance=1., radius=20, below_skin_distance=5, above_skin_distance=10,
                resolution_pos=1, resolution_angle=0, scalp_normals_smoothing_steps=20):
    """ Determine the coil positions and orientations for ADM TMS optimization
    Sample a 3D cylindrical grid! 

    Parameters
    ----------
    mesh: simnibs.msh.mesh_io.Msh object
        Simnibs mesh object
    pos: ndarray
        Coordinates (x, y, z) of reference position
    distance: float or None
        Coil distance to skin surface [mm]. (Default: 1.)
    radius: float or None
        Radius of cylindrical region of interest around the reference position, where the
        bruteforce simulations are conducted
    maxdist: float or None
        Maximal distance of the sampling positions from the skin surface, defines height of cylindrical ROI
    resolution_pos: float or None
        Resolution in mm of the coil positions in the region of interest.
   resolution_angle: 0 or 1 (Default 0)
        Determines which Hopf fibration sampling is used for rotation definition (default 0, 72 quaternions, if 1: 576 quaternions)
    scalp_normals_smoothing_steps: int
        number of smoothing steps for scalp surface (Default: 20)

    Returns
    -------
    matsimnibs_list: ndarray of size 4x4xNpos
        list of MATSIMNIBS matrices
    coil_dir: ndarray of size 3xNrot
        list of theoretical coil directions
    valid_coils: ndarray of 4x4xNpos of all valid coil positions (do not intersect with skin, are oriented correctly)
    """
    
    # check if input correct
    if resolution_angle not in (0,1):
        raise ValueError('Invalid value for angle resolution. Only 0 (=72 angles) or 1 (=576) is valid')
    if len(pos) != 3:
        raise ValueError ('Target not defined! Define target in [x,y,z]')
        
    # creates the general sampling
    coords, normals, startrot, below_skin, layer_position = _create_3d_grid(mesh, 
                                                                           pos,
                                                                           below_skin_distance, 
                                                                           above_skin_distance, 
                                                                           radius, 
                                                                           resolution_pos, 
                                                                           scalp_normals_smoothing_steps)
    

    y_seed = np.array([0., 1., 0.])
    matrices = []
    for p, n in zip(coords, normals):
        z = -n
        y = y_seed - (z * y_seed.dot(z))
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        Rm = np.array([x, y, z]).T
        A = np.eye(4)
        A[:3, :3] = Rm
        A[:3, 3] = p
        matrices.append(A)
    
    matrices = np.array(matrices).transpose(1, 2, 0) 
    numeric_quats = _rotate_system_Hopf(resolution_angle)
    npos = matrices.shape[2]
    
    #for plausibility check get mesh of skin! 
    msh_surf = mesh.crop_mesh(elm_type=2)
    msh_skin = msh_surf.crop_mesh([5, 1005]) 
    target_skin = msh_skin.find_closest_element(pos) #find skin element closest to desired stimulation position
    elm_center = msh_skin.elements_baricenters()[:]
    elm_mask_roip = np.linalg.norm(elm_center - target_skin, axis=1) < 50#100
    msh_roip = msh_skin.crop_mesh(elements=msh_skin.elm.elm_number[elm_mask_roip])
    elm_center_roip = msh_roip.elements_baricenters()[:] #all element baricenters
    meshhull = ConvexHull(elm_center_roip)
    verts_ind = meshhull.vertices
    verts = elm_center_roip[verts_ind]
 
    # xr = (-90, 90)
    # yr = (-75, 75)
    # xx,yy = np.meshgrid(np.arange(xr[0],xr[1]+1, 5), np.arange(yr[0], yr[1]+1, 5))
    
    def next_one_index(layer, current_idx):
        for j in range(current_idx + 1, len(layer)):
            if layer[j] > 0:
                return j
        return None  # no 1 found afterwards

        
    def position_valid(ind):
        #print(f'Current coil {ind} out of {npos}')
        coil_matrix=matrices[:,:,ind]
        layer = layer_position[ind]
        angles_degrees = np.arange(-180,180,15)
        angles_radians = np.deg2rad(angles_degrees)
        position = coil_matrix[:3,3]
        startrot = coil_matrix[:3,:3]
        r_start = R.from_matrix(startrot)
        stepwise_rot = []
        vcs = []
        acs = []
        
        for angle in angles_radians:
            rxy = R.from_euler('z', angle)
            matrix_rxy = rxy.as_matrix()
            stepwise_rot.append(matrix_rxy)
    
            
        rotated_matrices = []

        for q in numeric_quats:
            r = R.from_quat(q)
            rotated_r = r * r_start
            rotated_matrix = rotated_r.as_matrix()
            rotated_matrices.append(rotated_matrix)    

        rotated_matrices = np.array(rotated_matrices)
        rotated_matrices = np.insert(rotated_matrices, 0, startrot, axis=0)
        all_zaxes = np.round(rotated_matrices[:,:,2],2)
        dtype = np.dtype((np.void, all_zaxes.dtype.itemsize * all_zaxes.shape[1]))
        struct_array = np.ascontiguousarray(all_zaxes).view(dtype)
        _, unique_indices, counts = np.unique(struct_array, return_index=True, return_counts=True)
        unique_rotations = rotated_matrices[unique_indices]
        
        #check z-direction: pointing towards or away from skin?
        for index in range(unique_rotations.shape[0]):
            matrix = unique_rotations[index, :,:]
            
            if np.dot(startrot[:,2], matrix[:,2]) < 0:
                continue
            if layer<=0:
                #below skin: find an above skin reference to check validity
                ref_ind = next_one_index(layer_position, ind)
                ref_pos = matrices[:3,3,ref_ind]
            else:
                ref_pos = position
            diffs = verts-ref_pos
            distances = np.dot(diffs, matrix[:,2])

            

            # Add rotations around z axis
            if np.all(distances > 0) or np.all(distances < 0): #make sure that coil does not go through head
                vc = np.column_stack([matrix, position])
                vc = np.vstack([vc, [0,0,0,1]])#
                vcs.append(vc)
                #if valid coil -> rotate around z axis in steps    
                # for rxy_mat in stepwise_rot:
                #       rxymatrix = np.dot(rxy_mat, matrix[:3,:3].T).T
                #       vc = np.column_stack([rxymatrix, position])
                #       vc = np.vstack([vc, [0,0,0,1]])
                #       acs.append(vc)
            else:
                continue
            
        
                
        return vcs

    valid_coils = []
    #parallelize the checking of validity for coils
    with ThreadPoolExecutor(max_workers=10) as executor:
         results = list(executor.map(position_valid, range(npos)))
         for coils in results:
             valid_coils.extend(coils)

    nvcoils = len(valid_coils)
    valid_coils = np.array(valid_coils)
    
    print(f'total number of valid coil positions going into ADM: {nvcoils} ')
    rotations = np.array(optimize_tms._rotate_system(np.eye(3), (-180,180), 15))[:,1].T
    
    all_coils = []
    stepwise_rot = []
    angles_degrees = np.arange(-180,180,15)
    angles_radians = np.deg2rad(angles_degrees)
    
    for angle in angles_radians:
        rxy = R.from_euler('z', angle)
        matrix_rxy = rxy.as_matrix()
        stepwise_rot.append(matrix_rxy)
        
    for coil in valid_coils:
        r_base = coil[:3,:3]
        t_base = coil[:3,3]
        
        for rxy_mat in stepwise_rot:
               r_new = np.dot(rxy_mat, r_base.T).T
               A_new = np.eye(4)
               A_new[:3,:3] = r_new
               A_new[:3,3] = t_base
               all_coils.append(A_new)
        
    
    
    return valid_coils, rotations, all_coils




class APSim(opt_struct.TMSoptimize):
    """
    Class is based on TMSoptimize class, just some minor functional updates to integrate 
    the new sampling.
    """
    def __init__(self, matlab_struct=None):
        self.angle_sampling = 0
        self.above_skin_distance = 10
        self.below_skin_distance = 2
        self.spatial_resolution = 2
        self.search_radius = 10
        
        super().__init__(matlab_struct=None)
        
    def _ADM_optimize(self, cond_field, target_region): 
        """
        overwrites the initital _ADM_optimize function
        Changes:
            -) new grid creation
            -) just one rotation per matrix, as matrices are already rotated
            -) coil matrices are saved separately
       """
        
        coil_matrices,rotations, allcoils = get_opt_3d_grid_ADM( 
            self.mesh, self.centre,
            below_skin_distance=self.below_skin_distance, 
            radius=self.search_radius,
            above_skin_distance=self.above_skin_distance,
            resolution_pos=self.spatial_resolution,
            resolution_angle=self.angle_sampling,
 	        scalp_normals_smoothing_steps=self.scalp_normals_smoothing_steps
        )
        # save all coil positions to file
        flatcoils = np.vstack(allcoils)
        np.savetxt(f'{self.pathfem}/coil_matrices.txt', flatcoils)
        coil_matrices = coil_matrices.transpose(1,2,0)

        ## from here everything remains unchanged to opt_struct.TMSoptimize._ADM_optimize!
        # transform coil matrix to meters
        coil_matrices[:3, 3, :] *= 1e-3
        
        baricenters = self.mesh.elements_baricenters()

        th = self.mesh.elm.elm_type == 4
        if not self.fnamecoil.endswith('.ccd'):
            raise ValueError('ADM optimization is only possible with ".ccd" coil files')
        if not np.all(th[target_region - 1]):
            raise ValueError('Target region must contain only tetrahedra')
        ccd_file = np.loadtxt(self.fnamecoil, skiprows=2)
        dipoles, moments = ccd_file[:, 0:3], ccd_file[:, 3:]
        # Run dipole simulations
        S = fem.FEMSystem.electric_dipole(
            self.mesh, cond_field,
            solver_options=self.solver_options
        )

        vols = self.mesh.elements_volumes_and_areas()

        def calc_dipole_J(dipole_dir):
            Jp = mesh_io.ElementData(np.zeros((self.mesh.elm.nr, 3), dtype=float))
            Jp[target_region] = dipole_dir
            b = S.assemble_electric_dipole_rhs(Jp)
            v = mesh_io.NodeData(S.solve(b), mesh=self.mesh)
            m = fem.calc_fields(v, 'J', cond=cond_field)
            J = m.field['J'][:] + Jp[:]
            J /= np.sum(vols[target_region])
            return J

        if self.target_direction is None:
            J_x = calc_dipole_J([1, 0, 0]) * vols[:, None]
            J_y = calc_dipole_J([0, 1, 0]) * vols[:, None]
            J_z = calc_dipole_J([0, 0, 1]) * vols[:, None]
            del S
            gc.collect()
            logger.info('Running ADM')
            # Notice that there is an uknown scale factor
            # as we need to know the pulse angular frequency
            # \Omega and amplitude A
            E_roi = ADMlib.ADMmag(
                baricenters[th].T * 1e-3,
                J_x[th].T, J_y[th].T, J_z[th].T,
                dipoles.T, moments.T,  # .ccd file is already in SI units
                coil_matrices, rotations
            ) * self.didt
        else:
            if len(self.target_direction) != 3:
                raise ValueError('target direction should have 3 elements!')
            direction = np.array(self.target_direction, dtype=float)
            direction /= np.linalg.norm(direction)
            J_d = calc_dipole_J(direction) * vols[:, None]
            E_roi = ADMlib.ADM(
                baricenters[th].T * 1e-3,
                J_d[th].T,
                dipoles.T, moments.T,  # .ccd file is already in SI units
                coil_matrices, rotations
            ) * self.didt

        z = np.array([0., 0., 1.])
        pos_matrices = []
        coil_matrices[:3, 3, :] *= 1e3
        for cm in coil_matrices.transpose(2, 0, 1):
            for r in rotations.T:
                R = np.eye(4)
                R[:3, :3] = np.array([np.cross(r, z), r, z]).T
                pos_matrices.append(cm.dot(R))

        return E_roi.T.reshape(-1), pos_matrices






    




