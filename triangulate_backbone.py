import multiprocessing as mp
from triangulation_3d import triangulate_3d, triangulate_3d_return_pose
from itertools import repeat
import pdb
import numpy as np

def triangulate_3d_parallel(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, proj_params_cpu, pool):
    pose_array, loss_array  = zip(*pool.starmap(triangulate_3d, zip(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, repeat(proj_params_cpu))))
    return np.array(pose_array), np.array(loss_array)

def triangulate_3d_return_pose_parallel(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, proj_params_cpu, pool):
    pose_array = pool.starmap(triangulate_3d_return_pose, zip(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, repeat(proj_params_cpu)))
    return np.array(pose_array)

def triangulate_3d_return_eyes_parallel(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, proj_params_cpu, pool):
    result = pool.starmap(triangulate_3d_return_pose, zip(pose_recon_b_list, pose_recon_s1_list, pose_recon_s2_list, repeat(proj_params_cpu)))
    min_loss = 100000000000
    for i in range(0, 4):
        if result.loss < min_loss:
            min_loss = result.loss
            combination = i

