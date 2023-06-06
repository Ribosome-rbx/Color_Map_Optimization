from os.path import join
import open3d as o3d
import numpy as np
import copy
from utils import *
from sklearn.neighbors import NearestNeighbors

## this file is for color_icp
depth_nbrs = None

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])
    return source_temp + target

def load_depth_and_cam(dir_depth, poses, timings, timestamp, K_parameters_depth):
    global depth_nbrs
    if not depth_nbrs:
        depth_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(timings[:, 1].reshape(-1, 1))

    _, frame_number_depth = depth_nbrs.kneighbors(np.array(float(timestamp) + 0 * (10 ** 4)).reshape(-1, 1))
    frame_number_depth = frame_number_depth[0][0]

    filename_depth = join(dir_depth, '{:06d}.png'.format(frame_number_depth))
    print(f"loading depth image {filename_depth}")
    depth = load_depth(filename_depth)
    

    M_depth = poses[frame_number_depth, 1:].reshape(4, 4).copy()
    K_depth = K_parameters_depth[:9].reshape(3, 3) # intrinsics
    M_depth[:3, 3] *= 1000 
    M_depth = np.dot(axis_transform, np.linalg.inv(M_depth))

    cam_depth = {}
    cam_depth['K_dist'] = K_depth  
    cam_depth['M_dist'] = M_depth 

    return depth, cam_depth

def main():
    dir_seq = '../AnnaTrain'

    dir_depth = join(dir_seq, 'Depth')
    dir_rgb = join(dir_seq, 'Video')

    # Depth
    poses_depth = np.loadtxt(join(dir_depth, 'Pose.txt'))
    timing_depth = np.loadtxt(join(dir_depth, 'Timing.txt'))

    K_parameters_depth = np.loadtxt(join(dir_depth, 'Intrinsics.txt'))
    dist_coeffs = np.array(K_parameters_depth[9:14]).astype('float32')
    
    frame_number_depth_1 = 0
    time_stamp = timing_depth[frame_number_depth_1, 1]
    depth_1, cam_depth_calib_1 = load_depth_and_cam(dir_depth,
                                                poses_depth,
                                                timing_depth,
                                                time_stamp,
                                                K_parameters_depth)
    K_depth_1 = cam_depth_calib_1['K_dist'] # intrinsic
    M_depth_1 = cam_depth_calib_1['M_dist'] # extrinsic
    source = o3d.io.read_point_cloud(f"frames/frame{frame_number_depth_1}.pcd")
    source.estimate_normals()
    # o3d.visualization.draw_geometries([source])

    frame_number_depth_2 = 1
    time_stamp = timing_depth[frame_number_depth_2, 1]
    depth_2, cam_depth_calib_2 = load_depth_and_cam(dir_depth,
                                                poses_depth,
                                                timing_depth,
                                                time_stamp,
                                                K_parameters_depth)
    K_depth_2 = cam_depth_calib_2['K_dist'] # intrinsic
    M_depth_2 = cam_depth_calib_2['M_dist'] # extrinsic
    target = o3d.io.read_point_cloud(f"frames/frame{frame_number_depth_2}.pcd")
    target.estimate_normals()
    # draw initial alignment
    print("1. Identity transformation point cloud registration")
    current_transformation = np.identity(4)
    initial_alignment = draw_registration_result_original_color(source, target, current_transformation)

    # point-to-plane icp
    # baseline
    print("2. Point-to-plane point cloud registration")
    distance_threshold = 0.02
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    point_plane_icp = draw_registration_result_original_color(source, target,
                                            result_icp.transformation)

    # color_icp
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    color_icp = draw_registration_result_original_color(source, target,
                                            result_icp.transformation)

    # save pcd files
    o3d.io.write_point_cloud("initial_alignment.pcd", initial_alignment)
    o3d.io.write_point_cloud("point_plane_icp.pcd", point_plane_icp)
    o3d.io.write_point_cloud("color_icp.pcd", color_icp)


if __name__ == "__main__":
    main()