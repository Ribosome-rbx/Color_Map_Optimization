import open3d as o3d
from os.path import join, exists
from os import makedirs, listdir
import numpy as np
import cv2
from utils import *
from pcd2mesh import pcd2mesh
from sklearn.neighbors import NearestNeighbors

depth_nbrs = None
rgb_nbrs = None

def depth_to_colormapjet(depth):
    depth_color = depth.copy()
    min_d, max_d = np.min(depth_color), np.max(depth_color)
    depth_color = depth_color * 255. / (max_d - min_d) 
    depth_color = np.uint8(depth_color)
    depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
    return depth_color

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
    # M_depth[:3, 3] *= 1000 
    M_depth = np.dot(axis_transform, np.linalg.inv(M_depth))

    cam_depth = {}
    cam_depth['K_dist'] = K_depth  
    cam_depth['M_dist'] = M_depth 

    return depth, cam_depth

def load_rgb_and_cam(dir_rgb, poses_rgb, timing_rgb, time_stamp, K_parameters_rgb):
    global rgb_nbrs
    if not rgb_nbrs:
        rgb_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
            timing_rgb[:, 1].reshape(-1, 1))

    _, frame_number_rgb = rgb_nbrs.kneighbors(
        np.array(float(time_stamp) + 0 * (10 ** 4)).reshape(-1, 1))
    frame_number_rgb = frame_number_rgb[0][0]

    filename_rgb = join(dir_rgb, '{:06d}.png'.format(frame_number_rgb))
    print(f"loading rgb image {filename_rgb}")
    rgb = cv2.imread(filename_rgb)

    K_color = K_parameters_rgb[:9].reshape(3, 3)
    M_color = poses_rgb[frame_number_rgb, 1:].reshape(4, 4).copy() 
    cam_rgb = {}
    cam_rgb['M_origin'] = np.dot(axis_transform, np.linalg.inv(M_color))
    # M_color[:3, 3] *= 1000
    M_color = np.dot(axis_transform, np.linalg.inv(M_color))

    
    cam_rgb['K_color'] = K_color
    cam_rgb['M_color'] = M_color

    return rgb, cam_rgb

def create_point_cloud_from_depth(depth, cam_depth, remove_outlier=True, remove_close_to_cam=300):
    '''
    output: point cloud in the depth frame
    '''
    K_depth = cam_depth['K_dist']

    img2d_converted = depthConversion(depth, K_depth[0][0], K_depth[0][2], K_depth[1][2]) # point depth to plane depth, basically, undistortion
    # img2d_converted_color = depth_to_colormapjet(img2d_converted) # plane depth color map jet
    # cv2.imshow('img2d_converted_color', img2d_converted_color)
    points = generatepointcloud(img2d_converted, K_depth[0][0], K_depth[1][1], K_depth[0][2], K_depth[1][2]) # in the depth coor
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if remove_outlier:
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=50)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=4.0)

    if remove_close_to_cam > 0:
        center = np.array([0, 0, 0])
        R = np.eye(3)
        extent = np.array([remove_close_to_cam, remove_close_to_cam, remove_close_to_cam])
        bb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        close_points_indices = bb.get_point_indices_within_bounding_box(pcd.points)
        pcd = pcd.select_by_index(close_points_indices, invert=True) #select outside points
    return pcd

def stitch_pcd(source, target, transformation):
    source.transform(transformation)
    # o3d.visualization.draw_geometries([source, target])
    return source + target

def main():

    dir_seq = '../AnnaTrain'
    dir_depth = join(dir_seq, 'Depth')
    dir_rgb = join(dir_seq, 'Video')

    ####### loading from dataset ######
    # Depth
    poses_depth = np.loadtxt(join(dir_depth, 'Pose.txt'))
    timing_depth = np.loadtxt(join(dir_depth, 'Timing.txt'))
    K_parameters_depth = np.loadtxt(join(dir_depth, 'Intrinsics.txt'))
    dist_coeffs = np.array(K_parameters_depth[9:14]).astype('float32')
    w_depth, h_depth = [int(x) for x in K_parameters_depth[-2:]]

    # RGB
    poses_rgb = np.loadtxt(join(dir_rgb, 'Pose.txt'))
    timing_rgb = np.loadtxt(join(dir_rgb, 'Timing.txt'))
    K_parameters_rgb = np.loadtxt(join(dir_rgb, 'Intrinsics.txt'))
    w_color, h_color = [int(x) for x in K_parameters_rgb[-2:]]

    ###### prepare for color map optimization ######
    # we use a continuous frames of videos as input for Color Map Optimization
    # initization of containers
    debug_mode = True
    start_frame = 0            # the first frame of video
    total_depth_frames = 1    # total number of frames to be processed
    rgbd_images = []           # container for rgbd images
    camera_parameters = []     # container for camera intrinsic and extrinsic parameters
    whole_pcd = None           # Collection of while point clouds
    pcd_list = []
    ################ select images ###################
    for frame_number_depth in range(start_frame, start_frame+total_depth_frames):
        time_stamp = timing_rgb[frame_number_depth, 1]
        # find the nearest depth frame
        depth, cam_depth_calib = load_depth_and_cam(dir_depth,
                                                    poses_depth,
                                                    timing_depth,
                                                    time_stamp,
                                                    K_parameters_depth)
        K_depth = cam_depth_calib['K_dist']
        depth_undistort = cv2.undistort(depth, K_depth, dist_coeffs, None, K_depth)
        if debug_mode:
            # visualize depth & undistorted depth
            cv2.imshow('depth_undistort', depth_to_colormapjet(depth_undistort))
            cv2.imshow('depth', depth_to_colormapjet(depth))
        # find the nearest rgb frame
        rgb, cam_rgb_calib = load_rgb_and_cam(dir_rgb,
                                            poses_rgb,
                                            timing_rgb,
                                            time_stamp,
                                            K_parameters_rgb)
        
        # build and store point cloud
        # NOTE: pcd_colored in the world frame, pcd still in the depth frame
        pcd = create_point_cloud_from_depth(depth_undistort, cam_depth_calib, remove_outlier=True, remove_close_to_cam=1500) # depth frame
        # pcd_colored = get_colored_pcd(pcd, rgb, cam_rgb_calib, cam_depth_calib)
        # o3d.visualization.draw_geometries([pcd_colored])

        # Aligned Depth to be the same size with rgb image
        depth_aligned = map_depth_to_rgb(pcd, rgb, cam_rgb_calib, cam_depth_calib, reference='rgb')
        if debug_mode:
            # visualize aligned depth
            cv2.imshow('aligned depth', depth_to_colormapjet(depth_aligned))

        # store RGBD of this frame
        depth = o3d.geometry.Image(depth_aligned.astype(np.uint16))
        color = o3d.geometry.Image(rgb[:,:,::-1].astype(np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=1000, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)
        if debug_mode:
            cv2.imshow('rgb_color', rgb)
            cv2.imshow('rgb_depth', depth_to_colormapjet(np.array(rgbd_image.depth)))
        
        # store camera Intrinsic and Extrinsic parameters
        height, width = rgb.shape[:2]
        fx, fy, cx, cy = [cam_rgb_calib['K_color'][0,0], cam_rgb_calib['K_color'][1,1], cam_rgb_calib['K_color'][0,2], cam_rgb_calib['K_color'][1,2]]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        extrinsic = cam_rgb_calib['M_color']
        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = intrinsic
        camera.extrinsic = extrinsic
        camera_parameters.append(camera)

        # Stitch point clouds
        # use RGBD to generate new pcd, which can be correctly rendered in color_map_optimization
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
        # transform pcd from depth frame to the world frame
        pcd.transform(np.linalg.inv(cam_rgb_calib['M_color']))
        # # remove outliers (may need further parameter tuning)
        # pcd, _ = pcd.remove_radius_outlier(nb_points=200, radius=0.04)
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=200,std_ratio=4.0)

        if debug_mode:
            # pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
            # if not exists('./outputs'): makedirs('./outputs')
            # pcd_path = "outputs/frame" + str(frame_number_depth) + ".pcd"
            # o3d.io.write_point_cloud(pcd_path, pcd_down)
            o3d.visualization.draw_geometries([pcd])
        
        # pcd_list.append(pcd)
        if whole_pcd is None: whole_pcd = pcd
        else: whole_pcd += pcd
        #################### ICP stitching ###########################
        # if whole_pcd is None: 
        #     whole_pcd = pcd
        # else:
        #     pcd.estimate_normals()
        #     whole_pcd.estimate_normals()
        #     result_icp = o3d.pipelines.registration.registration_icp(pcd, whole_pcd, 0.1, np.identity(4),
        #                                                 o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #     whole_pcd = stitch_pcd(pcd, whole_pcd, result_icp.transformation)

    #################### downsample and remove outliers ###########################
    # whole_pcd = whole_pcd.select_by_index(range(0,len(whole_pcd.points),10))
    # whole_pcd, _ = whole_pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=8.0)

    # visualize and store the whole scene
    o3d.visualization.draw_geometries([whole_pcd])
    pcd_down = whole_pcd.voxel_down_sample(voxel_size=0.005)
    if not exists('./outputs'): makedirs('./outputs')
    write_path = f"outputs/scene{start_frame}-{start_frame+total_depth_frames}"
    o3d.io.write_point_cloud(write_path+".pcd", pcd_down)
    pcd2mesh(write_path+".pcd")

    # RUN color map optimization
    camera_traj = o3d.camera.PinholeCameraTrajectory()
    camera_traj.parameters = camera_parameters
    mesh = o3d.io.read_triangle_mesh(write_path+".ply")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, camera_traj = o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, camera_traj,
            o3d.pipelines.color_map.RigidOptimizerOption(
                maximum_iteration=100,
                maximum_allowable_depth=2.5 * 1000000,
                depth_threshold_for_visibility_check=0.03,
                depth_threshold_for_discontinuity_check=0.1*1000000
                ))
    print("show with cmop")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # store optimized mesh
    o3d.io.write_triangle_mesh(f"{write_path}_cmop.ply", mesh)

if __name__ == "__main__":
    main()
    