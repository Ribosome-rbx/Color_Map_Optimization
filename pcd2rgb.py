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

    return rgb, cam_rgb, frame_number_rgb

def main(interpolate = True):

    dir_seq = '../AnnaTest' # test dataset
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

    frame_number_depth = 508
    time_stamp = timing_rgb[frame_number_depth, 1] # 6.38064766e+17

    depth, cam_depth_calib = load_depth_and_cam(dir_depth,
                                                poses_depth,
                                                timing_depth,
                                                time_stamp,
                                                K_parameters_depth)
    # depth is (288, 320) np.array of depth img
    # cam_depth_calib is dict{3*3, 4*4} extrinsic and intrinsic matrix
    
    K_depth = cam_depth_calib['K_dist']
    # deal with camera distortion
    depth_undistort = cv2.undistort(depth, K_depth, dist_coeffs, None, K_depth)

    rgb, cam_rgb_calib, filename_rgb = load_rgb_and_cam(dir_rgb,
                                        poses_rgb,
                                        timing_rgb,
                                        time_stamp,
                                        K_parameters_rgb)

    M_depth = cam_depth_calib['M_dist'] 
    M_color = np.array(cam_rgb_calib['M_color'])

    pcd = o3d.io.read_point_cloud("./frames/full_scene.pcd") # in the world frame 
    pcd.transform(M_color) # transform into the rgb camera frame
    
    pcd_points = np.array(pcd.points)
    colors = np.array(pcd.colors)[:,::-1]
    # filter on 3d points
    colors = colors[pcd_points[:,2]>0]
    pcd_points = pcd_points[pcd_points[:,2]>0]
    points2d = project_2d(pcd_points, np.array(cam_rgb_calib['K_color']))
    # filter on 2d points
    height, width = rgb.shape[:2]
    mask = np.logical_and(np.logical_and(np.logical_and(points2d[:,0,0]>0, points2d[:,0,0]<width), points2d[:,0,1]>0), points2d[:,0,1]<height)
    pcd_points = pcd_points[mask]
    points2d = points2d[mask]
    colors = colors[mask]

    image = np.zeros((height, width, 3))
    depth = np.zeros((height, width))

    x = []
    y = []
    r = []
    g = []
    b = []
    assert len(pcd_points) == len(points2d) == len(colors)
    for i in range(points2d.shape[0]):
        dx = int(points2d[i, 0, 0])
        dy = int(points2d[i, 0, 1])
        d_i = pcd_points[i, 2]
        if i % 10000 == 0:
            print(f"processing {i}/{points2d.shape[0]}")
        if dx < width and dx > 0 and dy < height and dy > 0:
            if d_i < 0:
                continue
            # Handle 3D occlusions
            if depth[dy, dx] == 0 or d_i < depth[dy, dx]:
                image[dy, dx] = colors[i]
                depth[dy, dx] = d_i

                x.append(dx*1./width)
                y.append(dy*1./height)
                
                r.append(colors[i][0])
                g.append(colors[i][1])
                b.append(colors[i][2])

    
    cv2.imshow("rgb", image)

    if interpolate:
        X = np.array(range(width))*1./width
        Y = np.array(range(height))*1./height
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        r_interp = NearestNDInterpolator(np.array(list(zip(x, y))).reshape(-1, 2), r)
        g_interp = NearestNDInterpolator(np.array(list(zip(x, y))).reshape(-1, 2), g)
        b_interp = NearestNDInterpolator(np.array(list(zip(x, y))).reshape(-1, 2), b)
        R = r_interp(X, Y)
        G = g_interp(X, Y)
        B = b_interp(X, Y)
        image[:,:,0] = R
        image[:,:,1] = G
        image[:,:,2] = B
        
        cv2.imshow("interpolated_rgb", image)
        cv2.imwrite(f"./pcd_shot{filename_rgb}.png", image*255)

    cv2.waitKey(0)



if __name__ == "__main__":
    main(True)