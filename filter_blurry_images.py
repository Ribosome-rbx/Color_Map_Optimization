import glob
import open3d as o3d
from os.path import join, exists
from os import makedirs, listdir
import numpy as np
import cv2
from utils import *
from pcd2mesh import pcd2mesh
from sklearn.neighbors import NearestNeighbors
from visualization import VisOpen3D

depth_nbrs = None
rgb_nbrs = None

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

def get_camera_pose(dir_seq = '../AnnaTest', frame_number_depth = 1):
    # dir_seq = '../AnnaTest' # test dataset
    dir_depth = join(dir_seq, 'Depth')
    dir_rgb = join(dir_seq, 'Video')

    ####### loading from dataset ######
    poses_rgb = np.loadtxt(join(dir_rgb, 'Pose.txt'))
    timing_rgb = np.loadtxt(join(dir_rgb, 'Timing.txt'))
    K_parameters_rgb = np.loadtxt(join(dir_rgb, 'Intrinsics.txt'))

    time_stamp = timing_rgb[frame_number_depth, 1]

    rgb, cam_rgb_calib, filename_rgb = load_rgb_and_cam(dir_rgb,
                                        poses_rgb,
                                        timing_rgb,
                                        time_stamp,
                                        K_parameters_rgb)

    # Set the camera parameters
    height, width = rgb.shape[:2]
    fx, fy, cx, cy = [cam_rgb_calib['K_color'][0,0], cam_rgb_calib['K_color'][1,1], width/2-0.5, height/2-0.5]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrinsic = cam_rgb_calib['M_color']
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = intrinsic
    camera.extrinsic = extrinsic
    if not exists(dir_seq+'/poses'): makedirs(dir_seq+'./poses')
    o3d.io.write_pinhole_camera_parameters(dir_seq+"/poses" + "/%06d.json" %frame_number_depth, camera)
    return intrinsic.intrinsic_matrix, extrinsic

def main():
    w = 1280
    h = 720
    threshold = 16
    data_root = '../AnnaTest'
    frame_num= 1
    mesh = o3d.io.read_triangle_mesh('./frames/room_cmop.ply')
    # pcd = o3d.io.read_point_cloud('./frames_backup/full_scene.pcd')
    # total_frames = len(listdir(data_root+"/Video")) - 4
    image_paths = glob.glob(data_root+"/Video/*.png")
    total_frames = len(image_paths)

    for frame_num in range(total_frames):

        path = image_paths[frame_num]
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm >= threshold:
            cv2.imwrite(data_root+f"/synthesized/{frame_num:06d}.png", image)
            continue

        # if frame_num %100 == 0: print("processing" + "/"+ data_root.split("/")[-1] + f"{frame_num}/{total_frames}")
        print("processing" + "/"+ data_root.split("/")[-1] + f"{frame_num}/{total_frames}")
        intrinsic, extrinsic = get_camera_pose(data_root, frame_num) # get i_th depth map in test set

        # create window
        window_visible = False
        vis = VisOpen3D(width=w, height=h, visible=window_visible)

        # point cloud
        vis.add_geometry(mesh)

        # update view
        vis.update_view_point(intrinsic, extrinsic)

        # save view point to file
        # vis.save_view_point("view_point.json")
        # vis.load_view_point("view_point.json")

        # capture images
        # depth = vis.capture_depth_float_buffer(show=True)
        # image = vis.capture_screen_float_buffer(show=True)

        # save to file
        rgb_store = data_root+'/rendered/img'
        depth_store = data_root+'/rendered/img_depth'
        if not exists(rgb_store): makedirs(rgb_store)
        if not exists(depth_store): makedirs(depth_store)
        vis.capture_screen_image(data_root+f"/synthesized/{frame_num:06d}.png")
        # vis.capture_depth_image(depth_store + "/%06d" %frame_num + ".png")

        # # draw camera
        if window_visible:
            # vis.load_view_point("view_point.json")
            vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
            # vis.update_view_point(intrinsic, extrinsic)


        if window_visible:
            # vis.load_view_point("view_point.json")
            vis.run()

        del vis


if __name__ == "__main__":
    '''
    convert images into video:
    ffmpeg -framerate 25  -i 'rendered/%06d.png' -c:v libx264 -pix_fmt yuv420p rendered.mp4
    '''
    main()
