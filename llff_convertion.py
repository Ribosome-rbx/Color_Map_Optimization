import numpy as np
import glob
from os.path import join
import cv2
from scipy.spatial.transform import Rotation

def convert(rgb_dir, depth_dir, intrinsics_path, extrinsics_path, save_dir, testing_hold=5, depth_start=0):

    image_paths = glob.glob(rgb_dir)
    img_idxs = []
    for path in image_paths:
        img_idxs.append(int(path.split("\\")[-1][:-4]))
    img_idxs = np.array(img_idxs)

    # set bounds according to the max and min depth in depth map
    # depth_paths = glob.glob(depth_dir)
    # bnds = np.zeros([len(image_paths),2])
    # for i, path in enumerate(depth_paths):
    #     depth = cv2.imread(path)
    #     bnds[i] = depth.min(), depth.max()
 
    # generate intrinsics
    intrinsic_txt = np.loadtxt(intrinsics_path)
    intrinsics = intrinsic_txt[:9]
    focus = (intrinsics[0] + intrinsics[4]) / 2.0
    W, H = [int(x) for x in intrinsic_txt[-2:]]

    # get all extrinsics
    extrinsics = np.loadtxt(extrinsics_path)[:, 1:].reshape(-1,4,4)[img_idxs]
    axis_transform = np.linalg.inv(np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    poses_bounds = []
    print(len(extrinsics), "images in total")

    for ind in range(len(extrinsics)):
        # convert pose into "camera2world" in Colmap format
        pose = np.linalg.inv(np.dot(axis_transform, np.linalg.inv(extrinsics[ind])))[:3,:]
        
        # get current row of intrinsics and concat with HWF
        cur_pose = np.hstack([pose, np.array([[H], [W], [focus]])])
        cur_pose = np.hstack([cur_pose.reshape(-1), np.array([0.5262,  3.8873])]) # manually set the bounds
        poses_bounds.append(cur_pose)

    # breakpoint()
    print("saving poses_bounds......")
    np.save(join(save_dir, "poses_bounds_inv.npy"), np.array(poses_bounds))
    print("finished")

if __name__ == "__main__":
    convert(rgb_dir="F:/deblur/AnnaTest/Video/*.png",
            depth_dir="none",
            intrinsics_path="F:/deblur/AnnaTest/Video/Intrinsics.txt",
            extrinsics_path="F:/deblur/AnnaTest/Video/Pose.txt",
            save_dir="F:/deblur/AnnaTest",
            testing_hold=5,
            depth_start=0)
