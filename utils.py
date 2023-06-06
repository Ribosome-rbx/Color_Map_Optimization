import numpy as np
import open3d as o3d
import png
import cv2
from scipy.interpolate import NearestNDInterpolator

try:
    from itertools import imap
except ImportError:
    # Python 3...
    imap = map




def get_handpose_connectivity():
    # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
    return [
        [0, 1],

        # Thumb
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],

        # Index
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],

        # Middle
        [1, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Ring
        [1, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        # Pinky
        [1, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [24, 25]
    ]


def read_hand_pose_txt_new(hand_path, is_stereokit=False):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            if is_stereokit:
                line_data = list(map(float, line.split('\t')))

                if line_data[3] == 0.0:  # if hand pose does not exist.
                    # add empty hand location
                    hand_array.append(line_data[:4]+[0]*3*26)
                elif line_data[3] == 1.0:  # if hand pose does exist.
                    line_data_reshape = np.reshape(
                        line_data[4:], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                    line_data_xyz = []
                    for line_data_reshape_elem in line_data_reshape:
                        # To get translation of the hand joints
                        location = np.dot(line_data_reshape_elem,
                                        np.array([[0, 0, 0, 1]]).T)
                        line_data_xyz.append(location[:3].T[0])

                    line_data_xyz = np.array(line_data_xyz).T
                    hand = line_data[:4]
                    hand.extend(line_data_xyz.reshape(-1))
                    hand_array.append(hand)
            else:
                line_data = list(map(float, line.split('\t')))
                line_data_reshape = np.reshape(
                    line_data[2:-52], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                line_data_xyz = []
                for line_data_reshape_elem in line_data_reshape:
                    # To get translation of the hand joints
                    location = np.dot(line_data_reshape_elem,
                                    np.array([[0, 0, 0, 1]]).T)
                    line_data_xyz.append(location[:3].T[0])

                line_data_xyz = np.array(line_data_xyz).T
                hand = line_data[:4]
                hand.extend(line_data_xyz.reshape(-1))
                hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array


def read_hand_pose_txt(hand_path):
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            line_data = list(map(float, line.split('\t')))
            if line_data[3] == 0.0:  # if hand pose does not exist.
                # add empty hand location
                hand_array.append(line_data[:4]+[0]*3*26)
            elif line_data[3] == 1.0:  # if hand pose does exist.
                line_data_reshape = np.reshape(
                    line_data[4:], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                line_data_xyz = []
                for line_data_reshape_elem in line_data_reshape:
                    # To get translation of the hand joints
                    location = np.dot(line_data_reshape_elem,
                                      np.array([[0, 0, 0, 1]]).T)
                    line_data_xyz.append(location[:3].T[0])

                line_data_xyz = np.array(line_data_xyz).T
                hand = line_data[:4]
                hand.extend(line_data_xyz.reshape(-1))
                hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array


def depthConversion(PointDepth, f, cx, cy):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]

    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    distance_from_center = ((rows - cy)**2 + (columns - cx)**2) ** 0.5
    plane_depth = PointDepth / (1 + (distance_from_center / f)**2) ** 0.5

    return plane_depth

axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


def generatepointcloud(depth, Fx, Fy, Cx, Cy):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    depth_scale = 1
    z = depth * depth_scale

    x = z * (c - Cx) / Fx
    y = z * (r - Cy) / Fy
    points = np.dstack((x, y, z))
    points = points.reshape(-1, 3)
    points = points[~np.all(points == 0, axis=1)]
    return points


def load_depth(path):
    # PyPNG library is used since it allows to save 16-bit PNG
    r = png.Reader(filename=path)
    im = np.vstack(imap(np.uint16, r.asDirect()[2])).astype(np.float32)#[:, ::-1]
    return im


def project_2d(points3d, K, R=np.eye(3), t=np.zeros(3), dist_coeffs=np.zeros(5,)):
    pts2d, _ = cv2.projectPoints(points3d, R, t, K, dist_coeffs) # project onto image plane
    return pts2d


def project_2d_kinect(points3d, cam_calib_data):
    R = np.array(cam_calib_data['M_color'])[:3, :3] # rotation 
    t = np.array(cam_calib_data['M_color'])[:3, 3] # linear transformation
    K = np.array(cam_calib_data['K_color']) # intrinsics
    pts2d = project_2d(points3d, K, R, t)

    return pts2d

def project_2d_kinect_depth(points3d, cam_calib_data):
    R = np.array(cam_calib_data['M_dist'])[:3, :3] # rotation 
    t = np.array(cam_calib_data['M_dist'])[:3, 3] # linear transformation
    K = np.array(cam_calib_data['K_dist']) # intrinsics
    pts2d = project_2d(points3d, K, R, t)

    return pts2d

def get_colored_pcd(pcd, rgb, cam_calib_rgb, cam_calib_depth):
    M_depth = cam_calib_depth['M_dist'] 
    pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
    pcd.transform(np.linalg.inv(M_depth)) # now, pcd goes to world frame
    points2d = project_2d_kinect(np.array(pcd.points), cam_calib_rgb) # points are projected onto image plane
    
    height, width = rgb.shape[:2]

    colors = np.zeros_like(np.array(pcd.points), dtype='float32')
    indices_with_color = []
    print(f'width: {width}')
    print(f'height: {height}')
    width_reduction = 0
    height_reduction = 0
    for i in range(points2d.shape[0]):
        # integer accumulates error
        dx = int(points2d[i, 0, 0])
        dy = int(points2d[i, 0, 1])

        if dx < (width - width_reduction) and dx > width_reduction and dy < (height-height_reduction) and dy > height_reduction:
            colors[i, :] = rgb[dy, dx, ::-1]/255.
            indices_with_color.append(i)
    indices_with_color = np.array(indices_with_color)
    # get index where all colors are zero
    # mask = np.where(np.amax(colors, axis=1)>0)
    pcd = pcd.select_by_index(indices_with_color)
    pcd_colored = o3d.geometry.PointCloud(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors[indices_with_color])
    return pcd_colored

def get_resized_depth_map(pcd, rgb, depth, cam_calib_rgb, cam_calib_depth):
    # convert pcd to world frame
    M_depth = cam_calib_depth['M_dist'] 
    pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
    pcd.transform(np.linalg.inv(M_depth)) # now, pcd goes to world frame

    # project pcd back into rgb and depth 2d plane
    rgb_2d = project_2d_kinect(np.array(pcd.points), cam_calib_rgb) # coor on both rgb and revised depth plane
    depth_2d = project_2d_kinect_depth(np.array(pcd.points), cam_calib_depth) # coor on origional depth plane
    
    # init revised depth plane
    depth_values = np.zeros(rgb.shape[:2])
    
    for i in range(rgb_2d.shape[0]):
        # integer accumulates error
        rgb_x = int(rgb_2d[i, 0, 0])
        rgb_y = int(rgb_2d[i, 0, 1])
        height, width = rgb.shape[:2]
        if rgb_x < 0 or rgb_x >= width or rgb_y < 0 or rgb_y >= height: continue
        depth_x = int(depth_2d[i, 0, 0])
        depth_y = int(depth_2d[i, 0, 1])
        height, width = depth.shape[:2]
        if depth_x < 0 or depth_x >= width or depth_y < 0 or depth_y >= height: continue
        depth_values[rgb_y, rgb_x] = depth[depth_y, depth_x]
    
    return depth_values # (720, 1280, 3)




def map_depth_to_rgb(pcd, rgb, cam_calib_rgb, cam_calib_depth, reference='depth', interpolate=True):
    M_depth = cam_calib_depth['M_dist'] 
    M_color = np.array(cam_calib_rgb['M_color'])
    pcd_points_depth = np.array(pcd.points) # in the depth frame
    
    # transform pcd from depth to world frame
    pcd.transform(np.linalg.inv(M_depth))
    pcd.transform(M_color)
    
    pcd_points = np.array(pcd.points)
    points2d = project_2d(pcd_points, np.array(cam_calib_rgb['K_color']))
    
    height, width = rgb.shape[:2]

    depth = np.zeros((height, width))

    x = []
    y = []
    z = []
    for i in range(points2d.shape[0]):
        dx = int(points2d[i, 0, 0])
        dy = int(points2d[i, 0, 1])

        if reference == 'depth':
            d_i = pcd_points_depth[i, 2]
        elif reference == 'rgb':
            d_i = pcd_points[i, 2]
        else:
            assert False, 'unknown reference'


        if dx < width and dx > 0 and dy < height and dy > 0:
            if d_i < 0:
                continue
            # Handle 3D occlusions
            if depth[dy, dx] == 0 or d_i < depth[dy, dx]:
                depth[dy, dx] = d_i

                x.append(dx*1./width)
                y.append(dy*1./height)
                z.append(d_i)

    
    if interpolate:
        X = np.array(range(width))*1./width
        Y = np.array(range(height))*1./height
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        interp = NearestNDInterpolator(np.array(list(zip(x, y))).reshape(-1, 2), z)
        Z = interp(X, Y)

        mask = np.zeros_like(depth).astype('uint8')
        indices = np.where(depth > 0)
        mask[indices] = 255
        mask = cv2.blur(mask, (5, 5))
        indices = np.where(mask > 0)
        depth[indices] = Z[indices]

    return depth



