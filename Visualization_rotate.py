
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_path = os.path.join(os.path.dirname(pyexample_path), 'test_data')


def custom_draw_geometry_with_rotation(pcd, aps = 5.0):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(aps, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

if __name__ == "__main__":
    sample_data = o3d.data.DemoCustomVisualization()
    pcd_flipped = o3d.io.read_point_cloud("./final/room.pcd")
    mesh = o3d.io.read_triangle_mesh('./frames/room_cmop.ply')

    custom_draw_geometry_with_rotation(pcd_flipped)
    custom_draw_geometry_with_rotation(mesh)
