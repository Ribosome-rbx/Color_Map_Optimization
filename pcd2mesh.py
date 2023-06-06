import open3d as o3d
import numpy as np

def pcd2mesh(pcd_name):
    # path = "/home/csvk20/eth/sem_3/MR/align_depth_to_rgb_with_interpolation/optimization/"
    MESH_RESOLUTION_DEPTH = 9
    MESH_DENSITY_QUANTILE_REMOVE = 0.01

    ###### Importing filtered PCD ######
    pcd = o3d.io.read_point_cloud(pcd_name)
    print(len(pcd.points), "points in 3D PC")
    # down sample point_cloud and remove outliers (different scenes may need different parameters)
    # poster # tmp_pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.015)
    # bookshelf # tmp_pcd = pcd.select_by_index(range(0,len(pcd.points),7))
    # bookshelf # pcd, _ = tmp_pcd.remove_radius_outlier(nb_points=20, radius=0.02)
    # for Debug usage
    # pcd2.paint_uniform_color([1., 0., 0.])
    # o3d.visualization.draw_geometries([pcd])


    ###### Estimating Normals for PCD ######
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30) # 100 didn't complete, 30 worked
    # flip normals
    normals = -1 * np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.io.write_point_cloud(path + "pc_with_normals.pcd", pcd)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)


    ###### PCD to Mesh ######
    # pcd = o3d.io.read_point_cloud(path + "pc_with_normals.pcd")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=MESH_RESOLUTION_DEPTH, linear_fit=True, n_threads=1)
    print(len(mesh.vertices), "vertices in mesh")
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


    ###### Density based Mesh filtering ######
    vertices_to_remove = densities < np.quantile(densities, MESH_DENSITY_QUANTILE_REMOVE)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(len(mesh.vertices), "vertices after density based removal")
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


    ###### Filtering mesh to include only region inside pointcloud bbox  ######
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    print(len(mesh.vertices), "vertices after cropping with PCD box")
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)


    ###### Save the mesh as a ply ######
    o3d.io.write_triangle_mesh(pcd_name[:-4]+".ply", mesh)

"""
NOTE for point cloud filtering
6033436 points in 3D PC - Already fit to 5cm grids and filtered
Depth=8, 0.01 quantile: 110413 vertices, 109308 after density based removal, 107644 after cropping PCD box
Depth=9, 0.01 quantile: 422770 vertices, 418542 after density based removal, 416282 after cropping PCD box
"""
if __name__ == "__main__":
    pcd2mesh("./outputs/room.pcd")