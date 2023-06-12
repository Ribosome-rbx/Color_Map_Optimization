# Room Reconstruction with Color Map Optimization
[Report](link) | [Video](https://youtube.com/playlist?list=PLUffCQyBEYtbOQg4-66ZrcuNmsX0OXVKv)

We implement the traditional reconstruction pipelne(point cloud + mesh), and use Color Map Optimization to draw clear textures on mesh.
![](https://github.com/Ribosome-rbx/Color_Map_Optimization/blob/main/resource/cmo_pipeline.png)

## Environment
All the dependencies used in this repo are listed as below:
* Open3d 0.16.0
* Opencv 4.7.0
* Sklearn 1.2.2
* Numpy 1.23.3
* Scipy 1.9.3
* Matplotlib 3.6.1
* Png
* Glob

## HoloLens Room Dataset
This dataset is captured by Hololens2 and consists of two video recordings of two different room scenes. Each capture contains thousands of RGB video frames in 1280×720, monocular depth frames in a lower capturing frequency, the intrinsic parameters of the camera, and the corresponding camera poses and timestamp for each RGB frame. For the first capture ([AnnaTrain](https://drive.google.com/file/d/1ejI0oGDvouf8kSXmtE2YtDnUD5xQ9CJ0/view)/[GowthamTrain](https://drive.google.com/file/d/1SDoMu82SKCXeIN0Jx5hPdFrSIh5NdLd5/view)), the HoloLens has a relatively slow movement, which results in a dataset containing less motion blur. While the second capture (named [AnnaTest](https://drive.google.com/file/d/1GM86hnksWmncO_VzHofgo8cX0_KKEzvO/view)/[GowthamTest](https://drive.google.com/file/d/1ch8T6YyFJjmdYxV6ZIc7_MvTgNo4QHTE/view)) contains more motion blur.


Directory Structure
```
..
├── AnnaTrain
├── AnnaTest
├── GowthamTrain
├── GowthamTest
└── Color_Map_Optimization
     ├── color_map_optimization.py
     ├── ...
```

## Run
![](https://github.com/Ribosome-rbx/Color_Map_Optimization/blob/main/resource/AnnaRoom.gif)

We have a well reconstructed room mesh(.obj) in `./resource`. To visulize it, use:
```
cd ./Color_Map_Optimization && python ./Visualization_rotate.py
```
To reconstruct the room from scatch, please download aforementioned dataset, and run:
```
python ./color_map_optimization.py
```
## Illustration of each file
* **color_map_optimization.py:** main implementation. Including dataloader, aligning RGB and depth images, point_cloud and mesh reconstruction, and color map optimization.
* **filter_blurry_images.py:** select blurry images from the mixture of clear and blurry images.
* **llff_convertion.py:** generate poses_bounds.npy for NeRF-based methods.
* **mesh2rgb.py:** input camera poses to render rgb images with pre-built models.
* **metrices_compute.py:** run evaluation metrices to output images of NeRF-based methods.
* **pcd_stitching.py:** use ICP for point cloud stitching
* **pcd2mesh.py:** implement poisson surface reconstruction to recover mesh from point clouds
* **pcd2rgb.py:** input camera poses to render rgb images with pre-built point clouds.
* **Visualization_rotate.py** visualize (point cloud/mesh)files in a rotating form.
* **visualization.py** dependencies for visualization.
