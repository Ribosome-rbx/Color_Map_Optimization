# Room Reconstruction with Color Map Optimization
[Report](link) | [Video](https://youtube.com/playlist?list=PLUffCQyBEYtbOQg4-66ZrcuNmsX0OXVKv)
![](https://github.com/Ribosome-rbx/Color_Map_Optimization/blob/main/resource/cmo_pipeline.png)

## Environment
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
To reconstruct the room from stratch, please download aforementioned dataset, and run:
```
python ./color_map_optimization.py
```