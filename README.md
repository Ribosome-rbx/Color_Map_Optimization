# Color Map Optimization

## Dataset
Dataset 1 (Gowtham Room)
| [Training set](https://drive.google.com/file/d/1SDoMu82SKCXeIN0Jx5hPdFrSIh5NdLd5/view?usp=share_link) | [Testing set](https://drive.google.com/file/d/1ch8T6YyFJjmdYxV6ZIc7_MvTgNo4QHTE/view?usp=share_link) |


Dataset 2 (Anna Room)
| [Training set](https://drive.google.com/file/d/1ejI0oGDvouf8kSXmtE2YtDnUD5xQ9CJ0/view?usp=share_link) | [Testing set](https://drive.google.com/file/d/1GM86hnksWmncO_VzHofgo8cX0_KKEzvO/view?usp=share_link) |

Directory Structure
```
..
├── AnnaTrain
├── AnnaTest
├── GowthamTrain
├── GowthamTest
└── boxiang
     ├── color_map_optimization.py
     ├── pcd_stitching.py
     ├── pcd2mesh.py
     └── utils.py 
```

## Run
```
cd ./boxiang
python ./color_map_optimization.py
```