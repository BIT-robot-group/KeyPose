# KeyPose: Category-Level 6D Object Pose Estimation with Self-Adaptive Keypoints
This is the official implementation of AAAI25 paper "KeyPose: Category-Level 6D Object Pose Estimation with Self-Adaptive Keypoints"

[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/33046/35201)]

## Citation
```
@inproceedings{yu2025keypose,
  title={KeyPose: Category-Level 6D Object Pose Estimation with Self-Adaptive Keypoints},
  author={Yu, Sheng and Zhai, Di-Hua and Xia, Yuanqing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={9},
  pages={9653--9661},
  year={2025}
}
```

## Environment Settings
The code has been tested with

- python 3.7
- torch 1.12
- cuda 11.3

Some dependencies:
```
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```
## Data Processing
### NOCS dataset
- Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
- Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under ```PROJ_DIR/data```and the final file structure is as follows:
```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   ├── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```
### HouseCat6D
Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:
```
HOUSECAT6D_DIR
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```
## Train
### Training on NOCS
```
python train.py --config config/REAL/camera_real.yaml
```
### Training on HouseCat6D
```
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluate 
- Evaluate on NOCS:
```
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```
- Evaluate on HouseCat6D:
```
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```

## Visualization
For visualization, please run
```
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```

## Acknowledgements
Our implementation leverages the code from these works:
- [NOCS](https://github.com/hughw19/NOCS_CVPR2019)
- [AG-Pose](https://github.com/Leeiieeo/AG-Pose)
- [SPD](https://github.com/mentian/object-deformnet)
- [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)
- [DPDN](https://github.com/JiehongLin/Self-DPDN)
- [VI-Net](https://github.com/JiehongLin/VI-Net)
- [HouseCat6D Toolbox](https://github.com/Junggy/HouseCat6D)

We appreciate their generous sharing.
## License
Our code is released under MIT License (see LICENSE file for details).

