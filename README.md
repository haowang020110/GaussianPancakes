# Reproduce the Gaussian Pancakes using gt pose and depth as an alternative way compared to RNNSLAM (since we not have access to RNNSLAM)
## main modification
I modified the data pipeline, using the gt pose and depth to generate the init_pts similar strategy used in EndoGaussians.
By the way the most important thing is that we used the real range depth in C3VD (0-100mm) on both initing point cloud, depth loss and following metrics computation.
see scene/dataset_readers.py scene/C3VD_loader.py render.py and metrics.py for further information.
## Environment
see GaussianPancakes source repo and gaussian-splatting for details.
I'd like to offer help if you need. My email is wanghao020110@hust.edu.cn
## C3VD Dataset
```
dataset_root/
│
├── depths/               # Directory containing depth maps (e.g., .png files)
│
├── images/               # Directory containing image frames
│
├── camera_poses.txt      # TUM format pose file:
│                         # timestamp tx ty tz qx qy qz qw
│
├── camera.json           # i.e. {"w": 338, "h": 270, "fx": 200.87690518518517, "fy": 200.47125, "cx": 167.31901333333332, "cy": 136.93325}
│
└── points3D.ply          # 3D points in PLY format
```

