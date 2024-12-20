# Reproduce the Gaussian Pancakes using gt pose and depth as an alternative way compared to RNNSLAM (since we not have access to RNNSLAM)

## C3VD Dataset

dataset_root/
│
├── depths/               # Directory containing depth maps (e.g., .png files)
│
├── images/               # Directory containing image frames
│
├── camera_poses.txt      # TUM format pose file:
│                         # timestamp tx ty tz qx qy qz qw
│
├── camera.json          
│
└── points3D.ply          # 3D points in PLY format
```

