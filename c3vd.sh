GPU=2
# CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/C3VD/undistorted/cecum_t1_a -m output/C3VD/cecum_t1_a --eval --port 6099
# # CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/cecum_t1_a 

# CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/C3VD/undistorted_downsize_270x338/cecum_t1_a -m output/C3VD/undistorted_downsize_270x338/cecum_t1_a  --eval --port 6070
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/cecum_t1_a --iteration 7_000 --skip_train --skip_spiral
CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m output/C3VD/undistorted_downsize_270x338/cecum_t1_a
# CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/C3VD/undistorted_downsize_270x338/cecum_t4_b -m output/C3VD/undistorted_downsize_270x338/cecum_t4_b  --eval --port 6070
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/cecum_t4_b --iteration 7_000 --skip_train
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/cecum_t4_b --iteration 15_000
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/cecum_t4_b --iteration 30_000
# CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m output/C3VD/undistorted_downsize_270x338/cecum_t4_b

# CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/C3VD/undistorted_downsize_270x338/desc_t4_a -m output/C3VD/undistorted_downsize_270x338/desc_t4_a  --eval --port 6070
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/desc_t4_a --iteration 7_000
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/desc_t4_a --iteration 15_000
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/desc_t4_a --iteration 30_000
# CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m output/C3VD/undistorted_downsize_270x338/desc_t4_a

# CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/C3VD/undistorted_downsize_270x338/trans_t1_a -m output/C3VD/undistorted_downsize_270x338/trans_t1_a  --eval --port 6070
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/trans_t1_a --iteration 7_000
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/trans_t1_a --iteration 15_000
# CUDA_VISIBLE_DEVICES=$GPU python render.py  -m output/C3VD/undistorted_downsize_270x338/trans_t1_a --iteration 30_000
# CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m output/C3VD/undistorted_downsize_270x338/trans_t1_a