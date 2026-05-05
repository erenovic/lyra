#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
    --input_image_path assets/samples \
    --sample_id 4 \
    --experiment lyra2 \
    --checkpoint_dir checkpoints/model \
    --prompt_dir assets/samples \
    --output_path outputs/zoomgs \
    --num_frames_zoom_in 81 \
    --num_frames_zoom_out 241 \
    --zoom_in_strength 0.5 \
    --zoom_out_strength 1.5 \
    --use_dmd


# PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
#   --input_image_path assets/custom_trajectory_examples/example_0/first_frame.png \
#   --trajectory_path assets/custom_trajectory_examples/example_0/trajectory.npz \
#   --experiment lyra2 \
#   --checkpoint_dir checkpoints/model \
#   --captions_path assets/custom_trajectory_examples/example_0/captions.json \
#   --num_frames 481 \
#   --output_path outputs/custom_traj

# Generate 3D Gaussian primitives based:
# - VIPE: Camera pose estimation (extrinsics, intrinsics, pcd)
# - Depth Anything 3: Depth estimation
# - Unproject pixels to 3D
# - Fit Gaussians
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
    --input_video_path outputs/zoomgs/videos/04.mp4