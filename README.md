# Mapathon Calibration Tools

This package contains the tools to analysis the calibration parameters of stereo camera (probably with lidar), which is collected from icsens Mapathon 2023 dataset.
Including: 
- platform_calibration_loader.py to read the calibration parameters (interior, exterior of stereo rig and camera-to-platform transformation)
- compute_relative_stereo.py to compute relative orientation for stereo images with the matching points during the drive (refer to the algorithm by David Nistér. An efficient solution to the five-point relative pose problem. Pattern Analysis and Machine Intelligence, based on the implementation of Opencv via [this link](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html#ga1b2f149ee4b033c4dfe539f87338e243))
- rectify_stereo_images.py to rectify the stereo image with the provided interior (or intrinsic) and exterior (or extrinsic) parameters.

## Running compute relative orientation for stereo sequence:

    python compute_relative_stereo.py --on_sequence --intrinsic_file calibration_data/intrinsics.txt --extrinsic_file calibration_data/extrinsics.txt --left_image_dir /sequence/left/ --right_image_dir /sequence/right/ --start 0 --step 5 --num_samples 9 --saving_stats_file sample_stats.txt

Then, we can see how the relative orientation changes during the sequence with option ``--draw_report``, with the computed relative orientation saved in *sample_stats.txt* file
    

    python compute_relative_stereo.py --draw_report --intrinsic_file calibration_data/intrinsics.txt --extrinsic_file calibration_data/extrinsics.txt --left_image_dir /sequence/left/ --right_image_dir /sequence/right/ --start 0 --step 5 --num_samples 9 --saving_stats_file sample_stats.txt

## Running rectify image for stereo sequence:

    python rectify_stereo_images.py --src_raw_folder raw_image/ --dst_folder rectified_image/ --sampling_rate 1