# COLMAP Frame Processor for Neural 3D Video

This script processes multi-view video data from the [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) dataset for COLMAP reconstruction. It processes each frame independently, creating a separate COLMAP reconstruction for each timestamp across all camera views.

## Overview

The script takes synchronized multi-view video frames and performs the following operations:
- Organizes data by frame timestamps
- Runs COLMAP reconstruction for each frame independently
- Creates undistorted images and sparse reconstruction data
- Maintains a clean directory structure for further processing

## Requirements

- Python 3.7+
- COLMAP (with CUDA support recommended)
- numpy
- The [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) dataset

## Installation

1. Clone this repository:
```bash
git clone git@github.com:MrNeRF/COLMAP-Frame-Processor-for-Neural-3D-Video 
cd COLMAP-Frame-Processor-for-Neural-3D-Video
```

2. Install Python dependencies:
```bash
pip install numpy
```

3. Install COLMAP:
```bash
# Ubuntu/Debian
sudo apt-get install colmap

# For other systems, see COLMAP documentation:
# https://colmap.github.io/install.html
```

## Directory Structure

Input data should follow the Neural 3D Video dataset structure. The script will create the following structure for each frame:

```
frame_XXXX/
├── images/            # Undistorted images from all cameras
└── sparse/
    └── 0/            # COLMAP reconstruction data
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Usage

Basic usage:
```bash
# Process existing frames
python process_frames.py /path/to/scene_data

# Extract frames from videos and then process
python process_frames.py /path/to/scene_data --extract-frames

# Disable GPU usage
python process_frames.py /path/to/scene_data --no-gpu
```

### Options
- `--extract-frames`: Extract frames from MP4 videos before processing
- `--no-gpu`: Disable GPU usage in COLMAP processing

## Input Data Format

The script expects either:
1. A directory containing MP4 files (when using `--extract-frames`)
2. A directory with already extracted frames in the `images` subdirectory

When using `--extract-frames`, the script will:
1. Look for `.mp4` files in the input directory
2. Extract frames using ffmpeg with format `{camera_name}_%04d.png`
3. Place extracted frames in the `images` subdirectory
4. Proceed with COLMAP processing

## Processing Pipeline

For each frame, the script:
1. Extracts images from all camera views
2. Performs feature extraction using COLMAP
3. Matches features across views
4. Runs bundle adjustment
5. Creates undistorted images
6. Organizes output in a clean directory structure

## Citation

If you use this code, please cite the original Neural 3D Video paper:

```bibtex
@inproceedings{gao2023neural3dvideo,
  title={Neural 3D Video Synthesis from a Single Video},
  author={Gao, Chen and Rong, Gengshan and Chen, Boyi and Mu, Ye and Wang, Yang and Black, Michael J. and Szeliski, Richard},
  booktitle={CVPR},
  year={2023}
}
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This script is designed to work with the [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) dataset from Meta Research.
